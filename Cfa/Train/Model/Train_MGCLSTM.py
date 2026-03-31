import os, glob, math, time, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===================== 基础设置 =====================
def set_seed(seed: int = 43):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(343)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TIME_COLS = ["Year","Month","Day","Hour","Minute"]
FEATURE_COLUMNS = [
    "GHI","Clearsky GHI","Cloud Type","Dew Point","Solar Zenith Angle",
    "Relative Humidity","Precipitable Water","Wind Speed","Wind Direction","Temperature",
]
TARGET_COL = "GHI"
CANONICAL_MAP = {
    "year":"Year","month":"Month","day":"Day","hour":"Hour","minute":"Minute",
    "ghi":"GHI","clearsky ghi":"Clearsky GHI","cloud type":"Cloud Type","dew point":"Dew Point",
    "solar zenith angle":"Solar Zenith Angle","relative humidity":"Relative Humidity",
    "precipitable water":"Precipitable Water","wind speed":"Wind Speed","wind direction":"Wind Direction",
    "temperature":"Temperature",
}
def _clean_columns(cols): return [str(c).replace("\ufeff","").strip() for c in cols]
def _canonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [CANONICAL_MAP.get(c.replace("\ufeff","").strip().lower(), c.replace("\ufeff","").strip())
                  for c in df.columns]
    return df

def read_station_csvs(station_dir: str) -> pd.DataFrame:
    frames = []
    csv_paths = sorted(glob.glob(os.path.join(station_dir, "*.csv")))
    if not csv_paths: raise ValueError(f"站点目录为空：{station_dir}")
    for csv in csv_paths:
        df = pd.read_csv(csv, skiprows=2, engine="python")
        df.columns = _clean_columns(df.columns); df = _canonize_columns(df)
        need_cols = TIME_COLS + FEATURE_COLUMNS
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv} 缺少必要列: {missing}")
        df = df[need_cols].copy()
        # 数值化
        for c in FEATURE_COLUMNS: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.interpolate(method="linear", limit_direction="forward") 
        df = df.fillna(method="ffill")  

        # 时间索引
        dt = pd.to_datetime(df[["Year","Month","Day","Hour","Minute"]], errors="coerce")
        df.index = dt
        df = df.drop(columns=TIME_COLS).sort_index()
        df = df[~df.index.isna()]
        frames.append(df)
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out

def load_all_sites(root_folder: str) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex, Dict[str, Tuple[float, float]]]:
    site_dirs = [d for d in sorted(os.listdir(root_folder)) if os.path.isdir(os.path.join(root_folder, d))]
    if not site_dirs: raise ValueError(f"未找到站点文件夹：{root_folder}")

    lat_lon = {}
    for s in site_dirs:
        csv_paths = glob.glob(os.path.join(root_folder, s, "*.csv"))
        csv_name = os.path.basename(csv_paths[0])
        parts = csv_name.replace(".csv", "").split("_")
        lat = float(parts[-3]); lon = float(parts[-2])
        lat_lon[s] = (lat, lon)

    site_dfs = {s: read_station_csvs(os.path.join(root_folder, s)) for s in site_dirs}
    common_index = None
    for df in site_dfs.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    site_names = sorted(site_dfs.keys())
    T, N, F = len(common_index), len(site_names), len(FEATURE_COLUMNS)
    Y = np.zeros((T, N, F), dtype=np.float32)
    for j, s in enumerate(site_names):
        sub = site_dfs[s].loc[common_index, FEATURE_COLUMNS]
        Y[:, j, :] = sub.values.astype(np.float32)

    return Y, site_names, common_index, lat_lon

# ===================== 邻接矩阵（核心修改：严格对称归一化） =====================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # 地球半径（公里）
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def normalize_adj(A: np.ndarray) -> np.ndarray:
    """严格实现 Kipf & Welling (2016) 的对称归一化：
    \tilde{A} = A + I（添加自环）
    \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}（对称归一化）
    """
    N = A.shape[0]
    # 1. 添加自环
    A_with_self_loop = A + np.eye(N, dtype=np.float32)  # \tilde{A} = A + I
    # 2. 计算度矩阵 D（每行求和）
    degree = np.sum(A_with_self_loop, axis=1, keepdims=False)  # 形状 (N,)
    # 3. 避免除零（处理孤立节点）
    degree[degree == 0] = 1e-6
    # 4. 对称归一化：D^{-1/2} * A * D^{-1/2}
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree))  # D^{-1/2}
    A_normalized = degree_inv_sqrt @ A_with_self_loop @ degree_inv_sqrt  # 矩阵乘法
    return A_normalized

def compute_multigraph_adjacencies(Y, site_names, lat_lon, sigma=1.0):
    N = len(site_names)
    # 1. 距离图（静态）
    A_dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        lat1, lon1 = lat_lon[site_names[i]]
        for j in range(N):
            if i != j:  # 初始不包含自环（归一化时统一添加）
                lat2, lon2 = lat_lon[site_names[j]]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                A_dist[i, j] = np.exp(-(dist**2) / (2 * sigma**2))  # 距离衰减权重
    A_dist = normalize_adj(A_dist)  # 应用 Kipf 归一化（含自环）

    # 2. GHI相似度图（静态）
    ghi_idx = FEATURE_COLUMNS.index("GHI")
    ghi_series = Y[:, :, ghi_idx]  # 形状 (T, N)
    A_sim = np.corrcoef(ghi_series.T)  # 计算站点间GHI相关性（N×N）
    A_sim = np.nan_to_num(A_sim, nan=0.0)  # 处理NaN
    np.fill_diagonal(A_sim, 0.0)  # 初始不包含自环（归一化时统一添加）
    A_sim = normalize_adj(np.abs(A_sim))  # 应用 Kipf 归一化（含自环）
    
    return {"distance": A_dist, "similarity": A_sim}

# ===================== 数据集 =====================
class MultiSiteSeqDataset(Dataset):
    def __init__(self, Y, seq_len, target_col_idx, scalers, fit_scaler):
        self.seq_len = seq_len
        self.T, self.N, self.F = Y.shape
        X_all, y_all = Y.copy(), Y[:,:,target_col_idx].copy()
        self.scalers = scalers if scalers else {}
        if fit_scaler:
            self.scalers["x"] = StandardScaler().fit(X_all.reshape(-1, self.F))
            self.scalers["y"] = StandardScaler().fit(y_all.reshape(-1, 1))
        X_all = self.scalers["x"].transform(X_all.reshape(-1, self.F)).reshape(self.T, self.N, self.F)
        y_all = self.scalers["y"].transform(y_all.reshape(-1, 1)).reshape(self.T, self.N)
        Xs, ys = [], []
        for t in range(self.seq_len, self.T-1):
            Xs.append(X_all[t-self.seq_len:t]); ys.append(y_all[t+1])
        self.X, self.y = np.stack(Xs), np.stack(ys)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)

# ===================== 模型（核心修改：自适应图归一化） =====================
class AdaptiveAdjacency(nn.Module):
    def __init__(self, in_feats, adj_hidden=24):
        super().__init__()
        self.fc = nn.Linear(in_feats, adj_hidden)  # 论文中的24神经元全连接层

    def forward(self, xt):
        """
        生成自适应邻接矩阵并应用 Kipf 归一化，同时保留原论文对称化逻辑
        xt: 输入特征，形状 (B, N, F) 或 (B*L, N, F)
        """
        B, N, _ = xt.shape  # B为批量大小（或批量×时间步），N为站点数
        # 1. 论文公式：特征映射→ReLU→内积→对称化
        Z = F.relu(self.fc(xt))  # (B, N, 24)
        A = torch.matmul(Z, Z.transpose(1, 2))  # (B, N, N)，内积计算相似度
        A = 0.5 * (F.softmax(A, dim=-1) + F.softmax(A, dim=-2))  # 对称化（行/列softmax平均）
        
        # 2. 应用 Kipf & Welling 归一化（含自环）
        # 2.1 添加自环：\tilde{A} = A + I
        A_with_self_loop = A + torch.eye(N, device=xt.device).unsqueeze(0)  # (B, N, N)
        # 2.2 计算度矩阵 D（每行求和）
        degree = torch.sum(A_with_self_loop, dim=-1)  # (B, N)
        # 2.3 避免除零
        degree = torch.clamp(degree, min=1e-6)
        # 2.4 对称归一化：D^{-1/2} * A * D^{-1/2}
        degree_inv_sqrt = torch.diag_embed(torch.pow(degree, -0.5))  # (B, N, N)，对角线为D^{-1/2}
        A_normalized = torch.bmm(torch.bmm(degree_inv_sqrt, A_with_self_loop), degree_inv_sqrt)  # 批量矩阵乘法
        
        return A_normalized

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)  # 特征线性变换

    def forward(self, A, X):
        """一阶GCN：ReLU(A * WX)"""
        return F.relu(torch.bmm(A, self.lin(X)))  # A: (B, N, N), X: (B, N, F) → 输出 (B, N, out_feats)

class MultiGraphSpatialEncoder(nn.Module):
    def __init__(self, in_feats, static_adj, gcn_hidden=64, gcn_layers=2):
        super().__init__()
        # 静态图邻接矩阵（已归一化，转为Tensor）
        self.static_adj = {
            k: torch.tensor(v, dtype=torch.float32, requires_grad=False) 
            for k, v in static_adj.items()
        }
        # 自适应图生成器
        self.adaptive_gen = AdaptiveAdjacency(in_feats)
        # 每个图分支独立的GCN层
        self.gcns = nn.ModuleDict({
            name: nn.ModuleList([
                GCNLayer(in_feats if i == 0 else gcn_hidden, gcn_hidden) 
                for i in range(gcn_layers)
            ]) for name in list(static_adj.keys()) + ["adaptive"]
        })
        # 注意力融合多图特征
        self.attn = nn.Sequential(nn.Linear(gcn_hidden, 1), nn.Softmax(dim=1))

    def forward(self, xt):
        """
        xt: 输入特征，形状 (B*L, N, F)（展平的时序特征）
        输出：融合后的空间特征 (B*L, N, gcn_hidden)
        """
        B_L, N, F = xt.shape  # B_L = B*L（批量×时间步）
        feats = []  # 存储各图分支的特征

        # 1. 静态图分支（距离图和相似度图）
        for name, adj in self.static_adj.items():
            Z = xt  # 初始特征
            # 扩展静态邻接矩阵到批量维度：(1, N, N) → (B_L, N, N)
            A = adj.to(xt.device).unsqueeze(0).repeat(B_L, 1, 1)
            # 多层GCN
            for gcn in self.gcns[name]:
                Z = gcn(A, Z)
            feats.append(Z.unsqueeze(1))  # 增加图分支维度：(B_L, 1, N, gcn_hidden)

        # 2. 自适应图分支
        A_adap = self.adaptive_gen(xt)  # (B_L, N, N)，已归一化
        Z = xt
        for gcn in self.gcns["adaptive"]:
            Z = gcn(A_adap, Z)
        feats.append(Z.unsqueeze(1))  # (B_L, 1, N, gcn_hidden)

        # 3. 注意力融合
        fused = torch.cat(feats, dim=1)  # (B_L, 3, N, gcn_hidden)，3个图分支
        weights = self.attn(fused)  # (B_L, 3, N, 1)，每个图分支的权重
        out = torch.sum(fused * weights, dim=1)  # (B_L, N, gcn_hidden)，加权融合

        return out

class SharedMLP(nn.Module):
    def __init__(self, input_dim, mlp_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + mlp_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
            if i < len(dims) - 2:  # 最后一层前添加Dropout
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(mlp_dims[-1], 1)  # 输出预测值

    def forward(self, x):
        return self.out(self.mlp(x)).squeeze(-1)

class MGCLSTM(nn.Module):
    def __init__(self, num_nodes, in_feats, static_adj,
                 gcn_hidden=64, lstm_hidden=64, lstm_layers=2,
                 mlp_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.N = num_nodes
        # 空间编码器（多图融合）
        self.spatial = MultiGraphSpatialEncoder(in_feats, static_adj, gcn_hidden)
        # 每个站点独立的LSTM时间编码器
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=gcn_hidden,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.0  # 原论文LSTM无dropout
            ) for _ in range(num_nodes)
        ])
        # 共享MLP预测层
        self.mlp = SharedMLP(lstm_hidden, mlp_dims, dropout)

    def forward(self, x):
        """
        x: 输入序列，形状 (B, L, N, F)，其中 B=批量，L=序列长度，N=站点数，F=特征数
        输出：预测值，形状 (B, N)
        """
        B, L, N, F = x.shape
        # 1. 空间编码：展平时间步→提取空间特征→恢复时序结构
        z = self.spatial(x.reshape(B*L, N, F)).reshape(B, L, N, -1)  # (B, L, N, gcn_hidden)
        
        # 2. 时间编码：每个站点独立LSTM
        preds = []
        for i in range(N):
            # LSTM输入：(B, L, gcn_hidden)，取最后一层隐藏状态
            _, (hn, _) = self.lstms[i](z[:, :, i, :])  # hn形状 (lstm_layers, B, lstm_hidden)
            lstm_out = hn[-1]  # 取最后一层输出：(B, lstm_hidden)
            preds.append(self.mlp(lstm_out))  # 每个站点的预测值：(B,)
        
        # 3. 聚合所有站点预测
        return torch.stack(preds, dim=1)  # (B, N)

# ===================== 训练 =====================
def evaluate(model, loader, scaler_y):
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            yhat = model(Xb)
            # 反标准化
            yh = scaler_y.inverse_transform(yhat.cpu().numpy().reshape(-1, 1)).reshape(yhat.shape)
            y = scaler_y.inverse_transform(yb.cpu().numpy().reshape(-1, 1)).reshape(yb.shape)
            ys.append(y)
            yhats.append(yh)
    Y, YH = np.concatenate(ys), np.concatenate(yhats)
    return (
        math.sqrt(mean_squared_error(Y, YH)),
        mean_absolute_error(Y, YH),
        np.corrcoef(Y.reshape(-1), YH.reshape(-1))[0, 1]
    )

def train_agclstm(root_folder, seq_len=72, batch_size=512, epochs=80, lr=1e-3,
                  val_ratio=0.1, sigma=15.0, save_dir="./ckpt"):
    os.makedirs(save_dir, exist_ok=True)
    # 加载数据和静态图
    Y, site_names, idx, latlon = load_all_sites(root_folder)
    static_adj = compute_multigraph_adjacencies(Y, site_names, latlon, sigma)
    # 时间切分训练/验证集
    T = Y.shape[0]
    split = int(T * (1 - val_ratio))
    Y_tr, Y_val = Y[:split], Y[split:]
    target_idx = FEATURE_COLUMNS.index(TARGET_COL)
    # 初始化数据集
    train_ds = MultiSiteSeqDataset(Y_tr, seq_len, target_idx, None, True)
    scalers = train_ds.scalers
    val_ds = MultiSiteSeqDataset(Y_val, seq_len, target_idx, scalers, False)
    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    # 初始化模型
    model = MGCLSTM(
        num_nodes=len(site_names),
        in_feats=Y.shape[2],
        static_adj=static_adj
    ).to(device)
    # 优化器和损失函数
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.MSELoss()
    # 训练循环
    best_rmse = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        t0 = time.time()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(Xb)
            loss = crit(yhat, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * Xb.size(0)
        # 学习率衰减
        sched.step()
        # 评估
        train_mse = loss_sum / len(train_ds)
        rmse, mae, r = evaluate(model, val_loader, scalers["y"])
        print(
            f"Epoch {ep}/{epochs} | Train MSE {train_mse:.4f} | "
            f"Val RMSE {rmse:.3f} | Val MAE {mae:.3f} | R {r:.3f} | "
            f"Time {time.time()-t0:.1f}s"
        )
        # 保存最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "model": model.state_dict(),
                "scalers": scalers,
                "static_adj": static_adj,
                "config": {
                    "seq_len": seq_len, "feature_columns": FEATURE_COLUMNS,
                    "target_col": TARGET_COL, "site_names": site_names,
                    "num_nodes": len(site_names), "in_feats": Y.shape[2]
                }
            }, os.path.join(save_dir, "best_agclstm_multi_graph.pt"))
    print(f"训练完成，最佳RMSE={best_rmse:.3f}")
    return os.path.join(save_dir, "best_agclstm_multi_graph.pt")

# ===================== 主入口 =====================
if __name__ == "__main__":
    TRAIN_ROOT = "./Train/Oklahoma"  
    train_agclstm(TRAIN_ROOT, save_dir="./Train/agclstm_multi_graph_ckpt_MG")