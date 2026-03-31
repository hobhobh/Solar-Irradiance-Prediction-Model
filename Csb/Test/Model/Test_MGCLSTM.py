# -*- coding: utf-8 -*-
"""
多图 AGCLSTM 测试脚本（严格对齐论文公式）
- 加载训练好的 best_agclstm_multi_graph.pt
- 评估测试集整体与逐站点性能
"""

import os, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 添加上级目录，确保能导入 Train/Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Train.Model.Train_MGCLSTM import (
    MGCLSTM, MultiSiteSeqDataset,
    FEATURE_COLUMNS, TARGET_COL, read_station_csvs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 数据加载 =====================
def load_all_sites(root_folder):
    site_dirs = [d for d in sorted(os.listdir(root_folder)) if os.path.isdir(os.path.join(root_folder, d))]
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
    return Y, site_names, common_index

# ===================== 模型加载 =====================
def load_trained_model(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    config = state["config"]
    model = MGCLSTM(
        num_nodes=config["num_nodes"],
        in_feats=config["in_feats"],
        static_adj=state.get("static_adj", None),
        gcn_hidden=64,
        lstm_hidden=64,
        lstm_layers=2,
        mlp_dims=[256, 128, 64],
        dropout=0.2
    ).to(device)
    model.load_state_dict(state["model"])
    scalers = state["scalers"]
    return model, scalers, config

# ===================== 评估函数 =====================
def evaluate(model, loader, scaler_y):
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            yhat = model(Xb)
            # 反标准化
            yh = scaler_y.inverse_transform(
                yhat.cpu().numpy().reshape(-1, 1)
            ).reshape(yhat.shape[0], -1)
            y  = scaler_y.inverse_transform(
                yb.cpu().numpy().reshape(-1, 1)
            ).reshape(yb.shape[0], -1)
            ys.append(y); yhats.append(yh)
    Y, YH = np.concatenate(ys, 0), np.concatenate(yhats, 0)
    rmse = math.sqrt(mean_squared_error(Y.reshape(-1), YH.reshape(-1)))
    mae  = mean_absolute_error(Y.reshape(-1), YH.reshape(-1))
    r    = float(np.corrcoef(Y.reshape(-1), YH.reshape(-1))[0, 1])
    return rmse, mae, r, Y, YH

# ===================== 主测试流程 =====================
def test_agclstm(test_root, ckpt_path, seq_len=48, batch_size=512):
    print(f"加载模型: {ckpt_path}")
    model, scalers, config = load_trained_model(ckpt_path)

    # 加载测试数据
    Y_all, site_names, common_index = load_all_sites(test_root)
    target_idx = FEATURE_COLUMNS.index(TARGET_COL)
    test_ds = MultiSiteSeqDataset(Y_all, seq_len, target_idx, scalers, False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 整体性能
    rmse, mae, r, Y, YH = evaluate(model, test_loader, scalers["y"])
    y_flat = Y.reshape(-1)
    nrmse = rmse / (np.max(y_flat) - np.min(y_flat)) * 100   # range-normalized
    nmae  = mae  / np.mean(y_flat) * 100                    # mean-normalized
    print("\n===== 测试集整体性能 =====")
    print(f"综合RMSE: {rmse:.3f}")
    print(f"综合MAE:  {mae:.3f}")
    print(f"综合相关系数 R: {r:.4f}")
    print(f"综合nRMSE: {nrmse:.3f}%")
    print(f"综合nMAE : {nmae:.3f}%")

    # 逐站点性能
    print("\n===== 各站点性能 =====")
    for i, site in enumerate(site_names):
        y_true = Y[:, i]; y_pred = YH[:, i]
        rmse_i = math.sqrt(mean_squared_error(y_true, y_pred))
        mae_i  = mean_absolute_error(y_true, y_pred)
        nrmse_i = rmse_i / (np.max(y_true) - np.min(y_true)) * 100   # range-normalized
        nmae_i  = mae_i  / np.mean(y_true) * 100                    # mean-normalized
        print(f"{site}: RMSE={rmse_i:.3f}, MAE={mae_i:.3f}, "
              f"nRMSE={nrmse_i:.2f}%, nMAE={nmae_i:.2f}%")

# ===================== 主入口 =====================
if __name__ == "__main__":
    TEST_ROOT = "./Test/Oregon"   # 修改为你的测试集路径
    CKPT = "./Train/agclstm_multi_graph_ckpt_MG/best_agclstm_multi_graph.pt"
    test_agclstm(TEST_ROOT, CKPT)
