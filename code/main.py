import random
import gc
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from gcn import Model
import json

from sklearn.model_selection import KFold
# set_seed(666)
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=Warning)

def main(edge_idx_dict, n_drug, n_cir, drug_sim, cir_sim, args_config, device):
    lr = args_config['lr']
    weight_decay = args_config['weight_decay']
    kfolds = args_config['kfolds']
    num_epoch = args_config['num_epoch']
    knn_nums = args_config['knn_nums']
    pos_edges = edge_idx_dict['pos_edges']
    neg_edges = edge_idx_dict['neg_edges']
    metrics_tensor = np.zeros((1, 7))
    temp_drug_cir = np.zeros((n_drug, n_cir))
    temp_drug_cir[pos_edges[0], pos_edges[1]] = 1

    # def set_row_to_zero(matrix, row_index):
    #     matrix[row_index] = [0] * len(matrix[row_index])
    # set_row_to_zero(temp_drug_cir, 133)

    embd_size = args_config["embd_size"]



    # get_all_samples里面要加上随机种子
    seed = args_config["seed"]
    random.seed(seed)
    #   get_all_samples很强大的正负样本坐标提取函数！
    samples = get_all_samples(temp_drug_cir)
    metric = np.zeros((1, 7))
    auc_val =0
    #  5折交叉验证
    # end: define K-fold sample divider
    fold = 0
    print("------this is %dth cross validation------" % (fold))

    drug_sim_t, cir_sim_t = get_syn_sim(temp_drug_cir, drug_sim, cir_sim, 1)    # 融合了高斯核相似性
    drug_adj = k_matrix(drug_sim_t, knn_nums)   # drug的邻接矩阵
    cir_adj = k_matrix(cir_sim_t, knn_nums)
    edge_idx_drug, edge_idx_cir = np.array(tuple(np.where(drug_adj != 0))), np.array(tuple(np.where(cir_adj != 0)))
    edge_idx_drug = torch.tensor(edge_idx_drug, dtype=torch.long).to(device=device)    # drug的前25个边（x，y）
    edge_idx_cir = torch.tensor(edge_idx_cir, dtype=torch.long).to(device=device)
    # model = MNGACDA(
    #     n_drug + n_cir, num_hidden_layers, num_embedding_features, num_heads_per_layer,
    #     n_drug, n_cir, add_layer_attn, residual).to(device)

    het_mat = construct_het_mat(temp_drug_cir, cir_sim_t, drug_sim_t)  # 创建了一个大的特征矩阵
    adj_mat = construct_adj_mat(temp_drug_cir)
    drug_sim_in = torch.tensor(drug_sim_t, dtype=torch.float).to(device=device)
    cir_sim_in = torch.tensor(cir_sim_t, dtype=torch.float).to(device=device)
    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long).to(device=device)
    het_mat_device = torch.tensor(het_mat, dtype=torch.float32).to(device=device)

    # model = Net(het_mat_device, edge_idx_device, 64, n_drug, 64)

    model = Model(het_mat_device, edge_idx_device, drug_sim_in, edge_idx_drug, cir_sim_in, edge_idx_cir,
                  embd_size, hyperparam_dict["walk_embds"], hyperparam_dict["atten_head"], hyperparam_dict["end_embd"], device).to(device)
    num_u, num_v = n_drug, n_cir

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epoch):
        model.zero_grad()
        # Forward
        pred_mat = model(het_mat_device, drug_sim_in, cir_sim_in, edge_idx_device).reshape(
            num_u, num_v).to(device)
        # loss = calculate_loss(pred_mat, pos_edges, neg_edges)       # 错了
        loss_fun = torch.nn.BCELoss(reduction='mean')

        # 训练样本的坐标（从大矩阵中取值）
        n = np.array(samples).T
        m = n[:2, :]
        pre = pred_mat[tuple(m)]

        train_m = torch.tensor(temp_drug_cir, dtype=torch.float).to(device)
        loss = loss_fun(pre, train_m[tuple(m)])
        # loss = loss_fun(pred_mat, train_m)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print('------EOPCH {} of {}------'.format(epoch + 1, args_config['num_epoch']))
            print('Loss: {}'.format(loss))

    model.eval()
    with torch.no_grad():
        pred_mat = model(het_mat_device, drug_sim_in, cir_sim_in, edge_idx_device).cpu().reshape(num_u, num_v)
        predict_y_proba = pred_mat.reshape(n_drug, n_cir).detach().numpy()


# 保存DataFrame为CSV文件
def save_cvs(predict_y_proba, file_path):
    df = pd.DataFrame(predict_y_proba)
    try:
        df.to_csv(file_path, index=True, header=True)
        print("文件保存成功:", file_path)
    except Exception as e:
        print("保存文件时出现错误:", str(e))

def save_xlsx(matrix, file_path):
    # 创建 DataFrame 对象
    df = pd.DataFrame(matrix)
    # 将 DataFrame 保存为 Excel 文件
    try:
        df.to_excel(file_path, index=True, header=True)
        print("文件保存成功:", file_path)
    except Exception as e:
        print("保存文件时出现错误:", str(e))

if __name__ == '__main__':
    # set_seed(666)
    repeat_times = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddd = 128

    hyperparam_dict = {
        'kfolds':5,
        'num_epoch': 1000,
        'knn_nums':25,
        'lr': 0.001,
        'weight_decay': 5e-3,
        'add_layer_attn': True,
        'residual': True,
        "seed": 589,
        # 1234 ,123, 925(9069)
        "atten_head": 1,
        "embd_size": ddd,
        "end_embd": ddd,
        "walk_embds": 128,
        "max_seed": 0,
        "max_auc": 0
    }

    for i in range(repeat_times):
        sed = 589
        set_seed(sed)
        print(f'********************{i + 1} of {repeat_times}********************')
        hyperparam_dict["seed"] = sed
        drug_sim, cir_sim, edge_idx_dict, drug_dis_matrix = load_data()
        diag = np.diag(cir_sim)
        if np.sum(diag) != 0:
            cir_sim = cir_sim - np.diag(diag)
        diag = np.diag(drug_sim)
        if np.sum(diag) != 0:
            drug_sim = drug_sim - np.diag(diag)
        pred_mat=main(edge_idx_dict, drug_sim.shape[0], cir_sim.shape[0], drug_sim, cir_sim, hyperparam_dict, device)
        ss = hyperparam_dict["max_seed"]
        best_auc = hyperparam_dict["max_auc"]
        print(f"the best seed: {ss}, the auc is {best_auc}")
        print("\n\n")