# -*- coding: utf-8 -*-
# @Time : 2022/11/18 | 16:11
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : SeHG.py
# Software: PyCharm
import torch
from sklearn.preprocessing import normalize
from imports import *
# from GAT_layer_v2 import GATv2Conv
import pickle
import scipy.sparse


# 规范化
def normalize_matrix(mat: sp.csr_matrix) -> sp.csr_matrix:
    normalized_mat = normalize(mat, norm='l1', axis=1)
    return normalized_mat


#
#  hg: dgl.DGLHeteroGraph, metapath: list[str], feat_dict: dict[NodeType, FloatTensor]
def aggregate_metapath_neighbors(hg, node_feature_att, metapath, feat_dict) -> FloatArray:
    etypes = set(hg.canonical_etypes)  # 返回图中的所有规范边类型,eg:{（'disease','disease-protein','disease）,.....}
    etype_map = {etype[1]: etype for etype in
                 etypes}  # 建立边map，eg:{'disease-protein':（'disease','disease-protein','disease）,.....}
    src_ntype = etype_map[metapath[0]][0]  # 一条元路径的开始节点
    dest_ntype = etype_map[metapath[-1]][2]  # 一条元路径的终止节点
    assert src_ntype == dest_ntype  # 开始节点和终止节点必须统一
    feat = feat_dict[src_ntype].cpu().numpy()  # 加载节点的特征信息

    product = None

    # 对本条元路径进行节点乘法操作和特征聚合
    for etype in metapath:
        '''
        etype:('drug', 'drug_protein', 'protein'),...
        '''
        etype = etype_map[etype]

        adj_mat = hg.adj_external(etype=etype, scipy_fmt='csr').astype(np.float32)  # 提取图中每步元路径关系的邻接矩阵
        # adj_mat = hg.adj(etype=etype,scipy_fmt='csr').astype(np.float32)  # 提取图中每步元路径关系的邻接矩阵

        normalized_adj_mat = normalize_matrix(adj_mat)
        # print('normalized_adj_mat:',normalized_adj_mat)

        if product is None:
            product = normalized_adj_mat
        else:
            # 关系矩阵乘法
            product = product.dot(normalized_adj_mat)

    # 特征聚合
    # sub_g=dgl.heterograph

    node_feature_att = node_feature_att.cpu().detach().numpy()
    out = product.dot(node_feature_att * feat)
    assert isinstance(out, ndarray) and out.dtype == np.float32 and out.shape == feat.shape

    return out


# hg: dgl.DGLHeteroGraph,infer_ntype: NodeType,feat_dict: dict[NodeType, FloatTensor],metapath_list: list[list[str]]
def pre_aggregate_neighbor(aggr_feat_list, metapath_node_feature_att_matrix, hg, infer_ntype, feat_dict, metapath_list):
    aggr_feat_list.clear()

    # 将未经聚合的初始特征加入
    raw_feat = feat_dict[infer_ntype].cpu().numpy()
    aggr_feat_list.append(raw_feat)

    for metapath, node_feature_att in zip(metapath_list, metapath_node_feature_att_matrix):
        aggr_feat = aggregate_metapath_neighbors(
            hg=hg,
            node_feature_att=node_feature_att,
            metapath=metapath,
            feat_dict=feat_dict,
        )
        aggr_feat_list.append(aggr_feat)
    return aggr_feat_list


class Multiplex_InnerProductDecoder(nn.Module):
    '''
    Decoder model for drug-disease association prediction
    '''

    def __init__(self, input_feature_dim, num_metapaths, fea_dropout=0.4):
        super(Multiplex_InnerProductDecoder, self).__init__()
        # self.inner_feature_dim = inner_feature_dim
        self.input_feature_dim = input_feature_dim
        self.num_metapaths = num_metapaths
        self.dropout = nn.Dropout(fea_dropout)

        self.weight_attn = nn.Parameter(torch.Tensor(self.num_metapaths, 1, 1))
        nn.init.normal_(self.weight_attn)

        self.weights_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.input_feature_dim, self.input_feature_dim)) for _ in
             range(self.num_metapaths)])

    def forward(self, drug_fea_tensor, disease_fea_tensor):
        out_prediction_list = [drug_fea @ (weight(disease_fea)).T for drug_fea, weight, disease_fea in
                               zip(drug_fea_tensor, self.weights_list, disease_fea_tensor)]

        out_prediction_list = torch.stack(out_prediction_list)
        weight_attn = F.softmax(self.weight_attn, dim=0)

        out_predition = torch.sum(out_prediction_list * weight_attn, dim=0)
        return out_prediction_list, out_predition


class SeHG_bio(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_metapaths: int):
        super(SeHG_bio, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = 0.4
        self.feature_node_name = ['drug', 'disease']

        # 结点未经聚合的原始特征，作为一个特殊的元路径聚合结果
        self.num_metapaths = num_metapaths + 1
        # self.num_metapaths = num_metapaths

        # node feature attention params matrix
        self.drug_metapath_node_feature_att_matrix = nn.Parameter(torch.ones(self.num_metapaths, 894, 1))
        nn.init.uniform_(self.drug_metapath_node_feature_att_matrix)
        self.disease_metapath_node_feature_att_matrix = nn.Parameter(torch.ones(self.num_metapaths, 454, 1))
        nn.init.uniform_(self.disease_metapath_node_feature_att_matrix)

        # collection meta_path feature aggr list
        self.aggr_drug_feat_list = []
        self.aggr_disease_feat_list = []

        # MLP for drug feature_projection
        self.drug_feature_projector_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, hidden_dim),
            )
            for _ in range(self.num_metapaths)
        ])
        # MLP for disease feature_projection
        self.disease_feature_projector_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, hidden_dim),
            )
            for _ in range(self.num_metapaths)
        ])

        # transformer for feature semantic aggregation
        self.Q_drug = nn.Linear(hidden_dim, hidden_dim)
        self.K_drug = nn.Linear(hidden_dim, hidden_dim)
        self.V_drug = nn.Linear(hidden_dim, hidden_dim)
        self.beta_drug = Parameter(torch.ones(1))

        self.Q_disease = nn.Linear(hidden_dim, hidden_dim)
        self.K_disease = nn.Linear(hidden_dim, hidden_dim)
        self.V_disease = nn.Linear(hidden_dim, hidden_dim)
        self.beta_disease = Parameter(torch.ones(1))

        # prediction
        self.prediction_fun = Multiplex_InnerProductDecoder(self.hidden_dim, self.num_metapaths)

    def forward(self, g, feature, metapath_list) -> FloatTensor:
        # 1. Simplified Neighbor Aggregation

        # for drug feature aggregation
        self.aggr_drug_feat_list = pre_aggregate_neighbor(self.aggr_drug_feat_list,
                                                          self.drug_metapath_node_feature_att_matrix, g,
                                                          self.feature_node_name[0],
                                                          feature,
                                                          metapath_list[0])

        # for disease feature aggregation
        self.aggr_disease_feat_list = pre_aggregate_neighbor(self.aggr_disease_feat_list,
                                                             self.disease_metapath_node_feature_att_matrix, g,
                                                             self.feature_node_name[1],
                                                             feature,
                                                             metapath_list[1])

        assert len(self.aggr_drug_feat_list) == self.num_metapaths and len(
            self.aggr_disease_feat_list) == self.num_metapaths

        drug_num_nodes, disease_num_nodes = len(self.aggr_drug_feat_list[0]), len(self.aggr_disease_feat_list[0])

        # 2. Multi-layer Feature Projection
        assert len(self.aggr_drug_feat_list) == len(self.drug_feature_projector_list) and len(
            self.aggr_disease_feat_list) == len(self.disease_feature_projector_list)

        # print("aggr_drug_feat_list[0].size:", self.aggr_drug_feat_list[0].shape)
        # print("aggr_drug_feat_list[1].size:", self.aggr_drug_feat_list[1].shape)
        # print("aggr_disease_feat_list[0].size:", self.aggr_disease_feat_list[0].shape)
        # print("aggr_disease_feat_list[1].size:", self.aggr_disease_feat_list[1].shape)

        drug_proj_list = [
            proj(
                torch.from_numpy(feat).cuda()
            )
            for feat, proj in zip(self.aggr_drug_feat_list, self.drug_feature_projector_list)
        ]

        disease_proj_list = [
            proj(
                torch.from_numpy(feat).cuda()
            )
            for feat, proj in zip(self.aggr_disease_feat_list, self.disease_feature_projector_list)
        ]
        # print('drug_proj_list:',drug_proj_list)
        # Turn into tensor
        drug_proj = torch.stack(drug_proj_list)
        disease_proj = torch.stack(disease_proj_list)
        # print('drug_proj.size:', drug_proj.size())
        # print('disease_proj.size:', disease_proj.size())
        assert drug_proj.shape == (self.num_metapaths, drug_num_nodes, self.hidden_dim) and disease_proj.shape == (
            self.num_metapaths, disease_num_nodes, self.hidden_dim)

        drug_proj = drug_proj.transpose(0, 1)
        disease_proj = disease_proj.transpose(0, 1)
        # print('drug_proj.size:', drug_proj.size())
        # print('disease_proj.size:', disease_proj.size())

        assert drug_proj.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim) and disease_proj.shape == (
            disease_num_nodes, self.num_metapaths, self.hidden_dim)

        # 3. Transformer-based Semantic Aggregation
        Q_drug = self.Q_drug(drug_proj)
        K_drug = self.K_drug(drug_proj)
        V_drug = self.V_drug(drug_proj)
        assert Q_drug.shape == K_drug.shape == V_drug.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim)

        Q_disease = self.Q_disease(disease_proj)
        K_disease = self.K_disease(disease_proj)
        V_disease = self.V_disease(disease_proj)
        assert Q_disease.shape == K_disease.shape == V_disease.shape == (
            disease_num_nodes, self.num_metapaths, self.hidden_dim)

        # for drug transformer operation
        attn_drug = Q_drug @ (K_drug.transpose(1, 2))
        assert attn_drug.shape == (drug_num_nodes, self.num_metapaths, self.num_metapaths)

        attn_drug = torch.softmax(attn_drug, dim=-1)

        attn_out_drug = self.beta_drug * (attn_drug @ drug_proj) + drug_proj
        assert attn_out_drug.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim)

        attn_out_drug = attn_out_drug.view(self.num_metapaths, drug_num_nodes, self.hidden_dim)

        # for disease transformer operation
        attn_disease = Q_disease @ (K_disease.transpose(1, 2))
        assert attn_disease.shape == (disease_num_nodes, self.num_metapaths, self.num_metapaths)

        attn_disease = torch.softmax(attn_disease, dim=-1)

        attn_out_disease = self.beta_disease * (attn_disease @ disease_proj) + disease_proj
        assert attn_out_disease.shape == (disease_num_nodes, self.num_metapaths, self.hidden_dim)

        attn_out_disease = attn_out_disease.view(self.num_metapaths, disease_num_nodes, self.hidden_dim)

        out_list, out_prediction = self.prediction_fun(attn_out_drug, attn_out_disease)
        assert out_prediction.shape == (drug_num_nodes, disease_num_nodes)

        return out_prediction

# if __name__ == '__main__':
#     # meta_paths
#     DRUG_METAPATH_LIST = [
#         ['drug_drug'],
#         ['drug_protein', 'protein_drug'],
#         ['drug_disease', 'disease_drug'],
#         ['drug_disease', 'disease_pathway', 'pathway_disease', 'disease_drug']
#         # ['drug_drug', 'drug_protein', 'protein_drug'],
#         # ['drug_protein', 'protein_drug', 'drug-drug'],
#         # ['drug_drug', 'drug_disease', 'disease_drug'],
#         # ['drug_disease', 'disease_drug', 'drug_drug'],
#         # ['drug_drug', 'drug_protein', 'protein_gene', 'gene_protein', 'protein_drug'],
#     ]
#     DISEASE_METAPATH_LIST = [
#         ['disease_disease'],
#         ['disease_drug', 'drug_disease'],
#         ['disease_pathway', 'pathway_disease'],
#         ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
#         # ['disease_disease', 'disease_pathway', 'pathway_disease'],
#         # ['disease_disease', 'disease_drug', 'drug_disease'],
#         # ['disease_disease', 'disease_pathway', 'pathway_gene', 'gene_pathway', 'pathway_disease']
#     ]
#     # model core params
#     HYPER_PARAM = dict(
#         drug_metapath_list=DRUG_METAPATH_LIST,
#         disease_metapath_list=DISEASE_METAPATH_LIST,
#         hidden_dim=64,
#         num_epochs=200,
#         lr=0.001,
#         weight_decay=0.,
#     )
#
#     # load hete_graph
#     f = open('./heterograph/no_gene_graph.dgl.pkl', 'rb')
#     g = pickle.load(f)
#
#     feature = {'drug': g.nodes['drug'].data['h'], 'disease': g.nodes['disease'].data['h']}
#
#     in_feature_size = feature['drug'].shape[1]
#
#     metapath_list = [HYPER_PARAM['drug_metapath_list'], HYPER_PARAM['disease_metapath_list']]
#
#     model = SeHG_bio(in_dim=in_feature_size, hidden_dim=256, out_dim=128, num_metapaths=len(metapath_list[0]))
#
#     out_list, out_prediction = model(g, feature, metapath_list)
#
#     print('out_list.len:', len(out_list))
#     print('out_predition.size:', out_prediction.size())

# print('drug_proj_list[0].size:', drug_proj_list[0].shape)
# print('drug_proj_list[1].size:', drug_proj_list[1].shape)
# print('disease_proj_list[0].size:', disease_proj_list[0].shape)
# print('disease_proj_list[1].size:', disease_proj_list[1].shape)
# print('attn_out_drug.size:', attn_out_drug.size())
# print('attn_out_disease.size:', attn_out_disease.size())
