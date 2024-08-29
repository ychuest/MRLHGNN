# -*- coding: utf-8 -*-
# @Time : 2022/11/12 | 20:55
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : load_data.py
# Software: PyCharm

import numpy as np
import torch
import torch as th
import dgl
import warnings
import pandas as pd
import pickle

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


def data_loader():
    drug_protein = np.load('./base_meta_path/drug--protein.npy')  # (894,18877)
    protein_drug = drug_protein.transpose()  # (18877,984)

    disease_pathway = np.load('./base_meta_path/disease--pathway.npy')  # (453,314)
    pathway_disease = disease_pathway.transpose()  # (314,453)

    drug_simi_drug = np.load('./base_meta_path/drug--drug.npy')  # (894,894)
    disease_disease = np.load('./base_meta_path/disease--disease.npy')  # (454,454)
    protein_protein = np.load('./base_meta_path/protein--protein.npy')  # (18877,18877)
    pathway_pathway = np.load('./base_meta_path/pathway--pathway.npy')  # (314,314)

    protein_pathway = np.load('./base_meta_path/protein--pathway.npy')  # (18877,314)
    pathway_protein = np.load('./base_meta_path/pathway--protein.npy')  # (314,18877)

    drug_disease = np.load('./base_meta_path/drug--disease.npy')  # (894,454)

    drug_num, protein_num, pathway_num, disease_num = 894, 18877, 314, 454
    all_num = 20539

    A = np.zeros((12, all_num, all_num), dtype=np.int8)

    A[0, 0:drug_num, 0:drug_num] = drug_simi_drug
    A[1, drug_num:(drug_num + protein_num), drug_num:(drug_num + protein_num)] = protein_protein
    A[2, (drug_num + protein_num):(drug_num + protein_num + pathway_num),
    (drug_num + protein_num):(drug_num + protein_num + pathway_num)] = pathway_pathway
    A[3, (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num),
    (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num)] = disease_disease
    A[4, 0:drug_num, drug_num:(drug_num + protein_num)] = drug_protein
    A[5, drug_num:(drug_num + protein_num), 0:drug_num] = protein_drug
    A[6, 0:drug_num,
    (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num)] = drug_disease
    A[7, (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num),
    0:drug_num] = drug_disease.T
    A[8, drug_num:(drug_num + protein_num),
    (drug_num + protein_num):(drug_num + protein_num + pathway_num)] = protein_pathway
    A[9, (drug_num + protein_num):(drug_num + protein_num + pathway_num),
    drug_num:(drug_num + protein_num)] = pathway_protein
    A[10, (drug_num + protein_num):(drug_num + protein_num + pathway_num),
    (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num)] = pathway_disease
    A[11, (drug_num + protein_num + pathway_num):(drug_num + protein_num + pathway_num + disease_num),
    (drug_num + protein_num):(drug_num + protein_num + pathway_num)] = disease_pathway
    A = torch.from_numpy(A)

    return A


def load_data_graph():
    drug_protein = np.load('./base_meta_path/drug--protein.npy')  # (894,18877)
    protein_drug = drug_protein.transpose()  # (18877,984)

    disease_pathway = np.load('./base_meta_path/disease--pathway.npy')  # (453,314)
    pathway_disease = disease_pathway.transpose()  # (314,453)

    drug_drug = np.load('./base_meta_path/drug--drug.npy')  # (894,894)
    disease_disease = np.load('./base_meta_path/disease--disease.npy')  # (454,454)
    protein_protein = np.load('./base_meta_path/protein--protein.npy')  # (18877,18877)
    pathway_pathway = np.load('./base_meta_path/pathway--pathway.npy')  # (314,314)

    protein_pathway = np.load('./base_meta_path/protein--pathway.npy')  # (18877,314)
    pathway_protein = np.load('./base_meta_path/pathway--protein.npy')  # (314,18877)

    protein_gene = np.load('./base_meta_path/protein--gene.npy')
    # gene_protein = np.load('./base_meta_path/gene--protein.npy')

    # pathway_gene = np.load('./base_meta_path/pathway--gene.npy')
    # gene_pathway = np.load('./base_meta_path/gene--pathway.npy')

    # gene_gene = np.load('./base_meta_path/gene--gene.npy')

    drug_disease = np.load('./base_meta_path/drug--disease.npy')  # (894,454)

    drug_sim = np.load('./dataset/drug_drug_baseline.npy')
    disease_sim = np.load('./dataset/disease_disease_baseline.npy')
    # pathway_sim = np.load('./base_meta_path/pathway_pathway_baseline.npy')
    drug_num, protein_num, pathway_num, disease_num, gene_num = 894, 18877, 314, 454, 20561

    graph_data = {
        ('drug', 'drug_drug', 'drug'): drug_drug.nonzero(),
        ('drug', 'drug_protein', 'protein'): drug_protein.nonzero(),
        ('protein', 'protein_drug', 'drug'): protein_drug.nonzero(),
        ('protein', 'protein_protein', 'protein'): protein_protein.nonzero(),
        ('pathway', 'pathway_pathway', 'pathway'): pathway_pathway.nonzero(),
        ('pathway', 'pathway_disease', 'disease'): pathway_disease.nonzero(),
        ('disease', 'disease_pathway', 'pathway'): disease_pathway.nonzero(),
        ('disease', 'disease_disease', 'disease'): disease_disease.nonzero(),
        ('protein', 'protein_pathway', 'pathway'): protein_pathway.nonzero(),
        ('pathway', 'pathway_protein', 'protein'): pathway_protein.nonzero(),
        ('drug', 'drug_disease', 'disease'): drug_disease.nonzero(),
        ('disease', 'disease_drug', 'drug'): drug_disease.T.nonzero(),
        # ('protein', 'protein_gene', 'gene'): protein_gene.nonzero(),
        # ('gene', 'gene_protein', 'protein'): gene_protein.nonzero(),
        # ('pathway', 'pathway_gene', 'gene'): pathway_gene.nonzero(),
        # ('gene', 'gene_pathway', 'pathway'): gene_pathway.nonzero(),
        # ('gene', 'gene_gene', 'gene'): gene_gene.nonzero()
    }
    graph_node = {
        'drug': drug_num,
        'disease': disease_num,
        'protein': protein_num,
        'pathway': pathway_num,
        # 'gene': gene_num
    }
    g = dgl.heterograph(data_dict=graph_data, num_nodes_dict=graph_node)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    # g.nodes['pathway'].data['h'] = th.from_numpy(pathway_sim).to(th.float32)  # [314,314]
    return g


def load(topk):
    drug_drug = pd.read_csv('./dataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], topk)
        drug_drug[i, sorted_idx[-topk:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/interactions/protein_protein.csv')
    # gene_gene = pd.read_csv('./dataset/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], topk)
        disease_disease[i, sorted_idx[-topk:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/associations/drug_protein.csv')
    # protein_gene = pd.read_csv('./dataset/associations/protein_gene.csv')
    # gene_pathway = pd.read_csv('./dataset/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/associations/KFCdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        # ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
        #                                       th.tensor(protein_gene['Gene'].values)),
        # ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
        #                                       th.tensor(protein_gene['Protein'].values)),
        # ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
        #                                 th.tensor(gene_gene['Gene2'].values)),
        # ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
        #                                       th.tensor(gene_pathway['Pathway'].values)),
        # ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
        #                                       th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    return g


def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    # print('test_drug_id:', test_drug_id)

    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.from_numpy(np.array(test_drug_id)),
                          th.from_numpy(np.array(test_dis_id)),
                          etype=('drug', 'drug_disease', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id),
                          th.tensor(test_drug_id),
                          etype=('disease', 'disease_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'disease_drug', 'drug'))
    return g

# A = data_loader()
# print('A.shape:', A.shape)
# G = load_data_graph()
# print("G:", G)
# with open('./heterograph/no_gene_graph.dgl.pkl', 'wb') as fp:
#     pickle.dump(G, fp)
