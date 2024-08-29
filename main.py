# -*- coding: utf-8 -*-
# @Time : 2022/11/20 | 13:20
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : main.py
# Software: PyCharm

from imports import *
from SeHG import SeHG_bio
from warnings import simplefilter
from sklearn.model_selection import KFold
from load_data import load, remove_graph,load_D3
from utils import get_metrics_auc, set_seed, plot_result_auc, \
    plot_result_aupr, EarlyStopping, get_metrics
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General Arguments
parser.add_argument('-id', '--device_id', default='0', type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-da', '--dataset', default='mat_Drug_Disease', type=str,
                    help='Set the data set for training.')
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
parser.add_argument('-se', '--seed', default=5800, type=int,
                    help='Global random seed')

# Training Arguments
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=800, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=100, type=int,
                    help='Early Stopping argument')
parser.add_argument('-tk', '--topk', default=15, type=int,
                    help='topk_numbers')
# Model Arguments
parser.add_argument('-hf', '--hidden_feats', default=64, type=int,
                    help='The dimension of hidden tensor in the model')

# meta_paths
DRUG_METAPATH_LIST = [
    ['drug_drug'],
    ['drug_protein', 'protein_drug'],
    ['drug_disease', 'disease_drug'],

    ['drug_drug', 'drug_protein', 'protein_drug'],
    ['drug_drug', 'drug_disease', 'disease_drug'],

    ['drug_protein', 'protein_drug', 'drug_drug'],
    ['drug_disease', 'disease_drug', 'drug_drug'],

    ['drug_disease', 'disease_pathway', 'pathway_disease', 'disease_drug'],
    # ['drug_drug', 'drug_protein', 'protein_gene', 'gene_protein', 'protein_drug'],
]
DISEASE_METAPATH_LIST = [
    ['disease_disease'],
    ['disease_pathway', 'pathway_disease'],
    ['disease_drug', 'drug_disease'],

    ['disease_disease', 'disease_pathway', 'pathway_disease'],
    ['disease_disease', 'disease_drug', 'drug_disease'],

    ['disease_pathway', 'pathway_disease', 'disease_disease'],
    ['disease_drug', 'drug_disease', 'disease_disease'],

    ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease'],
    # ['disease_disease', 'disease_pathway', 'pathway_gene', 'gene_pathway', 'pathway_disease']
]

assert len(DRUG_METAPATH_LIST) == len(DISEASE_METAPATH_LIST)

# model core params
HYPER_PARAM = dict(
    drug_metapath_list=DRUG_METAPATH_LIST,
    disease_metapath_list=DISEASE_METAPATH_LIST,
)
metapath_list = [HYPER_PARAM['drug_metapath_list'], HYPER_PARAM['disease_metapath_list']]


def train():
    args = parser.parse_args()
    args.saved_path = args.saved_path + '_' + str(args.seed)
    set_seed(args.seed)
    print(args)
    simplefilter(action='ignore', category=FutureWarning)
    if args.device_id:
        print('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = th.device('cpu')
    try:
        os.mkdir(args.saved_path)
    except:
        pass

    # data loading,  numpy
    df = pd.read_csv('./dataset/Dataset3/{}.csv'.format(args.dataset), header=None).values
    # print('df:', df)
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    # print('data:', data)
    data = data.astype('int64')

    # pos:[[row_index,col_index,label],.....]
    # 所有正样本
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    print('all positive sample number:', len(data_pos))
    # 所有负样本
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    print('all negative sample number:', len(data_neg))
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold = 1
    pred_result = np.zeros(df.shape)

    # finished 5-CV, four for training, one for testing
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))
        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        # train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        # test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        # train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        # test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]
        train_pos_idx = [train_pos_id[:, 0], train_pos_id[:, 1]]
        test_pos_idx = [test_pos_id[:, 0], test_pos_id[:, 1]]
        train_neg_idx = [train_neg_id[:, 0], train_neg_id[:, 1]]
        test_neg_idx = [test_neg_id[:, 0], test_neg_id[:, 1]]

        # save train and test sets
        np.save('{}_fold_train_pos_idx_data.npy'.format(fold), np.array(train_pos_idx))
        np.save('{}_fold_train_neg_idx_data.npy'.format(fold), np.array(train_neg_idx))
        np.save('{}_fold_test_pos_idx_data.npy'.format(fold), np.array(test_pos_idx))
        np.save('{}_fold_test_neg_idx_data.npy'.format(fold), np.array(test_neg_idx))

        # load hetero_graph
        g = load_D3(args.topk)
        # remove test from test_set
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)
        # extracted drug and disease features
        feature = {'drug': g.nodes['drug'].data['h'], 'disease': g.nodes['disease'].data['h']}

        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
        mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
        mask_train = np.where(mask_label == 1)
        mask_train = [mask_train[0], mask_train[1]]
        mask_test = np.where(mask_label == 0)
        mask_test = [mask_test[0], mask_test[1]]

        # Number of total training samples: 324702, pos samples: 2164, neg samples: 322538
        print('Number of total training samples: {}, train pos samples: {}, train neg samples: {}'.format(
            len(mask_train[0]),
            len(train_pos_idx[0]),
            len(train_neg_idx[0])))
        # Number of total testing samples: 81174, pos samples: 540, neg samples: 80634
        print(
            'Number of total testing samples: {}, test pos samples: {}, test neg samples: {}'.format(len(mask_test[0]),
                                                                                                     len(test_pos_idx[
                                                                                                             0]),
                                                                                                     len(test_neg_idx[
                                                                                                             0])))
        assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])

        # true label
        label = th.tensor(df).float().to(device)

        # init model
        model = SeHG_bio(in_dim=feature['drug'].shape[1], hidden_dim=args.hidden_feats,
                         num_metapaths=len(metapath_list[0]))
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)
        criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        print('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)

        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model(g, feature, metapath_list,epoch,fold)
            pred = th.sigmoid(score)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC_, AUPR_ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                          pred[mask_train].cpu().detach().numpy())
            early_stop = stopper.step(loss.item(), AUC_, model)

            if epoch % 50 == 0:
                AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
                                            pred[mask_test].cpu().detach().numpy())
                print(
                    'Epoch {} Loss: {:.3f}; Train AUC: {:.3f}; Train AUPR: {:.3f}; Test AUC: {:.3f}; Test AUPR: {:.3f}'.format(
                        epoch,
                        loss.item(),
                        AUC_,
                        AUPR_,
                        AUC,
                        AUPR))
                print('-' * 50)

                if early_stop:
                    break

        stopper.load_checkpoint(model)
        model.eval()
        pred = model(g, feature, metapath_list,epoch,fold)
        pred = th.sigmoid(pred).cpu().detach().numpy()
        pred_result[test_pos_idx] = pred[test_pos_idx]
        pred_result[test_neg_idx] = pred[test_neg_idx]
        # print("fold{},weight_attn:{}".format(fold,weight_attn))
        fold += 1

    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC: {:.4f}; AUPR: {:.4f}; F1: {:.4f}; Acc: {:.4f}; Recall {:.4f}; Specificity {:.4f}; Precision {:.4f}'.
        format(AUC, aupr, f1, acc, rec, spe, pre))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path,
                                                  'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)
    return AUC, aupr, f1, acc, rec, spe, pre


if __name__ == '__main__':
    # all_reuslt = []
    # for i in range(5):
    #     AUC, AUPR, f1, acc, rec, spe, pre = train()
    #     all_reuslt.append([AUC, AUPR, f1, acc, rec, spe, pre])
    # all_reuslt = np.array(all_reuslt)
    # np.savetxt('all_result.csv', all_reuslt, delimiter=',')

    AUC, AUPR, f1, acc, rec, spe, pre = train()
