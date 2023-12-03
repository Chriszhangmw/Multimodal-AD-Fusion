
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.utils.data as data
import random
import torch
import gnn_model_w_change as models
import argparse
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import io_utils
import pickle
import os
import time

parser = argparse.ArgumentParser(description='enhanced_framework')
parser.add_argument('--metric_network', type=str, default='gnn', metavar='N',
                    help='gnn')
parser.add_argument('--dataset', type=str, default='AD', metavar='N',
                    help='AD')
parser.add_argument('--test_N_way', type=int, default=3, metavar='N')
parser.add_argument('--train_N_way', type=int, default=3, metavar='N')
parser.add_argument('--test_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--train_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--feature_num', type=int, default=31, metavar='N',
                    help='feature number of one sample')
parser.add_argument('--clinical_feature_num', type=int, default=4, metavar='N',
                    help='clinical feature number of one sample')
parser.add_argument('--w_feature_num', type=int, default=27, metavar='N',
                    help='feature number for w computation')
parser.add_argument('--w_feature_list', type=int, default=5, metavar='N',
                    help='feature list for w computation')
parser.add_argument('--alph', type=float, default=0.7,
                    help='the weight of GNN loss in the overall loss')
parser.add_argument('--type', type=int, default=3, metavar='N')
parser.add_argument('--c_s_m', type=int, default=0, metavar='N')
parser.add_argument('--feature_selection_type', type=int, default=0, metavar='N')
parser.add_argument('--iterations', type=int, default=400, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_train', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--test_interval', type=int, default=200, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--random_seed', type=int, default=2023, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU:', args.cuda)
random_seed = args.random_seed
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
all_index_bad1 = [26] # only for ADNI1 datasets
all_index_bad2 =['feature_169',
 'feature_66',
 'feature_178',
 'feature_192',
 'feature_139',
 'feature_165',
 'feature_52',
 'feature_182',
 'feature_161',
 'feature_98',
 'feature_67',
 'feature_175',
 'feature_189',
 'feature_123',
 'feature_171',
 'feature_147',
 'feature_158',
 'feature_174',
 'feature_20',
 'feature_155',
 'feature_90',
 'feature_71',
 'feature_143',
 'feature_151',
 'feature_122',
 'feature_84',
 'feature_156',
 'feature_145',
 'feature_22',
 'feature_116',
 'feature_183',
 'feature_86',
 'feature_124',
 'feature_77',
 'feature_120',
 'feature_109',
 'feature_115',
 'feature_80',
 'feature_188',
 'feature_172',
 'feature_162',
 'feature_61',
 'feature_110',
 'feature_107',
 'feature_117',
 'feature_159',
 'feature_176',
 'feature_179',
 'feature_50',
 'feature_68',
 'feature_99',
 'feature_73',
 'feature_132',
 'feature_51',
 'feature_79',
 'feature_149',
 'feature_103',
 'feature_114',
 'feature_75',
 'feature_136',
 'feature_64',
 'feature_113',
 'feature_57',
 'feature_194',
 'feature_83',
 'feature_63',
 'feature_112',
 'feature_100',
 'feature_111',
 'feature_187',
 'feature_185',
 'feature_76',
 'feature_148',
 'feature_184',
 'feature_3',
 'feature_12',
 'feature_157',
 'feature_5',
 'feature_121']

def setup_seed(seed=random_seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class Generator(data.DataLoader):
    def __init__(self, data):
        self.data = data
        self.channal = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(self, batch_size=5, n_way=4, num_shots=10, unlabeled_extra=0, cuda=False, variable=False):
        # init
        batch_x = np.zeros((batch_size,self.channal,self.feature_shape[0],self.feature_shape[1]), dtype='float32')  # features
        labels_x = np.zeros((batch_size, n_way), dtype='float32')  # labels
        labels_x_global = np.zeros(batch_size, dtype='int64')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(np.zeros((batch_size,self.channal,self.feature_shape[0],self.feature_shape[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype='float32')))

        # feed data
        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if class_num == pre_class:
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    # Test sample
                    batch_x[batch_counter,0, :,:] = samples[0]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                for samples_num in range(len(samples)):
                    try:
                        batches_xi[indexes_perm[counter]][batch_counter, :] = samples[samples_num]
                    except:
                        print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

def compute_adj(batch_x, batches_xi):
    x = torch.squeeze(batch_x)
    xi_s = [torch.squeeze(batch_xi) for batch_xi in batches_xi]

    nodes = [x] + xi_s
    nodes = [node.unsqueeze(1) for node in nodes]
    nodes = torch.cat(nodes, 1)
    age = nodes.narrow(2, 0, 1)
    age = age.cpu().numpy()
    gender = nodes.narrow(2, 1, 1)
    gendre = gender.cpu().numpy()
    apoe = nodes.narrow(2, 2, 1)
    apoe = apoe.cpu().numpy()
    edu = nodes.narrow(2, 3, 1)
    edu = edu.cpu().numpy()
    adj = np.ones(
        (args.batch_size, args.train_N_way * args.train_N_shots + 1, args.train_N_way * args.train_N_shots + 1, 1),
        dtype='float32') + 4

    for batch_num in range(args.batch_size):
        for i in range(args.train_N_way * args.train_N_shots + 1):
            for j in range(i + 1, args.train_N_way * args.train_N_shots + 1):
                if np.abs(age[batch_num, i, 0] - age[batch_num, j, 0]) <= 0.06:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if np.abs(edu[batch_num, i, 0] - edu[batch_num, j, 0]) <= 0.14:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if gendre[batch_num, i, 0] == gendre[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if apoe[batch_num, i, 0] == apoe[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
    adj = 1 / adj
    adj = torch.from_numpy(adj)
    return adj

def train_batch(model, data,args):
    [enhanced_framework, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi] = data
    all_index_bad1 = [26]
    batch_x = np.delete(batch_x, all_index_bad1, axis=3)
    batches_xi = [np.delete(batch_xi, all_index_bad1, axis=3) for batch_xi in batches_xi]
    z_clinical = batch_x[:, 0, 0, 0:4]
    zi_s_clinical = [batch_xi[:, 0, 0, 0:4] for batch_xi in batches_xi]
    # scores features
    z_scores = batch_x[:, :, :, 4:13]
    zi_s_scores = [batch_xi[:, :, :, 4:13] for batch_xi in batches_xi]
    z_mri_feature = batch_x[:, :, :,13:]
    zi_s_mri_feature = [batch_xi[:, :, :, 13:] for batch_xi in batches_xi]
    adj = compute_adj(z_clinical, zi_s_clinical)
    out_metric, out_logits = enhanced_framework(
        inputs=[z_clinical, z_scores, z_mri_feature, zi_s_clinical, zi_s_scores, zi_s_mri_feature, labels_yi,
              oracles_yi, adj],args=args)
    logsoft_prob = softmax_module.forward(out_logits)
    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()
    return loss

def test_one_shot(args, fold,testdata, model, test_samples=50, partition='test',io_path= 'run.log'):
    io = io_utils.IOStream(io_path)

    io.cprint('\n**** TESTING BEGIN ***' )

    loader = Generator(testdata)
    [enhanced_framework, softmax_module] = model
    enhanced_framework.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []
    iterations = int(test_samples / args.batch_size_test)
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra,cuda = args.cuda)
        [x_t, labels_x_cpu_t, _, _, xi_s, labels_yi_cpu, oracles_yi] = data



        batch_x = np.delete(x_t, all_index_bad1, axis=3)
        batches_xi = [np.delete(batch_xi, all_index_bad1, axis=3) for batch_xi in xi_s]
        z_clinical = batch_x[:, 0, 0, 0:4]
        zi_s_clinical = [batch_xi[:, 0, 0, 0:4] for batch_xi in batches_xi]
        # scores features
        z_scores = batch_x[:, :, :, 4:13]
        zi_s_scores = [batch_xi[:, :, :, 4:13] for batch_xi in batches_xi]
        z_mri_feature = batch_x[:, :, :, 13:]
        zi_s_mri_feature = [batch_xi[:, :, :, 13:] for batch_xi in batches_xi]

        '''
        For ADNI2
        batch_x = np.delete(batch_x, all_index_bad2, axis=3)
        batches_xi = [np.delete(batch_xi, all_index_bad2, axis=3) for batch_xi in batches_xi]
    
        # slice the first five features which are our risk factors
        z_clinical = batch_x[:, 0, 0, 0:args.clinical_feature_num-1]
        zi_s_clinical = [batch_xi[:,0,0,0:args.clinical_feature_num-1] for batch_xi in batches_xi]
    
        # scores features
        z_scores = batch_x[:, :, :, args.clinical_feature_num-1:args.clinical_feature_num + args.scores-2]
        zi_s_scores = [batch_xi[:, :, :, args.clinical_feature_num-1:args.clinical_feature_num + args.scores-2] for batch_xi
                       in batches_xi]
    
        # slice the remaining features after our clinical / risk factors
        z_mri_feature = batch_x[:, :, :, args.clinical_feature_num + args.scores-3:]
        zi_s_mri_feature = [batch_xi[:, :, :, args.clinical_feature_num + args.scores-3:] for batch_xi in batches_xi]
        '''

        adj = compute_adj(z_clinical, zi_s_clinical)
        x = x_t
        labels_x_cpu = labels_x_cpu_t
        labels_yi = labels_yi_cpu
        xi_s = [Variable(batch_xi) for batch_xi in zi_s_mri_feature]
        zi_s_scores = [Variable(batch_xi) for batch_xi in zi_s_scores]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        z_mri_feature = Variable(z_mri_feature)
        z_scores = Variable(z_scores)
        # Compute metric from embeddings
        output, out_logits = enhanced_framework(
            inputs=[z_clinical, z_scores,z_mri_feature, zi_s_clinical, zi_s_scores,xi_s, labels_yi, oracles_yi, adj],args=args)
        output = out_logits
        Y = softmax_module.forward(output)
        y_pred = softmax_module.forward(output)
        y_pred = y_pred.data.cpu().numpy()
        y_inter = [list(y_i) for y_i in y_pred]
        pre_all_num = pre_all_num + list(y_inter)
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.cpu().numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
        pre_all = pre_all+list(y_pred)
        real_all = real_all + list(labels_x_cpu)
        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1
    # labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    labels_x_cpu = Variable(torch.LongTensor(labels_x_cpu))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    loss_test_f = float(loss_test)
    del loss_test
    io.cprint('real_label:  '+str(real_all))
    io.cprint('pre_all:  '+str(pre_all))
    io.cprint('pre_all_num:  '+str(pre_all_num))
    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))

    enhanced_framework.train()

    return 100.0 * correct / total, loss_test_f

def generate_dic_data(features,labels,index_list):
    x = [features[id1] for id1 in index_list]
    y = [labels[id1] for id1 in index_list]
    data = {}
    list_0 = []
    list_1 = []
    list_2 = []
    for feature,label in zip(x,y):
        if label == 0:
            list_0.append(feature)
        elif label == 1:
            list_1.append(feature)
        elif label == 2:
            list_2.append(feature)
        else:
            print('error label')
    data[0] = list_0
    data[1] = list_1
    data[2] = list_2
    return data

if __name__ =='__main__':
    ## Kfold implementation
    features = []
    labels = []
    root_train = './dataset/adni1_train.pkl'
    root_test = './dataset/adni1_test.pkl'
    with open(root_train, 'rb') as load_data1:
        data_dict_train = pickle.load(load_data1)
    with open(root_test, 'rb') as load_data2:
        data_dict_test = pickle.load(load_data2)
    keys = ['CN', 'MCI', 'AD']
    for i in range(len(keys)):
        list1 = data_dict_train[keys[i]]
        features.extend(list1)
        labels.extend([i] * len(list1))
        list2 = data_dict_test[keys[i]]
        features.extend(list2)
        labels.extend([i] * len(list2))
    stratifiedKFolds = StratifiedKFold(n_splits=10, shuffle=False)
    test_performance = []
    test_best_acc = 0.0
    for (trn_idx, val_idx) in stratifiedKFolds.split(features, labels):
        traindata = generate_dic_data(features,labels,trn_idx)
        testdata = generate_dic_data(features,labels,val_idx)
        timedata = time.strftime("%F")
        name = timedata + '-3-classe'
        save_path = 'result/{}'.format(name)

        if name not in os.listdir('result/'):
            os.makedirs(save_path)
        io = io_utils.IOStream(save_path + '/run.log')
        print('the result will be saved in :', save_path)
        setup_seed(args.random_seed)
        counter = 0
        total_loss = 0
        val_acc, val_acc_aux = 0, 0
        test_acc = 0
        enhanced_framework = models.create_models(args, cnn_dim1=2)
        io.cprint(str(enhanced_framework))
        softmax_module = models.SoftmaxModule()
        weight_decay = 0
        opt_enhanced_framework = optim.Adam(enhanced_framework.parameters(), lr=args.lr, weight_decay=weight_decay)
        enhanced_framework.train()
        for batch_idx in range(args.iterations):
            da = Generator(traindata)
            data = da.get_task_batch(batch_size=args.batch_size_train, n_way=args.train_N_way,
                                     num_shots=args.train_N_shots, unlabeled_extra=args.unlabeled_extra, cuda=args.cuda)
            [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi] = data

            opt_enhanced_framework.zero_grad()

            loss_d_metric = train_batch(model=[enhanced_framework, softmax_module],
                                        data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi],args=args)
            opt_enhanced_framework.step()

            adjust_learning_rate(optimizers=[opt_enhanced_framework], lr=args.lr, iter=batch_idx)

            ####################
            # Display
            ####################
            counter += 1
            total_loss += loss_d_metric.item()
            if batch_idx % args.log_interval == 0:
                display_str = 'Train Iter: {}'.format(batch_idx)
                display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)
                io.cprint(display_str)
                counter = 0
                total_loss = 0
            ####################
            # Test
            ####################
            if (batch_idx + 1) % args.log_interval == 0:
                test_samples = 320
                test_acc_aux, test_loss_ = test_one_shot(args, 0, testdata, model=[enhanced_framework, softmax_module],
                                                         test_samples=test_samples, partition='test',
                                                         io_path=save_path + '/run.log')
                enhanced_framework.train()
                if test_acc_aux is not None and test_acc_aux >= test_acc:
                    test_acc = test_acc_aux
                    modelpath = 'result/{}/'.format(name)
                    torch.save(enhanced_framework, modelpath + 'enhanced_framework' + '_best_model.pkl')
                if args.dataset == 'AD':
                    io.cprint("Best test accuracy {:.4f} \n".format(test_acc))
        test_best_acc = test_acc
        test_performance.append(test_acc)
    print("Best test accuracy: ",test_best_acc)
    print("feature selection type: ", args.feature_selection_type)
    print("c_s_m tyoe : ", args.c_s_m)
    print("kfold test performance: ",test_performance)


