import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gnn_w_change as gnn_w


class LongitudinalcCNN(nn.Module):
    def __init__(self, args):
        super(LongitudinalcCNN, self).__init__()
        self.kernel_num = 2
        self.conv1 = nn.Conv2d(1,self.kernel_num, [2,1],bias=False)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.conv2 = nn.Conv2d(self.kernel_num,1,[3,1],bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.clinical_f_n = 2
    def forward(self, inputs):
        inputs_mri = inputs[:,:,:,self.clinical_f_n:]
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs_mri)), 1)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        e2 = self.bn2(self.conv2(x))
        x = F.leaky_relu(e2, 0.1, inplace=True)
        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x

class LongitudinalcCNN_6(nn.Module):

    def __init__(self, args):
        super(LongitudinalcCNN_6, self).__init__()
        self.kernel_num = 2
        self.conv1 = nn.Conv2d(1,self.kernel_num, [4,1],bias=False)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.conv2 = nn.Conv2d(self.kernel_num,1,[3,1],bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.clinical_f_n = 2
    def forward(self, inputs):
        inputs_mri = inputs[:,:,:,self.clinical_f_n:]
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs_mri)), 1)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        e2 = self.bn2(self.conv2(x))
        x = F.leaky_relu(e2, 0.1, inplace=True)
        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x

class LongitudinalcCNN_3(nn.Module):

    def __init__(self, args):
        super(LongitudinalcCNN_3, self).__init__()
        self.kernel_num = 2
        self.conv1 = nn.Conv2d(1,self.kernel_num, [2,1],bias=False)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.conv2 = nn.Conv2d(self.kernel_num,1,[2,1],bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.clinical_f_n = 2
    def forward(self, inputs):
        inputs_mri = inputs[:,:,:,self.clinical_f_n:]
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs_mri)), 1)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        e2 = self.bn2(self.conv2(x))
        x = F.leaky_relu(e2, 0.1, inplace=True)
        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x


class LongitudinalcCNN_2(nn.Module):

    def __init__(self, args):
        super(LongitudinalcCNN_2, self).__init__()
        self.kernel_num = 1
        self.conv1 = nn.Conv2d(1,self.kernel_num, [2,1],bias=False)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)

        self.clinical_f_n = 2
    def forward(self, inputs):
        inputs_mri = inputs[:,:,:,self.clinical_f_n:]
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs_mri)), 1)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x

class LongitudinalcCNN_2_2s(nn.Module):

    def __init__(self, args):
        super(LongitudinalcCNN_2_2s, self).__init__()
        self.kernel_num = 1
        self.conv1 = nn.Conv2d(1,self.kernel_num, [2,1],bias=False)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.clinical_f_n = 2
    def forward(self, inputs):
        inputs_mri = inputs[:,:,:,self.clinical_f_n:]
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs_mri)), 1)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x



class LongitudinalcCNN_1(nn.Module):

    def __init__(self, args):
        super(LongitudinalcCNN_1, self).__init__()
        self.kernel_num = 1
        self.clinical_f_n = 2
    def forward(self, inputs):
        x = inputs[:,:,:,self.clinical_f_n:]

        clinical = inputs[:,0,0,0:self.clinical_f_n]
        return clinical, x

class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs, dim=1)
        else:
            raise (NotImplementedError)


'''
Attention-MLP
'''
def AttentionClassify():
    '''
    60 is for testing
    :return:
    '''
    model = nn.Sequential(
        nn.Linear(60, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )
    return model
def CS_Attention():
    '''
    13 is for testing
    :return:
    '''
    model = nn.Sequential(
        nn.Linear(13, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )
    return model
def CM_Attention():
    '''
    21 is for testing
    :return:
    '''
    model = nn.Sequential(
        nn.Linear(21, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )
    return model
def MS_Attention():
    '''
    26 is for testing
    :return:
    '''
    model = nn.Sequential(
        nn.Linear(26, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )
    return model

class Enhanced_framework(nn.Module):
    def __init__(self, args):
        super(Enhanced_framework, self).__init__()
        self.metric_network = args.metric_network
        self.emb_size = 30
        self.args = args
        self.attention_classify = AttentionClassify()
        self.attention_C_S = CS_Attention()
        self.attention_C_M = CM_Attention()
        self.attention_M_S = MS_Attention()
        # 2571 is only  for testing
        self.p_gen_linear = nn.Linear(2571, 1)
        self.p_gen_linear_sc = nn.Linear(2524, 1)
        self.p_gen_linear_cm = nn.Linear(2532, 1)
        self.p_gen_linear_sm = nn.Linear(2537, 1)
        if self.metric_network == 'gnn':
            assert (self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            self.gnn_obj = gnn_w.GNN_nl(args, num_inputs, nf=96, J=1)
        else:
            raise NotImplementedError

    def cin2mri(self,mri,cli):
        scores = torch.matmul(mri.transpose(0, 1), cli)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, cli.transpose(0, 1))
        output = output.transpose(0, 1)
        return output
    def mri2cli(self,cli,mri):
        scores = torch.matmul(cli.transpose(0, 1), mri)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, mri.transpose(0, 1))
        output = output.transpose(0, 1)
        return output

    def cin2score(self,ns,cli):
        scores = torch.matmul(ns.transpose(0, 1), cli)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, cli.transpose(0, 1))
        output = output.transpose(0, 1)
        return output
    def score2cli(self,cli,ns):
        scores = torch.matmul(cli.transpose(0, 1), ns)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, ns.transpose(0, 1))
        output = output.transpose(0, 1)
        return output

    def score2mri(self,mri,ns):
        scores = torch.matmul(mri.transpose(0, 1), ns)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, ns.transpose(0, 1))
        output = output.transpose(0, 1)
        return output
    def mri2score(self,ns,mri):
        scores = torch.matmul(ns.transpose(0, 1), mri)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, mri.transpose(0, 1))
        output = output.transpose(0, 1)
        return output

    def mri2mri(self,mri):
        scores = torch.matmul(mri.transpose(0, 1), mri)
        attention = torch.softmax(scores, dim=1)
        output = torch.matmul(attention, mri.transpose(0, 1))
        output = output.transpose(0, 1)
        return output

    def gnn_iclr_forward_pgen(self, z_c, z_score, z_mri, zi_c, zi_score, zi_mri, labels_yi, adj,args):
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()
        labels_yi = [zero_pad] + labels_yi
        # calculate attention
        mri_f = z_mri.squeeze()
        score_f = z_score.squeeze()
        att_mri_clin_f = self.cin2mri(mri_f, z_c)
        att_cli_mri_f = self.mri2cli(z_c, mri_f)
        clin_mri_input_attention = torch.cat((att_mri_clin_f, att_cli_mri_f), dim=1)

        att_score_clin_f = self.cin2score(score_f, z_c)
        att_cli_score_f = self.score2cli(z_c, score_f)
        clin_score_input_attention = torch.cat((att_score_clin_f, att_cli_score_f), dim=1)

        att_score_mri_f = self.score2mri(mri_f, score_f)
        att_mri_score_f = self.mri2score(score_f, mri_f)
        mri_score_input_attention = torch.cat((att_score_mri_f, att_mri_score_f), dim=1)

        if args.c_s_m ==0:
            input_attention = torch.cat((clin_mri_input_attention, clin_score_input_attention, mri_score_input_attention),
                                        dim=1)
            output_attention = self.attention_classify(input_attention)
            # generate node features
            z = torch.cat((z_mri, z_score), dim=3)
            zi_s = [torch.cat((x, y), dim=3) for x, y in zip(zi_mri, zi_score)]
            zi_s = [z] + zi_s
            zi_c = [z_c] + zi_c
            zi_s = [torch.squeeze(zi_un) for zi_un in zi_s]
            zi_s_ = [torch.cat([zic, zi], 1) for zi, zic in zip(zi_s, zi_c)]

            nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s_, labels_yi)]
            nodes = [a.unsqueeze(1) for a in nodes]
            nodes = torch.cat(nodes, 1)
            logits, gnn_x = self.gnn_obj(nodes, adj)
            logits = logits.squeeze(-1)
            outputs = torch.sigmoid(logits)
            # weight parameter calculation
            gnn_x_update = gnn_x.view(gnn_x.shape[0], gnn_x.shape[1] * gnn_x.shape[2])
            pgen_input = torch.cat((gnn_x_update, input_attention,z_c, z_score, z_mri), dim=1)
            pgen = self.p_gen_linear(pgen_input)
            p_gen = F.sigmoid(pgen)
            gnn_output = p_gen * logits
            output_attention = (1 - p_gen) * output_attention
            logits = torch.cat([gnn_output, output_attention], 1)
            return outputs, logits
        elif args.c_s_m ==1:
            output_attention = self.attention_C_S(clin_score_input_attention)
            # generate node features
            z = torch.cat((z_mri, z_score), dim=3)
            zi_s = [torch.cat((x, y), dim=3) for x, y in zip(zi_mri, zi_score)]
            zi_s = [z] + zi_s
            zi_c = [z_c] + zi_c
            zi_s = [torch.squeeze(zi_un) for zi_un in zi_s]
            zi_s_ = [torch.cat([zic, zi], 1) for zi, zic in zip(zi_s, zi_c)]

            nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s_, labels_yi)]
            nodes = [a.unsqueeze(1) for a in nodes]
            nodes = torch.cat(nodes, 1)
            logits, gnn_x = self.gnn_obj(nodes, adj)
            logits = logits.squeeze(-1)
            outputs = torch.sigmoid(logits)

            # weight parameter calculation
            gnn_x_update = gnn_x.view(gnn_x.shape[0], gnn_x.shape[1] * gnn_x.shape[2])
            pgen_input = torch.cat((gnn_x_update, clin_score_input_attention,z_c, z_score, z_mri), dim=1)
            pgen = self.p_gen_linear_sc(pgen_input)
            p_gen = F.sigmoid(pgen)

            gnn_output = p_gen * logits
            output_attention = (1 - p_gen) * output_attention
            logits = torch.cat([gnn_output, output_attention], 1)
            return outputs, logits
        elif args.c_s_m ==2:
            output_attention = self.attention_C_M(clin_mri_input_attention)
            # generate node features
            z = torch.cat((z_mri, z_score), dim=3)
            zi_s = [torch.cat((x, y), dim=3) for x, y in zip(zi_mri, zi_score)]
            zi_s = [z] + zi_s
            zi_c = [z_c] + zi_c
            zi_s = [torch.squeeze(zi_un) for zi_un in zi_s]
            zi_s_ = [torch.cat([zic, zi], 1) for zi, zic in zip(zi_s, zi_c)]

            nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s_, labels_yi)]
            nodes = [a.unsqueeze(1) for a in nodes]
            nodes = torch.cat(nodes, 1)
            logits, gnn_x = self.gnn_obj(nodes, adj)
            logits = logits.squeeze(-1)
            outputs = torch.sigmoid(logits)

            # weight parameter calculation
            gnn_x_update = gnn_x.view(gnn_x.shape[0], gnn_x.shape[1] * gnn_x.shape[2])
            pgen_input = torch.cat((gnn_x_update, clin_mri_input_attention,z_c, z_score, z_mri), dim=1)
            pgen = self.p_gen_linear_cm(pgen_input)
            p_gen = F.sigmoid(pgen)

            gnn_output = p_gen * logits
            output_attention = (1 - p_gen) * output_attention
            logits = torch.cat([gnn_output, output_attention], 1)
            return outputs, logits
        elif args.c_s_m ==3:
            output_attention = self.attention_M_S(mri_score_input_attention)
            # generate node features
            z = torch.cat((z_mri, z_score), dim=3)
            zi_s = [torch.cat((x, y), dim=3) for x, y in zip(zi_mri, zi_score)]
            zi_s = [z] + zi_s
            zi_c = [z_c] + zi_c
            zi_s = [torch.squeeze(zi_un) for zi_un in zi_s]
            zi_s_ = [torch.cat([zic, zi], 1) for zi, zic in zip(zi_s, zi_c)]

            nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s_, labels_yi)]
            nodes = [a.unsqueeze(1) for a in nodes]
            nodes = torch.cat(nodes, 1)
            logits, gnn_x = self.gnn_obj(nodes, adj)
            logits = logits.squeeze(-1)
            outputs = torch.sigmoid(logits)

            # weight parameter calculation
            gnn_x_update = gnn_x.view(gnn_x.shape[0], gnn_x.shape[1] * gnn_x.shape[2])
            pgen_input = torch.cat((gnn_x_update, mri_score_input_attention,z_c, z_score, z_mri), dim=1)
            pgen = self.p_gen_linear_sm(pgen_input)
            p_gen = F.sigmoid(pgen)

            gnn_output = p_gen * logits
            output_attention = (1 - p_gen) * output_attention
            logits = torch.cat([gnn_output, output_attention], 1)
            return outputs, logits

    def forward(self, inputs,args):
        [z_c, z_score, z_mri, zi_c, zi_score, zi_s, labels_yi, _, adj] = inputs
        return self.gnn_iclr_forward_pgen(z_c, z_score, z_mri, zi_c, zi_score, zi_s, labels_yi, adj,args)


def create_models(args,cnn_dim1 = 4):
    return Enhanced_framework(args)

