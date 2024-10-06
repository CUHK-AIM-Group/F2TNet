import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import HCPfmrit1wScoreDataset
from torch_geometric.data import DataLoader
from net.braingnn import Network_regress_score
from imports.utils import train_val_test_split_hcp
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imports.vit import ViT
from net.Network_Dual_ViT import *
from loss_function.loss_cons  import *
from torchsummary import summary
torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/data/hzb/project/Brain_Predict_Score/ViTPre0219/F2TNet_Pytorch-main/data_hcp/HCP_pcp/Gordon/filt_noglobal', help='root directory of the dataset')

parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0005, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=10, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=10, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=333, help='feature dim')
parser.add_argument('--nroi', type=int, default=333, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
# path2 = opt.dataroot2
name = 'HCP'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))



################## Define Dataloader ##################################
#t1w
dataset = HCPfmrit1wScoreDataset(path,name)
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0
#fmri



tr_index,val_index,te_index = train_val_test_split_hcp(n_sub=850, fold=fold)
train_dataset = dataset[np.concatenate((tr_index,te_index))]
val_dataset = dataset[val_index]
test_dataset = dataset[te_index]



train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)



############### Define Graph Deep Learning Network ##########################

model = Network_regress_score_dual_multi(fmri_indim = 333,t1w_indim = 9, fmri_outdim=1024, image_size=333, patch_size=333, num_classes=1024, dim=1024, depth=1, heads=16,mlp_dim=2048).to(device)





print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################


###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()

    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #plt.imshow(data.x[0:333,:].detach().cpu().numpy())
        #plt.imshow(data.x.view(opt.batchSize,1,opt.nroi, opt.nroi)[0,0,:,:].detach().cpu().numpy())

        # t1 = model_v(data.x[0:333,:].unsqueeze(0).unsqueeze(0))
        # t1 = model(data.x.view(int(data.x.shape[0]/opt.nroi),1,opt.nroi, opt.nroi))
        # data_fmri = data.x[:,0:opt.nroi]
        # data_t1w = data.x[:,opt.nroi:]

        data_t1w = data.x[:,0:9]
        data_fmri = data.x[:,9:]
        # t11, t21, t1_node, t2_node = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9), data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))


        t11, t21, t12, t22,t13, t23,t14, t24,t15, t25,t16, t26, t1_node, t2_node, x1_class,x2_class  = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9), data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))

        # t2 = model(data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))
        # t2 = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9))



        # img = torch.randn(1,3,333,333).to(device)
        # model_v(img)

        # output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        # s1_list.append(s1.view(-1).detach().cpu().numpy())
        # s2_list.append(s2.view(-1).detach().cpu().numpy())
        data.y1 = torch.tensor(data.y[:, 0], dtype=torch.float)
        data.y2 = torch.tensor(data.y[:, 1], dtype=torch.float)
        data.y3 = torch.tensor(data.y[:, 2], dtype=torch.float)

        data.y4 = torch.tensor(data.y[:, 3], dtype=torch.float)
        data.y5 = torch.tensor(data.y[:, 4], dtype=torch.float)
        data.y6 = torch.tensor(data.y[:, 5], dtype=torch.float)

        # loss_c = F.nll_loss(output, data.y)
        # loss_c = F.mse_loss(t1[:,0,0], data.y)     #right

        # loss_c =  (F.mse_loss(t11, data.y1) +F.mse_loss(t21, data.y1) )\
        #           + 2* (F.mse_loss(t12, data.y2)  + F.mse_loss(t22, data.y2))\
        #           + (F.mse_loss(t13, data.y3)  + F.mse_loss(t23, data.y3))

        # loss_c =  (F.mse_loss(t11, data.y1)  )\
        #           + (F.mse_loss(t12, data.y2) )\
        #           + (F.mse_loss(t13, data.y3)  )\
        #           + (F.mse_loss(t14, data.y4) )\
        #           + (F.mse_loss(t15, data.y5) )\
        #           + (F.mse_loss(t16, data.y6) )

        loss_c =  (F.mse_loss(t11, data.y1) +F.mse_loss(t21, data.y1) )\
                  + (F.mse_loss(t12, data.y2)  + F.mse_loss(t22, data.y2))\
                  + (F.mse_loss(t13, data.y3)  + F.mse_loss(t23, data.y3))\
                  + (F.mse_loss(t14, data.y4)  + F.mse_loss(t24, data.y4))\
                  + (F.mse_loss(t15, data.y5)  + F.mse_loss(t25, data.y5)) + (F.mse_loss(t16, data.y6)  + F.mse_loss(t26, data.y6))







        # print(step)
        loss_test = contrastive_node_module(x1_class[:,0::,:], x2_class[:,0::,:] )


        # loss_test = torch.sum(1-F.cosine_similarity(t1_node[:,1::,:],t2_node[:,1::,:],dim=2))



        # print(loss_test)

        #############loss_constractive

        #t1_node[:,1::,:], t2_node[:,1::,:]  t1_node[:,2,:]  t2_node[:,2,:]



        # loss = opt.lamb0*loss_c

        # loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        # loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        # loss_tpk1 = topk_loss(s1,opt.ratio)
        # loss_tpk2 = topk_loss(s2,opt.ratio)
        # loss_consist = 0
        # for c in range(opt.nclass):
        #     loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_test # + opt.lamb2 * loss_p2 \
                  # + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        # writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        # writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        # writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        # writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        # writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        # writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        # s1_arr = np.hstack(s1_list)
        # s2_arr = np.hstack(s2_list)
    return loss_all / len(train_dataset) #, s1_arr, s2_arr ,w1,w2


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    # pred_score = torch.tensor([[0]], device=device)
    pred_score1 = torch.tensor([0], device=device)
    pred_score2 = torch.tensor([0], device=device)
    pred_score3 = torch.tensor([0], device=device)
    pred_score4 = torch.tensor([0], device=device)
    pred_score5 = torch.tensor([0], device=device)
    pred_score6 = torch.tensor([0], device=device)

    label_score1 = torch.tensor([0], device=device)
    label_score2 = torch.tensor([0], device=device)
    label_score3 = torch.tensor([0], device=device)
    label_score4 = torch.tensor([0], device=device)
    label_score5 = torch.tensor([0], device=device)
    label_score6 = torch.tensor([0], device=device)

    for data in loader:
        data = data.to(device)
        # outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        data_t1w = data.x[:,0:9]
        data_fmri = data.x[:,9:]
        # t11, t21, t1_node, t2_node = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9), data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))

        t11, t21, t12, t22,t13, t23,t14, t24,t15, t25,t16, t26, t1_node, t2_node, x1_class,x2_class  = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9), data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))


        data.y1 = torch.tensor(data.y[:, 0], dtype=torch.float)
        data.y2 = torch.tensor(data.y[:, 1], dtype=torch.float)
        data.y3 = torch.tensor(data.y[:, 2], dtype=torch.float)

        data.y4 = torch.tensor(data.y[:, 3], dtype=torch.float)
        data.y5 = torch.tensor(data.y[:, 4], dtype=torch.float)
        data.y6 = torch.tensor(data.y[:, 5], dtype=torch.float)



        pred_score1 = torch.cat((pred_score1, t11.data), dim=0)
        pred_score2 = torch.cat((pred_score2, t12.data), dim=0)
        pred_score3 = torch.cat((pred_score3, t13.data), dim=0)

        pred_score4 = torch.cat((pred_score4, t11.data), dim=0)
        pred_score5 = torch.cat((pred_score5, t12.data), dim=0)
        pred_score6 = torch.cat((pred_score6, t13.data), dim=0)


        label_score1 = torch.cat((label_score1, data.y1), dim=0)
        label_score2 = torch.cat((label_score2, data.y2), dim=0)
        label_score3 = torch.cat((label_score3, data.y3), dim=0)
        label_score4 = torch.cat((label_score4, data.y4), dim=0)
        label_score5 = torch.cat((label_score5, data.y5), dim=0)
        label_score6 = torch.cat((label_score6, data.y6), dim=0)


    # epoch_mae = torch.mean(torch.abs(pred_score.reshape(-1, 1) - label_score.reshape(-1, 1)))


    # correct = np.corrcoef(pred_score[1:,0].detach().cpu().numpy().T, label_score[1:].detach().cpu().numpy().T)[0,1]
    correct1 = np.corrcoef(pred_score1[1:].detach().cpu().numpy().T, label_score1[1:].detach().cpu().numpy().T)[0,1]
    correct2 = np.corrcoef(pred_score2[1:].detach().cpu().numpy().T, label_score2[1:].detach().cpu().numpy().T)[0, 1]
    correct3 = np.corrcoef(pred_score3[1:].detach().cpu().numpy().T, label_score3[1:].detach().cpu().numpy().T)[0, 1]
    correct4 = np.corrcoef(pred_score4[1:].detach().cpu().numpy().T, label_score4[1:].detach().cpu().numpy().T)[0,1]
    correct5 = np.corrcoef(pred_score5[1:].detach().cpu().numpy().T, label_score5[1:].detach().cpu().numpy().T)[0, 1]
    correct6 = np.corrcoef(pred_score6[1:].detach().cpu().numpy().T, label_score6[1:].detach().cpu().numpy().T)[0, 1]

    correct = (correct1 + correct2 + correct3+ correct4+ correct5+ correct6)/6

    print('corr')
    print(correct1*10, correct2*10, correct3*10, correct4*10, correct5*10, correct6*10 )


    ap = pred_score1[1:,].detach().cpu().numpy()
    al = label_score1[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse1 = np.sqrt(((ap-al) **2).mean())

    ap = pred_score2[1:,].detach().cpu().numpy()
    al = label_score2[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse2 = np.sqrt(((ap-al) **2).mean())

    ap = pred_score3[1:,].detach().cpu().numpy()
    al = label_score3[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse3 = np.sqrt(((ap-al) **2).mean())

    ap = pred_score4[1:,].detach().cpu().numpy()
    al = label_score4[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse4 = np.sqrt(((ap-al) **2).mean())


    ap = pred_score5[1:,].detach().cpu().numpy()
    al = label_score5[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse5 = np.sqrt(((ap-al) **2).mean())

    ap = pred_score6[1:,].detach().cpu().numpy()
    al = label_score6[1:].detach().cpu().numpy()
    ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    al = (al - np.min(al)) / (np.max(al) - np.min(al))
    epoch_rmse6 = np.sqrt(((ap-al) **2).mean())

    print('rmse')
    print(epoch_rmse1*10, epoch_rmse2*10, epoch_rmse3*10, epoch_rmse4*10, epoch_rmse5*10, epoch_rmse6*10)


    print('test1111111111111')
    # ap = pred_score6[1:,].detach().cpu().numpy()
    # al = label_score6[1:].detach().cpu().numpy()
    # ap = (ap - np.min(ap)) / (np.max(ap) - np.min(ap))
    # al = (al - np.min(al)) / (np.max(al) - np.min(al))
    # np.save('pred_score6', ap)
    # np.save('label_score6', al)


    #plt.scatter(pred_score1[1:].detach().cpu().numpy().T, label_score1[1:].detach().cpu().numpy().T)
    #plt.scatter(pred_score2[1:].detach().cpu().numpy().T, label_score2[1:].detach().cpu().numpy().T)
    #plt.scatter(pred_score3[1:].detach().cpu().numpy().T, label_score3[1:].detach().cpu().numpy().T)


    # np.save('pred_score1', pred_score1[1:, ].detach().cpu().numpy().T)
    # np.save('label_score1', label_score1[1:].detach().cpu().numpy().T)


    return correct    #/ len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)

        # t1 = model(data.x.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))
        data_t1w = data.x[:,0:9]
        data_fmri = data.x[:,9:]
        # t11, t21, t1_node, t2_node = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9), data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))

        t11, t21, t12, t22, t13, t23, t14, t24, t15, t25, t16, t26, t1_node, t2_node, x1_class, x2_class = model(
            data_t1w.view(int(data.x.shape[0] / opt.nroi), opt.nroi, 9),
            data_fmri.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi))

        # t2 = model(data_fmri.view(int(data.x.shape[0]/opt.nroi),opt.nroi, opt.nroi))
        # t2 = model(data_t1w.view(int(data.x.shape[0]/opt.nroi),opt.nroi, 9))

        # img = torch.randn(1,3,333,333).to(device)
        # model_v(img)

        # output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        # s1_list.append(s1.view(-1).detach().cpu().numpy())
        # s2_list.append(s2.view(-1).detach().cpu().numpy())
        data.y1 = torch.tensor(data.y[:, 0], dtype=torch.float)
        data.y2 = torch.tensor(data.y[:, 1], dtype=torch.float)
        data.y3 = torch.tensor(data.y[:, 2], dtype=torch.float)

        data.y4 = torch.tensor(data.y[:, 3], dtype=torch.float)
        data.y5 = torch.tensor(data.y[:, 4], dtype=torch.float)
        data.y6 = torch.tensor(data.y[:, 5], dtype=torch.float)

        # loss_c = F.nll_loss(output, data.y)
        # loss_c = F.mse_loss(t1[:,0,0], data.y)     #right

        # loss_c =  (F.mse_loss(t11, data.y1) +F.mse_loss(t21, data.y1) )\
        #           + 2* (F.mse_loss(t12, data.y2)  + F.mse_loss(t22, data.y2))\
        #           + (F.mse_loss(t13, data.y3)  + F.mse_loss(t23, data.y3))

        loss_c = (F.mse_loss(t11, data.y1)) \
                 + (F.mse_loss(t12, data.y2)) \
                 + (F.mse_loss(t13, data.y3)) \
                 + (F.mse_loss(t14, data.y4)) \
                 + (F.mse_loss(t15, data.y5)) \
                 + (F.mse_loss(t16, data.y6))


        loss_test = contrastive_node_module(x1_class[:,0::,:], x2_class[:,0::,:] )

        # loss_c = F.nll_loss(output, data.y)
        # loss_c = F.mse_loss(t1[:,0,0], data.y)     #right
        # loss_c = F.mse_loss(t1, data.y) + F.mse_loss(t2, data.y)
        # loss_test = torch.sum(1-F.cosine_similarity(t1_node[:,1::,:],t2_node[:,1::,:],dim=2))

        # loss = opt.lamb0*loss_c

        # loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        # loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        # loss_tpk1 = topk_loss(s1,opt.ratio)
        # loss_tpk2 = topk_loss(s2,opt.ratio)
        # loss_consist = 0
        # for c in range(opt.nclass):
        #     loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_test  # + opt.lamb2 * loss_p2 \






        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
for epoch in range(0, num_epoch):

    # tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
    tr_loss= train(epoch)

    tr_acc = test_acc(train_loader)
    since = time.time()
    val_acc = test_acc(val_loader)
    time_elapsed = time.time() - since
    val_loss = test_loss(val_loader,epoch)

    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc, val_loss, val_acc))

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
    # writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
    # writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

    if val_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if save_model:
            torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

if opt.load_model:
    model = Network_regress_score(opt.indim,opt.ratio,opt.nclass).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
    model.eval()
    preds = []
    correct = 0
    for data in val_loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(1)[1]
        preds.append(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
    preds = np.concatenate(preds,axis=0)
    trues = val_dataset.data.y.cpu().detach().numpy()
    cm = confusion_matrix(trues,preds)
    print("Confusion matrix")
    print(classification_report(trues, preds))

else:
   model.load_state_dict(best_model_wts)
   model.eval()
   test_accuracy = test_acc(test_loader)
   test_l= test_loss(test_loader,0)
   print("===========================")
   print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
   print(opt)



