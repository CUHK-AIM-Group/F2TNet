import torch
from imports.vit import *
import torch.nn.functional as F
import torch.nn as nn

class Network_regress_score_dual(torch.nn.Module):
    def __init__(self, fmri_indim,t1w_indim, fmri_outdim, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_dual, self).__init__()
        self.fmri_indim = fmri_indim
        self.t1w_indim = t1w_indim
        self.fmri_outdim = fmri_outdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim


        # t1w
        self.fc11 = torch.nn.Linear(self.t1w_indim, 64)
        self.fc12 = torch.nn.Linear(64, 1024)
        self.bn11 = torch.nn.BatchNorm1d(64)
        self.bn12 = torch.nn.BatchNorm1d(1024)
        #fmri
        self.fc21 = torch.nn.Linear(self.fmri_indim, 512)
        self.fc22 = torch.nn.Linear(512, 1024)
        self.bn21 = torch.nn.BatchNorm1d(512)
        self.bn22 = torch.nn.BatchNorm1d(1024)
        #####

        self.model_v = ViT_transout(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)
        # self.model_v = ViT_real(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)

        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 1)




    def forward(self, x1, x2):


        ###t1w
        x1 = F.relu(self.fc11(x1))
        x1 = F.dropout(x1, p=0.6, training=self.training)
        x1 = self.bn11(x1.transpose(1, 2)).transpose(1, 2)
        x1= F.relu(self.fc12(x1))
        x1 = F.dropout(x1, p=0.6, training=self.training)
        x1 = self.bn12(x1.transpose(1, 2)).transpose(1, 2)

        ###fMRI
        x2 = F.relu(self.fc21(x2))
        x2 = F.dropout(x2, p=0.6, training=self.training)
        x2 = self.bn21(x2.transpose(1, 2)).transpose(1, 2)
        x2 = F.relu(self.fc22(x2))
        x2 = F.dropout(x2, p=0.6, training=self.training)
        x2 = self.bn22(x2.transpose(1, 2)).transpose(1, 2)




        x1, x1_node = self.model_v(x1)   #
        x2, x2_node = self.model_v(x2)   #

        x1 = self.fc3(x1)
        x1 = self.fc4(x1)
        x2 = self.fc3(x2)
        x2 = self.fc4(x2)




        return x1[:,0], x2[:,0] , x1_node, x2_node #x[:,0,0] #x[:,0]

class Network_regress_score_fMRI(torch.nn.Module):
    def __init__(self, fmri_indim, fmri_outdim, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_fMRI, self).__init__()
        self.fmri_indim = fmri_indim
        self.fmri_outdim = fmri_outdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim



        self.fc1 = torch.nn.Linear(self.fmri_indim, 512)
        self.fc2 = torch.nn.Linear(512, 1024)
        self.model_v = ViT_real(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(1024)


    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))

        ######################################


        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.model_v(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # x = self.fc5(x[:,:,0])
        # x = self.fc5(x.transpose(1,2))

        return x[:,0]



class Network_regress_score_dual_multi(torch.nn.Module):
    def __init__(self, fmri_indim,t1w_indim, fmri_outdim, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_dual_multi, self).__init__()
        self.fmri_indim = fmri_indim
        self.t1w_indim = t1w_indim
        self.fmri_outdim = fmri_outdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim


        # t1w
        self.fc11 = torch.nn.Linear(self.t1w_indim, 64)
        self.fc12 = torch.nn.Linear(64, 1024)
        self.bn11 = torch.nn.BatchNorm1d(64)
        self.bn12 = torch.nn.BatchNorm1d(1024)
        #fmri
        self.fc21 = torch.nn.Linear(self.fmri_indim, 512)
        self.fc22 = torch.nn.Linear(512, 1024)
        self.bn21 = torch.nn.BatchNorm1d(512)
        self.bn22 = torch.nn.BatchNorm1d(1024)
        #####

        self.model_v = ViT_transout_multi(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)

        self.fc31 = torch.nn.Linear(1024, 256)
        self.fc32 = torch.nn.Linear(1024, 256)
        self.fc33= torch.nn.Linear(1024, 256)
        self.fc34 = torch.nn.Linear(1024, 256)
        self.fc35 = torch.nn.Linear(1024, 256)
        self.fc36 = torch.nn.Linear(1024, 256)

        self.fc41 = torch.nn.Linear(256, 1)
        self.fc42 = torch.nn.Linear(256, 1)
        self.fc43 = torch.nn.Linear(256, 1)
        self.fc44 = torch.nn.Linear(256, 1)
        self.fc45 = torch.nn.Linear(256, 1)
        self.fc46 = torch.nn.Linear(256, 1)

    def forward(self, x1, x2):


        ###t1w
        x1 = F.relu(self.fc11(x1))
        x1 = F.dropout(x1, p=0.6, training=self.training)
        x1 = self.bn11(x1.transpose(1, 2)).transpose(1, 2)
        x1= F.relu(self.fc12(x1))
        x1 = F.dropout(x1, p=0.6, training=self.training)
        x1 = self.bn12(x1.transpose(1, 2)).transpose(1, 2)

        ###fMRI
        x2 = F.relu(self.fc21(x2))
        x2 = F.dropout(x2, p=0.6, training=self.training)
        x2 = self.bn21(x2.transpose(1, 2)).transpose(1, 2)
        x2 = F.relu(self.fc22(x2))
        x2 = F.dropout(x2, p=0.6, training=self.training)
        x2 = self.bn22(x2.transpose(1, 2)).transpose(1, 2)




        x11, x12, x13, x14, x15, x16, x1_node = self.model_v(x1)
        x21, x22, x23, x24, x25, x26, x2_node = self.model_v(x2)




        x1_class = torch.cat((x11.unsqueeze(1), x12.unsqueeze(1), x13.unsqueeze(1)),dim=1)
        x2_class = torch.cat((x21.unsqueeze(1), x22.unsqueeze(1), x23.unsqueeze(1)),dim=1)

        x11 = self.fc31(x11)
        x11 = self.fc41(x11)
        x21 = self.fc31(x21)
        x21 = self.fc41(x21)

        x12 = self.fc32(x12)
        x12 = self.fc42(x12)
        x22 = self.fc32(x22)
        x22 = self.fc42(x22)

        x13 = self.fc33(x13)
        x13 = self.fc43(x13)
        x23 = self.fc33(x23)
        x23 = self.fc43(x23)

        x14 = self.fc33(x14)
        x14 = self.fc43(x14)
        x24 = self.fc33(x24)
        x24 = self.fc43(x24)

        x15 = self.fc33(x15)
        x15 = self.fc43(x15)
        x25 = self.fc33(x25)
        x25 = self.fc43(x25)

        x16 = self.fc33(x16)
        x16 = self.fc43(x16)
        x26 = self.fc33(x26)
        x26 = self.fc43(x26)



        return x11[:,0], x21[:,0], x12[:,0], x22[:,0], x13[:,0], x23[:,0], x14[:,0], x24[:,0],x15[:,0], x25[:,0], x16[:,0], x26[:,0],x1_node, x2_node, x1_class, x2_class #x[:,0,0] #x[:,0]


class Network_regress_score_t1w(torch.nn.Module):
    def __init__(self, fmri_indim, fmri_outdim, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_t1w, self).__init__()
        self.fmri_indim = fmri_indim
        self.fmri_outdim = fmri_outdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim



        self.fc1 = torch.nn.Linear(self.fmri_indim, 64)

        self.fc2 = torch.nn.Linear(64, 1024)

        self.model_v = ViT_real(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)


        self.fc3 = torch.nn.Linear(1024, 256)

        self.fc4 = torch.nn.Linear(256, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(1024)

        # self.fc5 = torch.nn.Linear(333, 1)








    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))

        ######################################


        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        x = self.model_v(x)

        x = self.fc3(x)

        x = self.fc4(x)
        # x = self.fc5(x[:,:,0])
        # x = self.fc5(x.transpose(1,2))


        return x[:,0] #x[:,0,0] #x[:,0]





class Network_regress_score_trans(torch.nn.Module):
    def __init__(self, fmri_indim, fmri_outdim, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_trans, self).__init__()
        self.fmri_indim = fmri_indim
        self.fmri_outdim = fmri_outdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim




        self.fc1 = torch.nn.Linear(self.fmri_indim, 512)

        self.fc2 = torch.nn.Linear(512, 1024)

        self.model_v = ViT(image_size = self.image_size, patch_size = self.patch_size, num_classes = self.num_classes, dim = self.dim, depth = self.depth, heads = self.heads, mlp_dim = self.mlp_dim)


        self.fc3 = torch.nn.Linear(1024, 512)

        self.fc4 = torch.nn.Linear(512, 1)
        self.fc5 = torch.nn.Linear(333, 1)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(512)







    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))

        ######################################


        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        x = self.model_v(x)



        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)

        x = self.fc4(x)
        # x = self.fc5(x[:,:,0])
        x = self.fc5(x.transpose(1,2))





        return x



class Network_regress_score_fMRI_MLP(torch.nn.Module):
    def __init__(self, fmri_indim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_regress_score_fMRI_MLP, self).__init__()
        self.fmri_indim = fmri_indim


        self.fc1 = torch.nn.Linear(self.fmri_indim, 512)

        self.fc2 = torch.nn.Linear(512, 1024)

        self.fc3 = torch.nn.Linear(1024, 512)

        self.fc4 = torch.nn.Linear(512, 1)
        self.fc5 = torch.nn.Linear(333, 1)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(512)





    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))

        ######################################


        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)

        x = self.fc4(x)
        # x = self.fc5(x[:,:,0])
        x = self.fc5(x.transpose(1,2))




        return x


##########################################################################################################################





