import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder
import torch
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GatingMechanism(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(GatingMechanism, self).__init__()
        self.fc1 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)



    def forward(self, output1, output2, output3, output4, output5):



        hidden1 = F.relu(self.fc1(output1))
        hidden2 = F.relu(self.fc1(output2))
        hidden3 = F.relu(self.fc1(output3))
        hidden4 = F.relu(self.fc1(output4))
        hidden5 = F.relu(self.fc1(output5))



        expert1 =  1- torch.sigmoid(self.fc2(hidden1))
        expert2 = 1 - torch.sigmoid(self.fc2(hidden2))
        expert3 = 1 - torch.sigmoid(self.fc2(hidden3))
        expert4 = 1 - torch.sigmoid(self.fc2(hidden4))
        expert5 = 1 - torch.sigmoid(self.fc2(hidden5))





        return expert1, expert2, expert3, expert4, expert5

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1568, 768)

    def forward(self, x):



        x = x.permute(0, 2, 1)
        # Apply the encoder
        z = self.linear(x)  # Output shape: (batch_size * feature_dim, 768)
        z = z.permute(0, 2, 1)



        return z
class MSTCT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes):
        super(MSTCT, self).__init__()


        self.TemporalEncoder1 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block)

        self.TemporalEncoder2 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                               num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                               num_block=num_block)
        self.TemporalEncoder3 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                                num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                                num_block=num_block)
        self.TemporalEncoder4 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                                num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                                num_block=num_block)
        self.TemporalEncoder5 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                                num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                                num_block=num_block)
        self.TemporalEncoder6 = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                                num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                                num_block=num_block)

        self.Temporal_Mixer1=Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)


        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)

        self.GatingMechanism1 = GatingMechanism(256, 32)
        self.GatingMechanism2 = GatingMechanism(384, 32)
        self.GatingMechanism3 = GatingMechanism(576, 32)
        self.GatingMechanism4 = GatingMechanism(864, 32)


        #Transformations
        self.linear_flow = nn.Linear(1568, 768)
        self.linear_depth = nn.Linear(128, 768)
        self.linear_pose = nn.Linear(256, 768)
        self.linear_SAM = nn.Linear(256, 768)
        self.linear_VLM = nn.Linear(512, 768)

        self.temperature = 3.0
        # self.KL = nn.MSELoss()
        self.KL =  nn.L1Loss()
        # self.KL = nn.KLDivLoss(reduction="batchmean")
        self.klm = False




    def forward(self, inputs_rgb, inputs_flow, inputs_depth, inputs_pose, inputs_SAM, inputs_VLM, is_train):

        if is_train:
            temperature = self.temperature

            inputs_depth = inputs_depth.permute(0, 2, 1)
            inputs_depth = self.linear_depth(inputs_depth)
            inputs_depth = inputs_depth.permute(0, 2, 1)

            inputs_flow = inputs_flow.permute(0, 2, 1)
            inputs_flow = self.linear_flow(inputs_flow)
            inputs_flow = inputs_flow.permute(0, 2, 1)

            inputs_pose = inputs_pose.permute(0, 2, 1)
            inputs_pose = self.linear_pose(inputs_pose)

            inputs_SAM = inputs_SAM.permute(0, 2, 1)
            inputs_SAM = self.linear_SAM(inputs_SAM)
            inputs_SAM = inputs_SAM.permute(0, 2, 1)

            inputs_VLM = inputs_VLM.permute(0, 2, 1)
            inputs_VLM = self.linear_VLM(inputs_VLM)
            inputs_VLM = inputs_VLM.permute(0, 2, 1)


            inputs_rgb = self.TemporalEncoder1(inputs_rgb)
            inputs_flow = self.TemporalEncoder2(inputs_flow)
            inputs_depth = self.TemporalEncoder3(inputs_depth)
            inputs_pose = self.TemporalEncoder4(inputs_pose)
            inputs_SAM = self.TemporalEncoder5(inputs_SAM)
            inputs_VLM = self.TemporalEncoder6(inputs_VLM)

            beta1, gamma1, zeta1, lambda1, yps1 = self.GatingMechanism1(inputs_flow[0], inputs_depth[0], inputs_pose[0],
                                                                        inputs_SAM[0], inputs_VLM[0])
            beta2, gamma2, zeta2, lambda2, yps2 = self.GatingMechanism2(inputs_flow[1], inputs_depth[1], inputs_pose[1],
                                                                        inputs_SAM[1], inputs_VLM[1])
            beta3, gamma3, zeta3, lambda3, yps3 = self.GatingMechanism3(inputs_flow[2], inputs_depth[2], inputs_pose[2],
                                                                        inputs_SAM[2], inputs_VLM[2])
            beta4, gamma4, zeta4, lambda4, yps4 = self.GatingMechanism4(inputs_flow[3], inputs_depth[3], inputs_pose[3],
                                                                        inputs_SAM[3], inputs_VLM[3])
            #
            inputs_flow = [beta1 * inputs_flow[0], beta2 * inputs_flow[1], beta3 * inputs_flow[2], beta4 * inputs_flow[3]]
            inputs_depth = [gamma1 * inputs_depth[0], gamma2 * inputs_depth[1], gamma3 * inputs_depth[2], gamma4 * inputs_depth[3]]
            inputs_pose = [zeta1 * inputs_pose[0], zeta2 * inputs_pose[1], zeta3 * inputs_pose[2], zeta4 * inputs_pose[3]]
            inputs_SAM = [lambda1 * inputs_SAM[0], lambda2 * inputs_SAM[1], lambda3 * inputs_SAM[2], lambda4 * inputs_SAM[3]]
            inputs_VLM = [yps1 * inputs_VLM[0], yps2 * inputs_VLM[1], yps3 * inputs_VLM[2], yps4 * inputs_VLM[3]]


            if self.klm:
                KL_1 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_flow[0] / temperature, dim=1))* (temperature ** 2)
                KL_2 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_flow[1] / temperature, dim=1))* (temperature ** 2)
                KL_3 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_flow[2] / temperature, dim=1))* (temperature ** 2)
                KL_4 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_flow[3] / temperature, dim=1))* (temperature ** 2)
                KL_1_mean = (KL_1 + KL_2 + KL_3 + KL_4)

                KL_5 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_depth[0] / temperature, dim=1)) * (temperature ** 2)
                KL_6 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_depth[1] / temperature, dim=1)) * (temperature ** 2)
                KL_7 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_depth[2] / temperature, dim=1)) * (temperature ** 2)
                KL_8 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_depth[3] / temperature, dim=1)) * (temperature ** 2)
                KL_2_mean = (KL_5 + KL_6 + KL_7 + KL_8)

                KL_9 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_pose[0] / temperature, dim=1)) * (temperature ** 2)
                KL_10 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_pose[1] / temperature, dim=1)) * (temperature ** 2)
                KL_11 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_pose[2] / temperature, dim=1)) * (temperature ** 2)
                KL_12 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_pose[3] / temperature, dim=1)) * (temperature ** 2)
                KL_3_mean = (KL_9 + KL_10 + KL_11 + KL_12)

                KL_13 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_SAM[0] / temperature, dim=1)) * (temperature ** 2)
                KL_14 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_SAM[1] / temperature, dim=1)) * (temperature ** 2)
                KL_15 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_SAM[2] / temperature, dim=1)) * (temperature ** 2)
                KL_16 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_SAM[3] / temperature, dim=1)) * (temperature ** 2)
                KL_4_mean = (KL_13 + KL_14 + KL_15 + KL_16)

                KL_17 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_VLM[0] / temperature, dim=1)) * (temperature ** 2)
                KL_18 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_VLM[1] / temperature, dim=1)) * (temperature ** 2)
                KL_19 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_VLM[2] / temperature, dim=1)) * (temperature ** 2)
                KL_20 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_VLM[3] / temperature, dim=1)) * (temperature ** 2)
                KL_5_mean = (KL_17 + KL_18 + KL_19 + KL_20)

            else:
                KL_1 =self.KL(inputs_rgb[0], inputs_flow[0])
                KL_2 = self.KL(inputs_rgb[1], inputs_flow[1]) # feature base knowdledge distilation
                KL_3 = self.KL(inputs_rgb[2], inputs_flow[2])
                KL_4 = self.KL(inputs_rgb[3], inputs_flow[3])
                KL_1_mean = (KL_1 + KL_2 + KL_3 + KL_4)

                KL_5 = self.KL(inputs_rgb[0], inputs_depth[0])
                KL_6 = self.KL(inputs_rgb[1], inputs_depth[1])
                KL_7 = self.KL(inputs_rgb[2], inputs_depth[2])
                KL_8 = self.KL(inputs_rgb[3], inputs_depth[3])
                KL_2_mean = (KL_5 + KL_6 + KL_7 + KL_8)

                KL_9 = self.KL(inputs_rgb[0], inputs_pose[0])
                KL_10 = self.KL(inputs_rgb[1], inputs_pose[1])
                KL_11 = self.KL(inputs_rgb[2], inputs_pose[2])
                KL_12 = self.KL(inputs_rgb[3], inputs_pose[3])
                KL_3_mean = (KL_9 + KL_10 + KL_11 + KL_12)

                KL_13 = self.KL(inputs_rgb[0], inputs_SAM[0])
                KL_14 = self.KL(inputs_rgb[1], inputs_SAM[1])
                KL_15 = self.KL(inputs_rgb[2], inputs_SAM[2])
                KL_16 = self.KL(inputs_rgb[3], inputs_SAM[3])
                KL_4_mean = (KL_13 + KL_14 + KL_15 + KL_16)

                KL_17 = self.KL(inputs_rgb[0], inputs_VLM[0])
                KL_18 = self.KL(inputs_rgb[1], inputs_VLM[1])
                KL_19 = self.KL(inputs_rgb[2], inputs_VLM[2])
                KL_20 = self.KL(inputs_rgb[3], inputs_VLM[3])
                KL_5_mean = (KL_17 + KL_18 + KL_19 + KL_20)


            # KL_1 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_flow[0] / temperature, dim=1))* (temperature ** 2)
            # KL_2 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_flow[1] / temperature, dim=1))* (temperature ** 2)
            # KL_3 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_flow[2] / temperature, dim=1))* (temperature ** 2)
            # KL_4 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_flow[3] / temperature, dim=1))* (temperature ** 2)
            # KL_1_mean = (KL_1 + KL_2 + KL_3 + KL_4)
            #
            # KL_5 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_depth[0] / temperature, dim=1)) * (temperature ** 2)
            # KL_6 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_depth[1] / temperature, dim=1)) * (temperature ** 2)
            # KL_7 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_depth[2] / temperature, dim=1)) * (temperature ** 2)
            # KL_8 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_depth[3] / temperature, dim=1)) * (temperature ** 2)
            # KL_2_mean = (KL_5 + KL_6 + KL_7 + KL_8)
            #
            # KL_9 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_pose[0] / temperature, dim=1)) * (temperature ** 2)
            # KL_10 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_pose[1] / temperature, dim=1)) * (temperature ** 2)
            # KL_11 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_pose[2] / temperature, dim=1)) * (temperature ** 2)
            # KL_12 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_pose[3] / temperature, dim=1)) * (temperature ** 2)
            # KL_3_mean = (KL_9 + KL_10 + KL_11 + KL_12)
            #
            # KL_13 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_SAM[0] / temperature, dim=1)) * (temperature ** 2)
            # KL_14 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_SAM[1] / temperature, dim=1)) * (temperature ** 2)
            # KL_15 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_SAM[2] / temperature, dim=1)) * (temperature ** 2)
            # KL_16 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_SAM[3] / temperature, dim=1)) * (temperature ** 2)
            # KL_4_mean = (KL_13 + KL_14 + KL_15 + KL_16)
            #
            # KL_17 = self.KL(torch.log_softmax(inputs_rgb[0] / temperature, dim=1), torch.softmax(inputs_VLM[0] / temperature, dim=1)) * (temperature ** 2)
            # KL_18 = self.KL(torch.log_softmax(inputs_rgb[1] / temperature, dim=1), torch.softmax(inputs_VLM[1] / temperature, dim=1)) * (temperature ** 2)
            # KL_19 = self.KL(torch.log_softmax(inputs_rgb[2] / temperature, dim=1), torch.softmax(inputs_VLM[2] / temperature, dim=1)) * (temperature ** 2)
            # KL_20 = self.KL(torch.log_softmax(inputs_rgb[3] / temperature, dim=1), torch.softmax(inputs_VLM[3] / temperature, dim=1)) * (temperature ** 2)
            # KL_5_mean = (KL_17 + KL_18 + KL_19 + KL_20)

            # MSE_1 = self.MSE(inputs_rgb[0], inputs_flow[0])
            # MSE_2 = self.MSE(inputs_rgb[1], inputs_flow[1])  # feature base knowdledge distilation
            # MSE_3 = self.MSE(inputs_rgb[2], inputs_flow[2])
            # MSE_4 = self.MSE(inputs_rgb[3], inputs_flow[3])
            # MSE_1_mean = (MSE_1 + MSE_2 + MSE_3 + MSE_4)
            #
            # MSE_5 = self.MSE(inputs_rgb[0], inputs_depth[0])
            # MSE_6 = self.MSE(inputs_rgb[1], inputs_depth[1])
            # MSE_7 = self.MSE(inputs_rgb[2], inputs_depth[2])
            # MSE_8 = self.MSE(inputs_rgb[3], inputs_depth[3])
            # MSE_2_mean = (MSE_5 + MSE_6 + MSE_7 + MSE_8)
            #
            # MSE_9 = self.MSE(inputs_rgb[0], inputs_pose[0])
            # MSE_10 = self.MSE(inputs_rgb[1], inputs_pose[1])
            # MSE_11 = self.MSE(inputs_rgb[2], inputs_pose[2])
            # MSE_12 = self.MSE(inputs_rgb[3], inputs_pose[3])
            # MSE_3_mean = (MSE_9 + MSE_10 + MSE_11 + MSE_12)
            #
            # MSE_13 = self.MSE(inputs_rgb[0], inputs_SAM[0])
            # MSE_14 = self.MSE(inputs_rgb[1], inputs_SAM[1])
            # MSE_15 = self.MSE(inputs_rgb[2], inputs_SAM[2])
            # MSE_16 = self.MSE(inputs_rgb[3], inputs_SAM[3])
            # MSE_4_mean = (MSE_13 + MSE_14 + MSE_15 + MSE_16)
            #
            # MSE_17 = self.MSE(inputs_rgb[0], inputs_VLM[0])
            # MSE_18 = self.MSE(inputs_rgb[1], inputs_VLM[1])
            # MSE_19 = self.MSE(inputs_rgb[2], inputs_VLM[2])
            # MSE_20 = self.MSE(inputs_rgb[3], inputs_VLM[3])
            # MSE_5_mean = (MSE_17 + MSE_18 + MSE_19 + MSE_20)

            # KL_total = KL_1_mean + KL_2_mean + KL_3_mean + KL_4_mean + KL_5_mean



            # Temporal Scale Mixer Module
            concat_feature, concat_feature_hm = self.Temporal_Mixer1(inputs_rgb)

            x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

            return x, x_hm, 0#, KL_total # B, T, C
        else:

            # Temporal Encoder Module
            x = self.TemporalEncoder1(inputs_rgb)

            # Temporal Scale Mixer Module
            concat_feature, concat_feature_hm = self.Temporal_Mixer1(x)
            x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

            return x, x_hm, 0 # B, T, C






