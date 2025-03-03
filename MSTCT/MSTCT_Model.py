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
        self.fc1 = nn.Conv1d(output_dim*2, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, output1, output2):
        combined_outputs = torch.cat((output1, output2), dim=1)

        hidden = F.relu(self.fc1(combined_outputs))
        gate = torch.sigmoid(self.fc2(hidden))


        #print('I am here: ', gate.shape)
        return gate

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

        self.dropout1=nn.Dropout()
        self.dropout2 =nn.Dropout()

        self.TemporalEncoder1=TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block)
        self.TemporalEncoder2 = TemporalEncoder(in_feat_dim=768, embed_dims=inter_channels,
                                               num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                               num_block=num_block)

        self.Temporal_Mixer1=Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)
        self.Temporal_Mixer2 = Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)
        self.GatingMechanism1 = GatingMechanism(2048, 32)
        self.GatingMechanism2 = GatingMechanism(2048, 32)
        # self.attention1 = nn.MultiheadAttention(256, 8).to(device)
        # self.attention2 = nn.MultiheadAttention(384, 8).to(device)
        # self.attention3 = nn.MultiheadAttention(576, 8).to(device)
        # self.attention4 = nn.MultiheadAttention(864, 8).to(device)
        self.linear = nn.Linear(1568, 768)

    def forward(self, inputs_rgb, inputs_flow):
        # inputs_rgb = self.dropout1(inputs_rgb)
        # inputs_flow = self.dropout2(inputs_flow)
        inputs_flow = inputs_flow.permute(0, 2, 1)
        inputs_flow = self.linear(inputs_flow)
        inputs_flow = inputs_flow.permute(0, 2, 1)

        inputs_rgb = self.TemporalEncoder1(inputs_rgb)
        inputs_flow = self.TemporalEncoder2(inputs_flow)
        # Temporal Encoder Module
        # x_rgb = self.TemporalEncoder1(inputs_rgb)
        # x_flow = self.TemporalEncoder2(inputs_flow)





        # Reshape for nn.MultiheadAttention: (sequence_length, batch_size, embed_dim)
        # rgb_reshaped = [x_rgb[0].transpose(1, 2), x_rgb[1].transpose(1, 2), x_rgb[2].transpose(1, 2), x_rgb[3].transpose(1, 2)]
        # flow_reshaped = [x_flow[0].transpose(1, 2), x_flow[1].transpose(1, 2), x_flow[2].transpose(1, 2), x_flow[3].transpose(1, 2)]
        #
        #
        #
        # attn_output_rgb1, _ = self.attention1(flow_reshaped[0], rgb_reshaped[0], rgb_reshaped[0])
        # attn_output_rgb2, _ = self.attention2(flow_reshaped[1], rgb_reshaped[1], rgb_reshaped[1])
        # attn_output_rgb3, _ = self.attention3(flow_reshaped[2], rgb_reshaped[2], rgb_reshaped[2])
        # attn_output_rgb4, _ = self.attention4(flow_reshaped[3], rgb_reshaped[3], rgb_reshaped[3])
        #
        # # # Compute cross-attention: flow as query, rgb as key and value
        # # attn_output_flow, _ = attention(flow_reshaped, rgb_reshaped, rgb_reshaped)
        #
        # # Reshape back to original shape: (batch_size, sequence_length, embed_dim)
        # output = [attn_output_rgb1.permute(0, 2, 1), attn_output_rgb2.permute(0, 2, 1),  attn_output_rgb3.permute(0, 2, 1), attn_output_rgb4.permute(0, 2, 1)]







        # Temporal Scale Mixer Module
        concat_feature, concat_feature_hm = self.Temporal_Mixer1(inputs_rgb)

        concat_feature_flow, concat_feature_hm_flow = self.Temporal_Mixer1(inputs_flow)

        beta = self.GatingMechanism1(concat_feature, concat_feature_flow)
        gama = self.GatingMechanism2(concat_feature_hm, concat_feature_hm_flow)

        concat_feature = beta * concat_feature + (1-beta) * concat_feature_flow
        concat_feature_hm = gama * concat_feature_hm + (1-gama) * concat_feature_hm_flow
        # # Classification Module
        x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

        return x, x_hm # B, T, C





