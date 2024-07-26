import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from builder import NETS
from model import Transformer,iTransformer,Informer,Reformer,Flowformer,Flashformer,FEDformer

from utils import get_attr
from types import SimpleNamespace



class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

        self.naf   = nn.LeakyReLU()

    def forward(self, inputs):
        # inputs: (batch, num_features, num_nodes, window_len)
        part1 = inputs.permute(0, 2, 1, 3)
        part2 = inputs.permute(0, 2, 3, 1)
        part1 = self.bn_w1(self.W1(part1).squeeze(-1))
        part1 = self.bn_w2(self.W2(part1))
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)  #
        # Maven's Fix - Clip the computed scores and replace ReLU with LeakyReLU
        score = torch.bmm(part1, part2)
        score = self.V(self.naf(score))
        score = torch.clip(score, max=10,min=-10)
        S = torch.softmax(score, dim=-1)
        return S

class Portattention(nn.Module):
    def __init__(self,c_out,pred_len,num_stocks):
        super(Portattention,self).__init__()
        # if self.attention_bool == True:
        #     self.attention = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout)
        #     self.query_projection = nn.Linear(self.num_stocks, self.d_model, bias=True)
        #     self.value_projection = nn.Linear(self.num_stocks, self.d_model, bias=True)
        #     self.key_projection = nn.Linear(self.num_stocks, self.d_model, bias=True)
        #     self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        #
        #     self.final_projection = nn.Linear(self.d_model, self.num_stocks, bias=True)
        # else:
        self.layer_norm = nn.LayerNorm(c_out, eps=1e-5)
        self.featureconv = nn.Conv1d(c_out, 1, kernel_size=3, padding=1)
        self.fc_mu = nn.Linear(pred_len, 1, bias=True)
        self.fc_std = nn.Linear(pred_len,1, bias=True)
    def forward(self,x):
        norm_scores = self.layer_norm(x).permute(0, 2, 1)
        conv_scores = self.featureconv(norm_scores).squeeze(1)
        mu = self.fc_mu(conv_scores).squeeze(-1)
        std = F.softplus(self.fc_std(conv_scores)).squeeze(-1)
        return mu,std



@NETS.register_module()
class AAAI_mse(nn.Module):
    def __init__(self,**kwargs):
        super(AAAI_mse, self).__init__()
        # self.model_name = model_name
        namespace = SimpleNamespace(**kwargs)
        self.model_name = get_attr(kwargs, "model", 'Transformer')
        self.attention_bool = get_attr(kwargs,'attention_bool',False)
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'FEDformer': FEDformer,}
        #for REINFORCE algorithm
        self.data = []
        self.optimizer = get_attr(kwargs,'optimizer','Adam')

        self.dropout = nn.Dropout(get_attr(kwargs,'dropout',0.05))
        self.model = self.model_dict[self.model_name].Model(SimpleNamespace(**kwargs))
        self.d_model = get_attr(kwargs, "d_model", 512)
        self.n_heads= get_attr(kwargs, "d_model", 8)
        self.num_stocks = get_attr(kwargs, "num_stocks", 29)
        self.seq_len = get_attr(kwargs,'seq_len',20)
        self.pred_len = get_attr(kwargs, 'pred_len', 5)
        self.label_len = get_attr(kwargs, 'label_len', 5)
        self.c_out = get_attr(kwargs,'c_out',16)

        self.linear_feature = nn.Linear(self.c_out,1)
        self.linear_time = nn.Linear(self.pred_len,1)
    def forward(self,  x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        inputs: [batch, num_stock, window_len, num_features]
        mask: [batch, num_stock]
        outputs: [batch, scores]
        """
        # Reshape inputs for BERT

        x_enc = torch.squeeze(x_enc,dim=0)
        x_mark_enc = torch.squeeze(x_mark_enc, dim=0)
        x_dec = torch.squeeze(x_dec, dim=0)
        x_mark_dec = torch.squeeze(x_mark_dec, dim=0)
        # decoder input
        dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.label_len, :], dec_inp], dim=1).float()
        pred_logit = self.model(x_enc,x_mark_enc,dec_inp,x_mark_dec)
        #
        pred_label = pred_logit[:, :, -1]
        #decoder
        # pred_time_scores = self.linear_feature(pred_label).squeeze(-1)
        # pred_scores = self.linear_time(pred_time_scores).squeeze(-1)


        return pred_label


# Example usage:
if __name__ == '__main__':
    # Example data
    a = torch.randn((16, 30, 16, 5))
    b = torch.randn((16, 30, 16, 5))
    # Instantiate the model
    num_nodes = 30  # Number of nodes (stocks)
    window_len = 16  # Length of the window
    # hidden_dim = 128  # Hidden dimension of BERT
    hidden_dim = 80  # Hidden dimension of BERT
    num_layers = 4  # Number of BERT layers
    dropout = 0.3  # Dropout rate
    model = ASU(num_nodes, window_len, hidden_dim, num_layers, dropout)
    # Forward pass
    output = model(a, b)
    print(output)