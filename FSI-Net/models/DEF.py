import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# # Cross Attention Module
# # class CrossAttention(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(CrossAttention, self).__init__()
# #         self.query_weight = nn.Parameter(torch.randn(input_dim, output_dim))
# #         self.key_weight = nn.Parameter(torch.randn(input_dim, output_dim))
# #         self.value_weight = nn.Parameter(torch.randn(input_dim, output_dim))

# #     def forward(self, Q, K, V):
# #         # 假设 Q, K, V 的形状为 (batch_size, channels, height, width)
# #         batch_size, channels, height, width = Q.size()

# #         # 展平空间维度为 seq_len (height * width)，将 Q, K, V 从 (batch_size, channels, height, width) 转换为 (batch_size, seq_len, channels)
# #         seq_len = height * width
# #         Q = Q.view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)
# #         K = K.view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)
# #         V = V.view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)

# #         # 计算查询、键和值的加权
# #         Q = torch.matmul(Q, self.query_weight)  # (batch_size, seq_len, output_dim)
# #         K = torch.matmul(K, self.key_weight)    # (batch_size, seq_len, output_dim)
# #         V = torch.matmul(V, self.value_weight)  # (batch_size, seq_len, output_dim)

# #         # 计算交叉注意力分数
# #         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
# #         attention_weights = F.softmax(attention_scores, dim=-1)

# #         # 通过注意力权重加权求和得到输出
# #         output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, output_dim)

# #         # 恢复输出形状到 (batch_size, channels, height, width)
# #         output = output.view(batch_size, channels, height, width)  # 使得通道数与输入一致

# #         return output
# # class CrossAttention(nn.Module):
# #     def __init__(self, in_channels, emb_dim):
# #         super(CrossAttention, self).__init__()
# #         self.emb_dim = emb_dim
# #         self.scale = emb_dim ** -0.5
 
# #         self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
 
# #         self.Wq = nn.Linear(emb_dim, emb_dim)
# #         self.Wk = nn.Linear(emb_dim, emb_dim)
# #         self.Wv = nn.Linear(emb_dim, emb_dim)
 
# #         self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
 
# #     def forward(self, x, context, pad_mask=None):
# #         '''
# #         :param x: [batch_size, c, h, w]
# #         :param context: [batch_szie, seq_len, emb_dim]
# #         :param pad_mask: [batch_size, seq_len, seq_len]
# #         :return:
# #         '''
# #         b, c, h, w = x.shape
 
# #         x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
# #         x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
# #         context = self.proj_in(context)   # [batch_size, c, h, w] = [3, 512, 512, 512]
# #         context = rearrange(context, 'b c h w -> b (h w) c') 

# #         Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
# #         K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
# #         V = self.Wv(context)
 
# #         # [batch_size, h*w, seq_len]
# #         att_weights = torch.einsum('bid,bjd -> bij', Q, K)
# #         att_weights = att_weights * self.scale
 
# #         if pad_mask is not None:
# #             # [batch_size, h*w, seq_len]
# #             att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
# #         att_weights = F.softmax(att_weights, dim=-1)
# #         out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]
 
# #         out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
# #         out = self.proj_out(out)   # [batch_size, c, h, w]
 
# #         print(out.shape)
 
# #         return out, att_weights

class CrossAttention(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(CrossAttention, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.query_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))
        self.key_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))
        self.value_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))

    def forward(self, Q, K, V):
        # 假设 Q, K, V 的形状为 (batch_size, channels, height, width)
        batch_size, channels, height, width = Q.size()

        # 展平空间维度为 seq_len (height * width)，将 Q, K, V 从 (batch_size, channels, height, width) 转换为 (batch_size, seq_len, channels)
        seq_len = height * width
        Q = Q.contiguous().view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)
        K = K.contiguous().view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)
        V = V.contiguous().view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)

        # 计算查询、键和值的加权
        Q = torch.matmul(Q, self.query_weight)
        K = torch.matmul(K, self.key_weight)
        V = torch.matmul(V, self.value_weight)

        # 计算交叉注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 通过注意力权重加权求和得到输出
        output = torch.matmul(attention_weights, V)
        bottleneck_dim = self.bottleneck_dim
        output = output.contiguous().view(batch_size, bottleneck_dim, height, width)  # 使得通道数与输入一致
        return output


# Residual Block (Bottleneck)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1)
        
        # 将 skip 连接的通道数与输入通道数一致
        self.skip = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)  # 保持与输入通道数一致
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x + residual  # 返回残差块输出


# Difference Feature Extractor (DFE)
class DifferenceFeatureExtractor(nn.Module):
    def __init__(self, input_dim,out_channels):
        super(DifferenceFeatureExtractor, self).__init__()
        output_dim = input_dim
        self.batch_norm = nn.BatchNorm2d(input_dim)  # 批处理归一化
        self.cross_attention = CrossAttention(input_dim, output_dim)  # 交叉注意力
        self.Conv3x3 = nn.Conv2d(2*output_dim,input_dim,kernel_size=3,padding=1)
        self.BR = nn.Sequential(
                                ResidualBlock(input_dim, input_dim),
                                ResidualBlock(input_dim, input_dim),
                                ResidualBlock(input_dim, input_dim),
                                nn.ReLU()  
        )
        self.res_block1 = ResidualBlock(input_dim, 2*output_dim)  # 第一个瓶颈残差块
        self.res_block2 = ResidualBlock(input_dim, 2*output_dim)  # 第二个瓶颈残差块
        self.layer_norm = nn.LayerNorm(input_dim)  # 层归一化
        self.conv1x1 = nn.Conv2d(input_dim, out_channels, kernel_size=1)
    def forward(self, f_l_i, f_hat_l_i):
        # 批处理归一化
        f_l_i = self.batch_norm(f_l_i)
        f_hat_l_i = self.batch_norm(f_hat_l_i)
        
        # 计算查询、键和值
        Q1 = f_hat_l_i
        K1 = f_l_i
        V1 = f_l_i

        Q2 = f_l_i
        K2 = f_hat_l_i
        V2 = f_hat_l_i

        # 交叉注意力
        attention_output1 = self.cross_attention(Q1, K1, V1)
        attention_output2 = self.cross_attention(Q2, K2, V2)
        # attention_output = self.Conv3x3(torch.cat((attention_output1,attention_output2), dim=1))
        attention_output = attention_output1 + attention_output2
        # 通过瓶颈残差块
        # res_output1 = self.BR(attention_output)
        # res_output = attention_output + res_output1
        # 层归一化
        # batch_size, channels, height, width = res_output.size()
        # x = res_output.view(batch_size, channels, -1)  # 展平空间维度 -> (batch_size, channels, height*width)
        # x = res_output.permute(0, 2, 3, 1)
        # x = self.layer_norm(x)  # 层归一化
        # x = x.permute(0, 3, 1, 2)
        # 恢复空间维度
        # x = x.view(batch_size, channels, height, width)
        # output = self.layer_norm(res_output)
        x = self.conv1x1(attention_output)
        return x

# Example usage
input_dim = 512
# bottleneck_dim = 256
# output_dim = 256

# 假设输入特征图大小为 (batch_size, input_dim, height, width)
f_l_i = torch.randn(2, input_dim, 16, 16)  # 特征图 f_l_i
f_hat_l_i = torch.randn(2, input_dim, 16, 16)  # 特征图 f_hat_l_i
out_channels=512
dfe = DifferenceFeatureExtractor(input_dim,out_channels)
def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

output = dfe(f_l_i, f_hat_l_i)

print(output.shape)  # 输出的特征图大小

# import torch.nn as nn
# import torch
# # 论文：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# # 论文地址：https://arxiv.org/pdf/2405.06880

# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()
#     channels_per_group = num_channels // groups
#     # reshape
#     x = x.view(batchsize, groups,
#                channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     # flatten
#     x = x.view(batchsize, -1, height, width)
#     return x

# def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
#     # activation layer
#     act = act.lower()
#     if act == 'relu':
#         layer = nn.ReLU(inplace)
#     elif act == 'relu6':
#         layer = nn.ReLU6(inplace)
#     elif act == 'leakyrelu':
#         layer = nn.LeakyReLU(neg_slope, inplace)
#     elif act == 'prelu':
#         layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
#     elif act == 'gelu':
#         layer = nn.GELU()
#     elif act == 'hswish':
#         layer = nn.Hardswish(inplace)
#     else:
#         raise NotImplementedError('activation layer [%s] is not found' % act)
#     return layer

# # Efficient up-convolution block (EUCB)
# class EUCB(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
#         super(EUCB, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.up_dwc = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
#                       padding=kernel_size // 2, groups=self.in_channels, bias=False),
#             nn.BatchNorm2d(self.in_channels),
#             act_layer(activation, inplace=True)
#         )
#         self.pwc = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         )

#     def forward(self, x):
#         x = self.up_dwc(x)
#         x = channel_shuffle(x, self.in_channels)
#         x = self.pwc(x)
#         return x
