import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 原始线性映射
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # 将每个头的线性映射结果拆分
        q = q.view(x.shape[0], -1, self.n_heads, self.head_dim)
        k = k.view(x.shape[0], -1, self.n_heads, self.head_dim)
        v = v.view(x.shape[0], -1, self.n_heads, self.head_dim)

        # 调整维度使其适应注意力计算
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 实现多头注意力计算
        attn_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_probs, v)

        # 调整维度并合并多头输出
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(x.shape[0], -1, self.d_model)

        return attention_output


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.LeakyReLU(inplace=False)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.feedforward = Feedforward(d_model, d_ff)

    def forward(self, x):
        attn_output = self.self_attn(x)
        ff_output = self.feedforward(attn_output + x)
        return ff_output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

    def split_CDF(self, x):
        # input: spectrum in CS
        # output: CDF and split into few pieces

        x = torch.softmax(x, dim=1)
        x = torch.cumsum(x, dim=1)
        return

    def forward(self, x):           # [batch_size, seq_len, d_model] seq_len 序列个数，d_model每个序列长度
        for layer in self.layers:
            x = layer(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x


class Cross_Pred_Net(nn.Module):
    def __init__(self, n_clip, sub_band):
        super(Cross_Pred_Net, self).__init__()
        self.n_clip = n_clip
        self.sub_band = sub_band
        self.nc_band = sub_band // n_clip

        self.fc1 = nn.Sequential(
            nn.Linear(self.sub_band, self.sub_band),
            nn.LeakyReLU(inplace=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.sub_band, self.sub_band),
            nn.LeakyReLU(inplace=False),
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.sub_band, self.sub_band // 2),
            nn.LeakyReLU(inplace=False),
            nn.Linear(self.sub_band // 2, self.sub_band // 2),
            nn.LeakyReLU(inplace=False),
            nn.Linear(self.sub_band // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, sample, x):

        sample = sample.expand_as(x)

        x1 = sample * x
        x2 = sample - x

        x1 = x1.view(x1.shape[0], 1, self.sub_band)
        x2 = x2.view(x2.shape[0], 1, self.sub_band)

        # for i in range(self.n_clip):
        #     layer = self.fc_layer[i]
        #     x[:, i, :] = layer(x[:, i, :])
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        Pred = self.MLP(x1+x2)

        return Pred

class CombineCov_Net(nn.Module):
    def __init__(self, n_clip, sub_band):
        super(CombineCov_Net, self).__init__()
        self.n_clip = n_clip
        self.sub_band = sub_band
        self.nc_band = sub_band // n_clip

        self.Conv = nn.ModuleList(
            [CombineCov() for i in range(2)]
        )

        self.MLP = nn.Sequential(
            nn.Linear(self.nc_band*2, self.nc_band),
            nn.LeakyReLU(inplace=False),
            nn.Linear(self.nc_band, self.nc_band // 2),
            nn.LeakyReLU(inplace=False),
            nn.Linear(self.nc_band // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, sample, x):
        sample = sample.expand_as(x)
        # sample = sample.view(sample.shape[0], 1, -1)
        # x = x.view(x.shape[0], 1, -1)
        # Cos = F.cosine_similarity(x, sample, dim=1) Cos.unsqueeze(1)
        concate = torch.cat((sample, x), dim=1)

        # y = self.Conv(concate)
        for layer in self.Conv:
            concate = layer(concate)

        Pred = self.MLP(concate.view(concate.shape[0], 1, concate.shape[-1]*2))
        return Pred

class CombineCov(nn.Module):
    def __init__(self):
        super(CombineCov, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)

        x = x + torch.cat((x1, x2), dim=1)
        return x

