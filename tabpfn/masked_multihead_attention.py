import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, seq_attention_mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        #if dim -1 sums to 1 then this is what we need to
        # remove tokens via mask
        if seq_attention_mask is not None:
            attention = attention * seq_attention_mask.view(1,1,-1)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                1e-20)
        return attention.matmul(value), attention


class MaskedMultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.seq_attention_mask = None
        self.avg_attentions = []

    def forward(self, q, k, v, mask=None):
        # given q,k,v
        # is # sequence length  (examples in dataset) x # batch size (number of datasets)
        # x # embedding dim (number of features in dataset)
        q,k,v = q.swapdims(1,0), k.swapdims(1,0), v.swapdims(1,0)
        # now swap batch size first, sequence lengths econd

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attention = ScaledDotProductAttention()(
            q, k, v, mask,
            seq_attention_mask=self.seq_attention_mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        y = y.swapdims(1,0)
        avg_attention = attention.view(
            -1, self.head_num, attention.shape[1], attention.shape[2]).mean(dim=1)
        self.avg_attentions.append(avg_attention.mean(dim=1))
        return y, avg_attention

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )