# -*- coding: utf-8 -*-
# @Time   : 2022/7/15
# @Author : Victor Chen
# @Email  : vic4code@gmail.com
# @File   : fasformer.py


r"""
Fastformer model(Additive attention can be all you need)
################################################
Reference:
    Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: 
    Additive attention can be all you need. arXiv preprint arXiv:2108.09084.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from recbole.model.abstract_recommender import ContextRecommender


class Fastformer(ContextRecommender):
    def __init__(self, hparams):
        super(Fastformer, self).__init__()
        self.hparams = hparams
        self.dense_linear = nn.Linear(hparams.hidden_size, hparams.output_size)
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.word_emb_dim, padding_idx=0)
        self.encoder = FastformerEncoder(hparams)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids, targets):
        mask = input_ids.bool().float()
        embds = self.word_embedding(input_ids)
        text_vec = self.encoder(embds, mask)
        score = self.dense_linear(text_vec)

        return score

class AttentionPooling(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(hparams.hidden_size, hparams.hidden_size)
        self.att_fc2 = nn.Linear(hparams.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, hparams):
        super(FastSelfAttention, self).__init__()
        self.hparams = hparams
        if hparams.hidden_size % hparams.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hparams.hidden_size, hparams.num_attention_heads))
        self.attention_head_size = int(hparams.hidden_size /hparams.num_attention_heads)
        self.num_attention_heads = hparams.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= hparams.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads * self.attention_head_size)
        pooled_query_repeat = pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat
        
        query_key_score = (self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value

class FastAttention(nn.Module):
    def __init__(self, hparams):
        super(FastAttention, self).__init__()
        self.selfattn = FastSelfAttention(hparams)
        self.output = nn.Sequential(
                                nn.Linear(hparams.hidden_size, hparams.hidden_size),
                                nn.Dropout(hparams.hidden_dropout_prob)
                            )
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        self_output = self.selfattn(input_tensor, attention_mask)
        attention_output = self.output(self_output)
        attention_output = self.LayerNorm(attention_output + input_tensor)

        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, hparams):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(hparams)
        self.intermediate = nn.Sequential(
                                nn.Linear(hparams.hidden_size, hparams.intermediate_size),
                                nn.GELU()
                            )
        self.output = nn.Sequential(
                                nn.Linear(hparams.intermediate_size, hparams.hidden_size),
                                nn.Dropout(hparams.hidden_dropout_prob)
                            )
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return layer_output
    
class FastformerEncoder(nn.Module):
    def __init__(self, hparams, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.hparams = hparams
        self.encoders = nn.ModuleList([FastformerLayer(hparams) for _ in range(hparams.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(hparams.max_position_embeddings, hparams.hidden_size)
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        self.dropout = nn.Dropout(hparams.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if hparams.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(hparams))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype = next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device = input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 
    