import torch.nn as nn
import torch as t
import numpy as np

class FM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.FC1 = nn.Linear(dim * 2, dim)
        self.FC2 = nn.Linear(dim, 1)
    
    def forward(self, emb_seq1, emb_seq2):
        # emb_seq1: 512 * 5 * 50
        # emb_seq2: 512 * 100 * 50
        viewer = t.zeros([emb_seq1.size(0), emb_seq1.size(1), emb_seq2.size(1), emb_seq1.size(-1)]).cuda() # 512 * 5 * 100 * 50
        emb_seq1 = emb_seq1.unsqueeze(2) + viewer   # 512 * 5 * 100 * 50
        emb_seq2 = emb_seq2.unsqueeze(1) + viewer   # 512 * 5 * 100 * 50
        emb = t.cat([emb_seq1, emb_seq2], -1)    
        temp = self.FC1(emb_seq1).tanh()
        print(emb.size(), temp.size())
        input()
        return self.FC2(self.FC1(emb_seq1).tanh()).unsqueeze()



class MM(nn.Module):

    def __init__(self, num_item, num_user, dim_emb = 50, len_seq = 5):
        super().__init__()
    
        self.EMB_ITEM = nn.Embedding(num_item, dim_emb)
        self.EMB_USER = nn.Embedding(num_user, dim_emb)
        #self.EMB_TIME = nn.Embedding(len_seq + 1,  dim_emb)
        #self.FM = FM(dim_emb)
        #self.time = t.tensor(range(len_seq + 1)).cuda().unsqueeze(0)        # 1 * 6
        #self.FC1, self.FC2 = nn.Linear(dim_emb, dim_emb), nn.Linear(dim_emb, dim_emb)

    def forward(self, users, sequences, pos_targets, neg_targets):
        #emb_tim = self.EMB_TIME(self.time)                                  # 1 * 6 * 50
        emb_seq = self.EMB_ITEM(sequences) #+ emb_tim[:,:5,:]                # 512 * 5 * 50 
        #emb_t_l = emb_tim[:,-1,:].squeeze().unsqueeze(0).unsqueeze(0)
        
        emb_can = self.EMB_ITEM(t.cat([pos_targets, neg_targets], 1))# + emb_t_l      # 512 * (1 + 100) * 50
        emb_neg = self.EMB_ITEM(neg_targets)# + emb_t_l                               # 512 * 100 * 50
        emb_usr = self.EMB_USER(users).unsqueeze(1)
        
        # i2i
        sim1 = emb_seq @ emb_can.permute(0,2,1) / (emb_seq.size(-1) ** .5)   # 512 * 5 * (1 + 100)
        sim2 = emb_usr @ emb_can.permute(0,2,1) / (emb_seq.size(-1) ** .5)   # 512 * 1 * (1 + 100)
        #sim = self.FM(emb_seq, emb_can)
        #sim = t.log_softmax(sim1 + sim2, -1)
        sim = t.log_softmax(sim1, -1)

        # i2s
        context = self._attention(emb_seq, emb_neg)                         # 512 * 5 * 1

        return sim + context                                                # 512 * 5 * (1 + 100)

    def _mask(self, len_pos, len_neg):
        return 1 - t.cat([t.eye(len_pos), t.zeros(len_neg, len_pos)], 0).cuda()

    def _attention(self, emb_seq, emb_neg, beta = 1):          # emb_seq: 512 * 5 * 50 
                                                               # emb_neg: 512 * 100 * 50
        dim_root = emb_neg.size(-1) ** .5
        mask = self._mask(emb_seq.size(1), emb_neg.size(1)).unsqueeze(0)# 1 * (5 + 100) * 5

        emb_can = t.cat([emb_seq, emb_neg], 1)                          # 512 * (5 + 100) * 50
        sim = emb_can @ emb_seq.permute(0,2,1) / dim_root               # 512 * (5 + 100) * 5
        sim = (sim * mask).softmax(-1).detach()

        context = (emb_seq.unsqueeze(1) * sim.unsqueeze(-1)).sum(-2)    # 512 * (5 + 100) * 50
        context_seq, context_neg = context[:,:5,:], context[:,5:,:]     # context_seq: 512 * 5 * 50
                                                                        # context_neg: 512 * 100 * 50
        
        weights_seq = emb_seq @ context_seq.permute(0,2,1) / dim_root   # 512 * 5 * 5
        #weights_seq = self.FM(emb_seq, context_seq)
        mask = t.eye(weights_seq.size(-1)).to(weights_seq.device).unsqueeze(0) 
        weights_seq = (weights_seq * mask).sum(-1, keepdim=True)        # 512 * 5 * 1
        weights_neg = emb_seq @ context_neg.permute(0,2,1) / dim_root   # 512 * 5 * 100
        #weights_neg = self.FM(emb_seq, context_neg)


        weights = t.cat([weights_seq, weights_neg], -1)                 # 512 * 5 * (1 + 100)
        weights = weights.log_softmax(-1)[:,:,0].unsqueeze(-1)          # 512 * 5 * 1
        
        return weights