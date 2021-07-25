import sys
import copy
import random
import numpy as np
from collections import defaultdict


class Interaction:
    def __init__(self, num_item, num_user):
        self.num_item = num_item   
        self.num_user = num_user 
        self.usrs, self.seqs, self.tars = [], [], []
    def append(self, usr, seq, tar):
        self.usrs.append(usr)
        self.seqs.append(seq)
        self.tars.append(tar)

def data_partition(fname, sequence_length, target_length):
    print(fname)
    # read file
    num_item, num_user = 0, 0
    DATA = defaultdict(list)
    for line in open('data/%s.txt' % fname, 'r'):
        u, i = [int(ele) for ele in line.rstrip().split()]
        num_item = max(num_item, i)        
        num_user = max(num_user, u)        
        DATA[u].append(i)
    num_item += 1 # 0 for padding
    num_user += 1 # 0 for padding
        
    # sequentialize
    train_interaction, test_interaction = Interaction(num_item, num_user), Interaction(num_item, num_user)
    for key in DATA:
        sequence = DATA[key]
        if len(sequence) < sequence_length + target_length + 1: continue
        
        num_sequence = len(sequence) + 1 - (sequence_length + target_length)

        for i in range(num_sequence):
            if i != num_sequence - 1:
                train_interaction.append(key, sequence[i:i+sequence_length], sequence[i+sequence_length:i+sequence_length+target_length])
            else: 
                test_interaction.append(key, sequence[i:i+sequence_length], sequence[i+sequence_length:i+sequence_length+target_length])
    return train_interaction, test_interaction

class Reader:
    def __init__(self, interaction, len_neg):
        self.usr, self.seq, self.tar = np.array(interaction.usrs, dtype=np.int64), np.array(interaction.seqs, dtype=np.int64), np.array(interaction.tars, dtype=np.int64)
        self.num_itm = interaction.num_item
        self.len_neg = len_neg

    def __len__(self):
        return len(self.usr)
    
    def __getitem__(self, idx):
        usr, seq, pos_tar = self.usr[idx], self.seq[idx], self.tar[idx]
        neg_tar = np.random.randint(1, self.num_itm, [self.len_neg], dtype=np.int64)
        return usr, seq, pos_tar, neg_tar



def logs(msg):
    clearline = '\b' * (len(msg) + 5)
    sys.stdout.write(clearline + msg)