from reader import *
from mm import *

from torch.utils.data import DataLoader
import torch.optim as optim
import torch as t
import argparse
import sys

'''fname: ['Beauty', 'Video', 'Games']'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, help = 'dataset name from Amazon: [Beaty Video Games]', default = 'Beauty')

parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--feature_dim', type=int, default=200)
parser.add_argument('--wd', type = float, default = 1e-5)

parser.add_argument('--sequence_length', type=int, default=5)
parser.add_argument('--target_length', type=int, default=1)
parser.add_argument('--negative_length', type=int, default=100)
args = parser.parse_args()

train_interaction, test_interaction = data_partition(args.dataset, args.sequence_length, args.target_length)
trainset = Reader(train_interaction, args.negative_length)
testset = Reader(test_interaction, args.negative_length)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=args.batch_size, shuffle=True, pin_memory=True)

model = MM(train_interaction.num_item, train_interaction.num_user, dim_emb = args.feature_dim).cuda()
optimizer = optim.Adam(model.parameters(), weight_decay= args.wd)
for epoch in range(args.epoch):
   
    for idx, (users, sequences, pos_targets, neg_targets) in enumerate(trainloader):
        users, sequences, pos_targets, neg_targets = users.cuda(), sequences.cuda(), pos_targets.cuda(), neg_targets.cuda()
        
        optimizer.zero_grad()

        log_p = model(users, sequences, pos_targets, neg_targets)[:,:,0] # 512 * 5
        loss = -log_p.max(1)[0].mean()
        
        loss.backward()
        optimizer.step()

        logs('EPOCH %d (%d/%d) | loss %.2f' % (epoch, idx, len(trainloader), loss.item()))

        
    print()
    HT, NDCG, TOTAL = 0,0,0
    with t.no_grad():
        for idx, (users, sequences, pos_targets, neg_targets) in enumerate(testloader):
            users, sequences, pos_targets, neg_targets = users.cuda(), sequences.cuda(), pos_targets.cuda(), neg_targets.cuda()
          
            predicts = -model(users, sequences, pos_targets, neg_targets).sum(1) # 512 * (1 + `100)
            rank = predicts.argsort(1).argsort(1)[:,0]
            
            HT += (rank < 10).sum().item()
            NDCG += ((rank < 10).float() / (rank.float() + 2).log2()).sum().item()
            TOTAL += users.size(0)

            logs('%d/%d | HT10 %.3f NDCG10 %.3f' % (idx, len(testloader), HT/TOTAL, NDCG/TOTAL))
        print('\n')