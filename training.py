"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import time
import numpy as np
from sklearn import cluster

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args, cluster_centers):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.cluster_centers = cluster_centers
        self.temperature_centers = self.args.temperature_centers
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
        
    # def train_step_virtual(self, input_ids, attention_mask, centers, i):
        
    #     embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

    #     # Instance-CL loss
    #     feat1, feat2 = self.model.contrast_logits(embd1, embd2)
    #     losses = self.contrast_loss(feat1, feat2)
    #     loss = self.eta * losses["loss"]

    #     # 更新聚类中心
    #     centers_old = torch.tensor(centers)
    #     k = self.args.num_classes
    #     for plm in range(1):
    #         output = self.model.get_cluster_prob(embd1.cuda(), torch.tensor(centers).cuda())
    #         label = torch.argmin(output, 1)
    #         for j in range(k):
    #             Q = output[:,j][label==j]
    #             Q = Q.unsqueeze(1)
    #             centers[j] = torch.sum(Q*embd1[label==j], dim=0) / (float(torch.sum(Q)))
    #         centers = torch.tensor(centers)
    #         # centers = F.normalize(centers, dim=1)*10
    #     centers_new = centers
    #     centers = centers_old + 0.1* (centers_new - centers_old) 
    #     # print(centers)
    #     # centers = centers_old + (0.25-(0.15*i/self.args.max_iter)) * (centers_new - centers_old)

    #     # Clustering loss
    #     if self.args.objective == "SCCL":
    #         output = self.model.get_cluster_prob(embd1.cuda(), torch.tensor(centers).cuda())
    #         target = target_distribution(output)
            
    #         cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
    #         # alpha = ((6*i/self.args.max_iter)*(1-i/self.args.max_iter)/cluster_loss).detach() # 二次函数，最大值为6/4=1.5，凸函数停留在高数值的迭代次数过多
    #         # alpha = (min(5,(10*np.exp(-0.5*(i-self.args.max_iter/2)**2)))/cluster_loss).detach() # exp函数，峰值后下降速度过快无意义，峰值为0.1
    #         # alpha = ((5*(1-2/self.args.max_iter*np.abs(i-self.args.max_iter/2)))/cluster_loss).detach()
    #         if cluster_loss<0.001:
    #             alpha = 0.005/cluster_loss.detach()
    #         else:
    #             alpha = 5
    #         loss += alpha*cluster_loss
    #         losses["cluster_loss"] = alpha*cluster_loss.item()

    #     loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     return losses, centers
    

    def train_step_explicit(self, input_ids, attention_mask, centers, i):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = losses["loss"]

        # 更新聚类中心
        centers_old = torch.tensor(centers)
        k = self.args.num_classes
        for plm in range(1):
            output = self.model.get_cluster_prob(embd1.cuda(), torch.tensor(centers).cuda())
            label = torch.argmin(output, 1)
            for j in range(k):
                Q = output[:,j][label==j]
                Q = Q.unsqueeze(1)
                centers[j] = torch.sum(Q*embd1[label==j], dim=0) / (float(torch.sum(Q)))
            centers = torch.where(torch.isnan(centers), torch.full_like(centers, 0), centers)
            centers = torch.tensor(centers)
            # centers = F.normalize(centers, dim=1)*10
        centers_new = centers
        centers = centers_old + 0.05* (centers_new - centers_old) 
        # centers = centers_old + (0.15-(0.05*i/self.args.max_iter)) * (centers_new - centers_old)

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1.cuda(), torch.tensor(centers).cuda())
            target = target_distribution(output)
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            # alpha = ((6*i/self.args.max_iter)*(1-i/self.args.max_iter)/cluster_loss).detach() # 二次函数，最大值为6/4=1.5，凸函数停留在高数值的迭代次数过多
            # alpha = (min(5,(10*np.exp(-0.5*(i-self.args.max_iter/2)**2)))/cluster_loss).detach() # exp函数，峰值后下降速度过快无意义，峰值为0.1
            # alpha = ((5*(1-2/self.args.max_iter*np.abs(i-self.args.max_iter/2)))/cluster_loss).detach()
            # if cluster_loss<0.001:
            #     alpha = 0.01/cluster_loss.detach()
            # else:
            #     alpha = 10
            # alpha = 1
            # self.eta = alpha
            loss += self.eta * cluster_loss
            losses["cluster_loss"] = self.eta * cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses, centers
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        centers = torch.tensor(self.cluster_centers)
        Iterations_Batches = self.args.max_iter / len(self.train_loader)

        for i in np.arange(self.args.max_iter+1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            if (i%Iterations_Batches==0) and (self.temperature_centers>0.05):
                self.temperature_centers = self.temperature_centers/1.5

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses, centers = self.train_step_virtual(input_ids, attention_mask, centers, i) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask, centers, i)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                self.evaluate_embedding(i, centers)
                self.model.train()

        return None   

    
    def evaluate_embedding(self, step, centers):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings, torch.tensor(centers).cuda())
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        
        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embeddings = all_embeddings.cpu().numpy()

        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
        
        # clustering accuracy 
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc":acc, "acc_model":acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        print('[Representation] Clustering scores:',confusion.clusterscores()) 
        print('[Representation] ACC: {:.3f}'.format(acc)) 
        print('[Model] Clustering scores:',confusion_model.clusterscores()) 
        print('[Model] ACC: {:.3f}'.format(acc_model))

        return None



             