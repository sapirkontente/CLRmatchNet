import torch
import random
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from matchnet.model import MatchNet
from clrnet.engine.scheduler import build_scheduler
from clrnet.models.losses.focal_loss import FocalLoss
from clrnet.models.losses.lineiou_loss import liou_loss

# Train matchnet in a teacher-student setup
class TrainMatchNet(): 
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = MatchNet().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.matchnet_lr)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.bce_loss = nn.BCELoss()
        self.batch_size = self.cfg.matchnet_batch_size # number of lanes per batch
        self.j = 1 # image index to select lanes from 
        self.img_w = self.cfg.img_w
        self.n_strips = self.cfg.num_points - 1
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        self.fb_iou_weight = self.cfg.fb_iou_weight
        self.fb_loss_thresh = self.cfg.fb_loss_thresh       
            
    def extract_data(self, data, output, output_predictions):
        """
        Selects b lanes out of the j image in the last stage (=2). 
        Extracts the following features for each gt-pred pair: delta theta, delta xy start, delta dist, neg_cls, pos_cls, iou. 
        Extracts the classic cost function match for each gt-pred pair.
        Calculates the feedback loss to each gt-pred pair.
        
        Returns an updated gt match for each gt-pred pair. 
        """
        # lanes
        targets = data['lane_line'][-self.j]
        predictions = output_predictions[-1][-self.j]

        # matchnet
        img_name = [q['full_img_path'] for item in data['meta'].data for q in item][-self.j] # image name
        pair_features = output['pair_features_batch'][:,-self.j:,:].reshape(-1,6) # delta theta, delta xy start, delta dist, neg_cls, pos_cls, iou
        pair_match = output['pair_match_batch'][:,-self.j:,:].reshape(-1,1).long() # gt-pred match = {0,1}
        sel_gt_ind = output['sel_gt_ind_batch'][:,-self.j:,:].reshape(-1,1).long() # sel_gt_ind = {0,1,2,3} 
        sel_pred_ind = output['sel_pred_ind_batch'][-self.j*self.batch_size:]  # lane index out of 192 predictions
        
        # update gt-pred match by the loss feedback
        self.loss_feedback(data, predictions, pair_match, sel_gt_ind, sel_pred_ind, self.j, self.batch_size)
        
        return pair_match, pair_features 

    def loss_feedback(self, data, predictions, pair_match, sel_gt_ind, sel_pred_ind, j=1, b=18):
        """
        For positive matches only - calculating the loss feedback of clrnet. 
        Pairs with loss higher than the threshold are updated to be negative matches
        """
        for m in range(j):
            sel_pred_ind[m*b:(m+1)*b] = [i+m*192 for i in sel_pred_ind[m*b:(m+1)*b]]
        
        mask_true,_ = torch.where(pair_match==1) # only positive pairs
        sel_pred_true = [sel_pred_ind[i] for i in mask_true]
        sel_gt_true = [sel_gt_ind[i] for i in mask_true]
        
        cls_loss, reg_xytl_loss, iou_loss = self.pair_loss(predictions, data['lane_line'][-1], sel_pred_true, sel_gt_true)
        fb_loss = (cls_loss + self.fb_iou_weight *iou_loss)
        pair_match_fb = torch.tensor([0] * len(sel_pred_true)).cuda()

        pair_match_fb[fb_loss<=self.fb_loss_thresh]=1
        pair_match[mask_true] = pair_match_fb[:,None].long()

    def pair_loss(self, predictions, target, matched_row_inds, matched_col_inds):
        
        """
        Return the loss for each gt-pred pair.
        """
        # classification targets
        cls_target = predictions.new_zeros(predictions.shape[0]).long()
        matched_row_inds = [int(i) for i in matched_row_inds]
        cls_target[matched_row_inds] = 1
        cls_pred = predictions[matched_row_inds, :2]             

        # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
        reg_yxtl = predictions[matched_row_inds, 2:6]
        reg_yxtl[:, 0] *= self.n_strips
        reg_yxtl[:, 1] *= (self.img_w - 1)
        reg_yxtl[:, 2] *= 180
        reg_yxtl[:, 3] *= self.n_strips
        target_yxtl = target[matched_col_inds, 2:6].clone()

        # regression targets -> S coordinates (all transformed to absolute values)
        reg_pred = predictions[matched_row_inds, 6:]
        reg_pred *= (self.img_w - 1)
        reg_targets = target[matched_col_inds, 6:].clone()
        with torch.no_grad():
            predictions_starts = torch.clamp(
                (predictions[matched_row_inds, 2] *
                 self.n_strips).round().long(), 0,
                self.n_strips)  # ensure the predictions starts is valid
            target_starts = (target[matched_col_inds, 2] *
                             self.n_strips).round().long()
            target_yxtl[:, -1] -= (predictions_starts - target_starts
                                   )  # reg lengt

        # Loss calculation
        if cls_pred.numel()!=0: # only not empty assignments
            cls_loss = self.focal_loss(cls_pred, cls_target[matched_row_inds])
            
            target_yxtl[:, 0] *= self.n_strips
            target_yxtl[:, 2] *= 180
            
            reg_xytl_loss = F.smooth_l1_loss(
                            reg_yxtl, target_yxtl,
                        reduction='none')

            iou_loss = liou_loss(
                        reg_pred, reg_targets,
                        self.img_w, length=15)

        return cls_loss,reg_xytl_loss,iou_loss
   
    def train_batch(self, data, output, output_predictions):
        """
        Train batch of matchNet
        """

        # Generate data
        pair_match, pair_features  = self.extract_data(data, output, output_predictions)

        # matchnet forward
        deep_match = self.model(pair_features.detach())
        loss = self.bce_loss(deep_match, pair_match.detach().float()) 

        # matchnet backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not self.cfg.lr_update_by_epoch:
            self.scheduler.step()
        
        return loss