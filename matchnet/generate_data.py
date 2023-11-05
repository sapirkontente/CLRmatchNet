import torch
import random
import numpy as np
import torch.nn as nn

class GeneratePairs(nn.Module):
    def __init__(self, cfg):
        super(GeneratePairs, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w 
        self.neg_dist_thresh = self.cfg.matchnet_neg_dist_thresh #0.9
        self.pos_dist_thresh = self.cfg.matchnet_pos_dist_thresh # 0.3
        self.batch_size = self.cfg.matchnet_batch_size

    def forward(self, theta_score_deep, start_xys_score_deep, 
                                    distances_score_deep, pos_score_deep, neg_score_deep, 
                                    iou_deep, matched_row_inds, matched_col_inds, b=18, img_w = 800):
        """
        Selecting a random subset of "b" lanes from 192 predictions per image as follows:   
            1. Selecting all "k" ground-truth (gt) and prediction (pred) lane pairs identified as positives by the classic cost function.
            2. Randomly choosing "b-k" additional gt-pred pairs that were initially assigned as negatives by the classic cost function.
    
        For each selected gt-pred pair, we record:
            1. sel_pred_ind - Index of the chosen prediction lane (0-191).
            2. sel_gt_ind - Index of the associated gt lane (0-3)
            3. pair_features - Computed features (delta theta, delta xy start, delta distance, neg_cls, pos_cls, IoU).
            4. pair_match - Binary indicator (1 for matches, 0 for non-matches).
        """
        theta_score_deep, start_xys_score_deep, distances_score_deep = self.norm(theta_score_deep, start_xys_score_deep, distances_score_deep, img_w)
    
        # Random selection of negative gt-pred pairs
        neg_match_pred_ind, neg_match_gt_ind = self.select_negative_matches(distances_score_deep, matched_row_inds, b)
    
        # Positive gt-pred pairs
        pos_match_pred_ind, pos_match_gt_ind = self.select_positive_matches(matched_row_inds, matched_col_inds, b)
        
        # selected gt-pred pairs
        sel_pred_ind = torch.tensor(pos_match_pred_ind + neg_match_pred_ind)
        sel_gt_ind = torch.tensor(pos_match_gt_ind + neg_match_gt_ind)
    
        # Select Features
        pair_features = torch.cat((theta_score_deep[sel_pred_ind,None,sel_gt_ind],start_xys_score_deep[sel_pred_ind,None,sel_gt_ind],\
                                distances_score_deep[sel_pred_ind,None,sel_gt_ind],pos_score_deep[sel_pred_ind,None,sel_gt_ind], \
                                    neg_score_deep[sel_pred_ind,None,sel_gt_ind],iou_deep[sel_pred_ind,None,sel_gt_ind]),dim=1).cuda()   
    
        # Match GT
        pair_match = torch.cat((torch.ones(len(pos_match_gt_ind)),torch.zeros(len(neg_match_gt_ind))))

        # Update positive pairs with high distance to be negative pairs
        mask = torch.where(distances_score_deep[pos_match_pred_ind,pos_match_gt_ind]<self.pos_dist_thresh)[0]
        for i in range(len(mask)): 
            pair_match[mask[i]]=0
    
        # Shuffle
        sel_pred_ind, sel_gt_ind, pair_features, pair_match = self.shuffle(sel_pred_ind, sel_gt_ind, pair_features, pair_match)
    
        return pair_features, sel_gt_ind[:,None], pair_match[:,None], sel_pred_ind 

    def shuffle(self, sel_pred_ind, sel_gt_ind, pair_features, pair_match):
        indices = [*range(len(sel_pred_ind))]
        random.shuffle(indices)
        sel_pred_ind = [sel_pred_ind[i] for i in indices]
        sel_gt_ind = sel_gt_ind[indices].cuda()
        pair_match = pair_match[indices].float().cuda()
        pair_features = pair_features[indices].cuda()
        return sel_pred_ind,sel_gt_ind,pair_features,pair_match

    def select_positive_matches(self, matched_row_inds, matched_col_inds, batch_size):
        pos_match_pred_ind = [int(item) for item in list(matched_row_inds)][:batch_size]
        pos_match_gt_ind = [int(item) for item in list(matched_col_inds)][:batch_size]
        return pos_match_pred_ind, pos_match_gt_ind

    def select_negative_matches(self, distances_score_deep, matched_row_inds, b):
        neg_match_num = np.maximum(b-matched_row_inds.shape[0],0)
        neg_match_pred, neg_match_gt = torch.where(distances_score_deep>0.3)
        neg_match_pred = ([int(s) for s in neg_match_pred if s not in matched_row_inds])
        indices = [*range(len(neg_match_pred))]
        random.shuffle(indices)
        neg_match_pred_ind = [neg_match_pred[i] for i in indices][:neg_match_num]
        neg_match_gt_ind = [int(neg_match_gt[i]) for i in indices][:neg_match_num]
        return neg_match_pred_ind, neg_match_gt_ind

    def norm(self, theta_score_deep, start_xys_score_deep, distances_score_deep, img_w):
        # theta normalization
        theta_score_deep[theta_score_deep>180]=180
        theta_score_deep = 1-theta_score_deep/180
        
        # xys normalization
        start_xys_score_deep[start_xys_score_deep>img_w] = img_w
        start_xys_score_deep = 1-start_xys_score_deep/img_w
        
        # dist normalization
        distances_score_deep[distances_score_deep>img_w] = img_w
        distances_score_deep = 1-distances_score_deep/img_w

        return theta_score_deep,start_xys_score_deep,distances_score_deep