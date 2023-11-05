import torch
import torch.nn as nn
from matchnet.model import MatchNet

# ADD EXPLANATION 

# Fine-tune CLRNet with matchnet
class MatchNetAssign(nn.Module):
    def __init__(self, cfg):
        super(MatchNetAssign, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w # img width
        self.conf_treshold = self.cfg.conf_treshold # 0.6 Denotes the minimum confidence score required in matchnet to be a positive match
        self.max_matches = self.cfg.max_matches  # Indicates the maximum number of matched pairs per ground truth lane.        
        self.model = MatchNet().cuda()

    def norm(self, theta_score_deep, start_xys_score_deep, distances_score_deep):
        theta_score_deep[theta_score_deep>180]=180
        theta_score_deep = 1-theta_score_deep/180

        # xys normalization
        start_xys_score_deep[start_xys_score_deep>self.img_w] = self.img_w
        start_xys_score_deep = 1-start_xys_score_deep/self.img_w

        # dist normalization
        distances_score_deep[distances_score_deep>self.img_w] = self.img_w
        distances_score_deep = 1-distances_score_deep/self.img_w
        return theta_score_deep,start_xys_score_deep,distances_score_deep

    def forward(self, theta_score_deep, start_xys_score_deep, distances_score_deep, pos_score_deep, neg_score_deep, iou_deep, targets):
        deep_matched_row_inds = torch.ones(1).cuda()
        deep_matched_col_inds = torch.ones(1).cuda()
        targets_index = targets[:, 1].clone()

        # theta normalization
        theta_score_deep, start_xys_score_deep, distances_score_deep = self.norm(theta_score_deep, start_xys_score_deep, distances_score_deep)

        # Iterate through each ground truth and selecting the corresponding matched prediction.
        for i in range(targets.shape[0]):
            tar_idx = targets_index.argmax()
            targets_index[tar_idx] = 0
                
            pair_features = torch.cat((theta_score_deep[:,None,tar_idx],start_xys_score_deep[:,None,tar_idx],
                                    distances_score_deep[:,None,tar_idx],pos_score_deep[:,None,tar_idx], 
                                    neg_score_deep[:,None,tar_idx],iou_deep[:,None,tar_idx]),dim=1).cuda()   
            matchnet_assign = self.model(pair_features)
                
            matchnet_assign[matchnet_assign<self.conf_treshold]=0 
            deep_values, deep_indices = torch.topk(matchnet_assign[:,0], self.max_matches)
            deep_matched_row_inds_i = deep_indices[deep_values>0.0]
            deep_matched_col_inds_i = (torch.zeros_like(deep_matched_row_inds_i)+tar_idx).cuda()
            deep_matched_row_inds = torch.cat((deep_matched_row_inds, deep_matched_row_inds_i),0)
            deep_matched_col_inds = torch.cat((deep_matched_col_inds, deep_matched_col_inds_i),0)
            
        return deep_matched_row_inds[1:].long(), deep_matched_col_inds[1:].long()


