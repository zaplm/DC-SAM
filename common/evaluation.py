r""" Evaluate mask prediction """
import torch
from common.metrics import db_eval_iou, db_eval_boundary
import numpy as np
import warnings


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls, args):
        cls.ignore_index = 255
        cls.use_ignore = args.use_ignore

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None and cls.use_ignore:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union

class EvaluatorVideo:
    @staticmethod
    def db_statistics(per_frame_values):
        """ Compute mean,recall and decay from per-frame evaluation.
        Arguments:
            per_frame_values (ndarray): per-frame evaluation

        Returns:
            M,O,D (float,float,float):
                return evaluation statistics: mean,recall,decay.
        """

        # strip off nan values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            M = np.nanmean(per_frame_values)
            O = np.nanmean(per_frame_values > 0.5)

        N_bins = 4
        ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
        ids = ids.astype(np.uint8)

        D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

        return M, O, D
    
    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask').detach().cpu().numpy()
        pred_mask = pred_mask.detach().cpu().numpy()
        
        j_metrics_res, f_metrics_res = np.zeros(gt_mask.shape[0]), np.zeros(gt_mask.shape[0])
        for ii in range(gt_mask.shape[0]):
            j_metrics_res[ii] = db_eval_iou(gt_mask[ii, ...], pred_mask[ii, ...])
            f_metrics_res[ii] = db_eval_boundary(gt_mask[ii, ...], pred_mask[ii, ...])
        
        return j_metrics_res * 100, f_metrics_res * 100