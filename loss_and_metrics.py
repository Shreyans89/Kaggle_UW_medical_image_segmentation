
from functools import partial

def dice_coeff(pred, targ,smooth=1):
    "Compute dice coeff per class then average b/w predicted and target masks"
    classes=range(pred.shape[1])
    ## get 1_hot pred masks
    pred_mask=pred.argmax(dim=1)
    pred_1hot=torch.stack([pred_mask==c for c in classes],dim=1)
    ## get 1_hot target for each class
    target_1hot=torch.stack([targ==c for c in classes],dim=1)
    ## find intersection (and union) for each class,by summing slong Batch,H,W dims
    intersect_each_class=(pred_1hot*target_1hot).sum(axis=(0,2,3))+smooth
    set_sum_each_class=target_1hot.sum(axis=(0,2,3))+pred_1hot.sum(axis=(0,2,3))+smooth
    ## accumalate class wise intersection/set_sum by taking average across classes
    intersect_by_sum=(intersect_each_class/set_sum_each_class).mean()
    
    return 2*intersect_by_sum

## implementing a smoothen version of the dice coeff, tranforming it to loss =1-dice_coef (to minimize)
def dice_loss(pred, targ,smooth=1):
    "Compute dice coeff b/w predicted and target masks"
    ## softmax instead of argmax to make it differentiable
    classes=range(pred.shape[1])
    pred_mask_soft=F.softmax(pred,dim=1)
    target_1hot=torch.stack([targ==c for c in classes],dim=1)
    ## find intersection (and union) for each class,by summing slong Batch,H,W dims
    intersect_each_class=(pred_mask_soft*target_1hot).sum(axis=(0,2,3))+smooth
    set_sum_each_class=target_1hot.sum(axis=(0,2,3))+pred_mask_soft.sum(axis=(0,2,3))+smooth
    ## accumalate class wise intersection/set_sum by taking average across classes
    intersect_by_sum=(intersect_each_class/set_sum_each_class).mean()
    
    return 1-2*intersect_by_sum

## what pctage of actual pixels of each class are detected
# class_accuracy_metrics={class_name:}

def pixel_accuracy(pred, targ):
    "Compute dice coeff b/w predicted and target masks"
    pred_mask=pred.argmax(dim=1)
    return (pred_mask==targ).float().mean()

def pixel_accuracy_cls(pred, targ,class_num):
    "Compute dice coeff b/w predicted and target masks"
    pred_mask=pred.argmax(dim=1)
    class_mask=(targ==class_num)
    detected_pxl_count=((pred_mask==class_num)*class_mask).sum()
    return detected_pxl_count/class_mask.sum()

#def focal_loss(pred,targ,gamma=0):
 #   """calculate the focal loss between pred and targ """
  #  classes=range(pred.shape[1])  
  #  pred_probs=F.softmax(pred,dim=1)#bs x num_classes x H x W
  #  target_1hot=torch.stack([targ==c for c in classes],dim=1)
  #  pred_likelyhood=(target_1hot*pred_probs).sum(dim=1)           
  #  pdb.set_trace()
  #  loss=-(1-pred_likelyhood)**gamma*torch.log(pred_likelyhood) ## bs X H X W
  #  return loss.mean()


def focal_loss(pred,targ,weights={'background':1.,'small_bowel':4.,'large_bowel':8.,'stomach':16.},
               codes={'background': 0, 'small_bowel': 3, 'large_bowel': 2, 'stomach': 1},gamma=2):
    """calculate the focal loss between pred and targ """
    weights=torch.Tensor([weights[class_name] for class_name in sorted(codes,key=codes.__getitem__)]).to('cuda')
    n_log_likelyhood=CrossEntropyLossFlat(axis=1,reduction='none',weight=weights)(pred,targ)      
    loss=n_log_likelyhood*(1-torch.exp(-n_log_likelyhood))**gamma ## bs X H X W
    return loss.mean()

def segmentation_cross_entropy_loss(img_out,mask):
    return F.cross_entropy(img_out.flatten(start_dim=2),mask.flatten(start_dim=1))
    
