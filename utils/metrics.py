import torch


def threshold(x, threshold=0.5):
    return (x > threshold).type(x.dtype)

'''
    TP (True Positive):  true: 1, predict: 1
    TN (True Negative):  true: 0, predict: 0
    FP (False Positive): true: 0, predict: 1
    FN (False Negative): true: 1, predict: 0
'''
def iou(predict, target, eps=1e-7):
    '''
        iou = |A cross B| / (|A| + |B| - |A cross B|)
    '''
    n = target.size(0)
    predict = predict.view(n, -1)
    target = target.view(n, -1)
    intersection = (predict * target).sum(dim=1)

    score = (intersection + eps) / (predict.sum(dim=1) + target.sum(dim=1) - intersection + eps)
    score = score.sum() / n
    return score

def dice_coefficient(predict, target, eps=1e-7):
    '''
        dice_coefficient = 2 * |A cross B| / (|A| + |B|)
    '''
    n = target.size(0)
    predict = predict.view(n, -1)
    target = target.view(n, -1)
    intersection = (predict * target).sum(dim=1)
    
    score = (2. * intersection + eps) / (predict.sum(dim=1) + target.sum(dim=1) + eps)
    score = score.sum() / n
    return score

def accuracy(predict, target):
    '''
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    '''
    n = target.size(0)
    predict = threshold(predict, threshold=0.5)
    predict = predict.view(n, -1)
    target = target.view(n, -1)
    
    tp_add_tn = predict.eq(target).sum(dim=1)
    
    score = tp_add_tn.float() / target.size(1)
    score = score.sum() / n
    return score

def precision(predict, target, eps=1e-7):
    '''
        precision = tp / (tp + fp)
    '''
    n = target.size(0)
    predict = threshold(predict, threshold=0.5)
    predict = predict.view(n, -1)
    target = target.view(n, -1)
    
    tp = torch.sum(predict * target, dim=1)
    fp = torch.sum(predict, dim=1) - tp
    
    score = (tp + eps) / (tp + fp + eps)
    score = score.sum() / n
    return score

def recall(predict, target, eps=1e-7):
    '''
        recall = tp / (tp + fn)
    '''
    n = target.size(0)
    predict = threshold(predict, threshold=0.5)
    predict = predict.view(n, -1)
    target = target.view(n, -1)
    
    tp = torch.sum(predict * target, dim=1)
    fn = torch.sum(target, dim=1) - tp
    
    score = (tp + eps) / (tp + fn + eps)
    score = score.sum() / n
    return score
