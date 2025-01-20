from classifiers.abstract_classifier import AbstractClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_label_smoothing_defense(config, model:AbstractClassifier):
    
    class LabelSmoothing(model.__class__):
        
        def __init__(self, config):
            super(LabelSmoothing, self).__init__(config)
            if config.model.criterion == "crossentropy":
                self.criterion = CrossEntropyLoss(label_smoothing=config.defense.alpha)
                self.criterionSum = CrossEntropyLoss(label_smoothing=config.defense.alpha, reduction='sum')

    label_smoothing_defended_model = LabelSmoothing(config)

    return label_smoothing_defended_model

class CrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing=0.0, reduction = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        
        if self.reduction == "mean":
            result = torch.mean(loss) / num_batch
        
        elif self.reduction == "sum":
            result = torch.sum(loss) / num_batch
        
        else:
            raise ValueError("No valid reduction")
        
        return result
