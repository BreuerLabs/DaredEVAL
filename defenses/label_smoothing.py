from classifiers.abstract_classifier import AbstractClassifier

def apply_label_smoothing_defense(config, model:AbstractClassifier):
    
    class LabelSmoothing(model.__class__):
        
        def __init__(self, config):
            super(LabelSmoothing, self).__init__(config)
            self.criterion.label_smoothing = config.defense.alpha
            self.criterionSum.label_smoothing = config.defense.alpha 

    label_smoothing_defended_model = LabelSmoothing(config)

    return label_smoothing_defended_model