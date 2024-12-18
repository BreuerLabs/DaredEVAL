from classifiers.abstract_classifier import AbstractClassifier

def apply_struppek_defense(config, model:AbstractClassifier):
    
    class Struppek(model.__class__):
        
        def __init__(self, config):
            super(Struppek, self).__init__(config)
            self.criterion.label_smoothing = config.defense.alpha
            self.criterionSum.label_smoothing = config.defense.alpha 

    struppek_defended_model = Struppek(config)

    return struppek_defended_model