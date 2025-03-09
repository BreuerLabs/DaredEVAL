from RoLSS.models.torchvision.models import resnet # use the version of resnet provided by the RoLSS authors that is altered to include the skip connection

from classifiers.abstract_classifier import AbstractClassifier

def apply_RoLSS_defense(config, model:AbstractClassifier):
    
    class RoLSS(model.__class__):
        
        def __init__(self, config):
            super(RoLSS, self).__init__(config)

            # load the model again, this time with the skip parameter
            arch = config.model.architecture.lower()
            pretrained = config.model.pretrained

            if 'resnet' in arch:

                if arch == 'resnet18':
                    weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
                    skip_defended_model = resnet.resnet18(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet152':    
                    weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                    skip_defended_model = resnet.resnet152(weights=weights, skip=config.defense.skip)
                
                skip_defended_model.fc = self.model.fc

            else:
                raise RuntimeError(
                    f'Model with the name {arch} not currently supported for RoLSS defense'
                )
            
            self.model = skip_defended_model # replace our previously loaded model


    RoLSS_defended_model = RoLSS(config)

    return RoLSS_defended_model