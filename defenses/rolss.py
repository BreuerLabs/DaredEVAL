from RoLSS.models.torchvision.models import resnet, densenet # use the version of resnet provided by the RoLSS authors that is altered to include the skip connection

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
                    skip_defended_feature_extractor = resnet.resnet18(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet34':
                    weights = resnet.ResNet34_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet34(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet50':
                    weights = resnet.ResNet50_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet50(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet101':
                    weights = resnet.ResNet101_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet101(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet152':    
                    weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet152(weights=weights, skip=config.defense.skip)
                
                skip_defended_feature_extractor.fc = self.model.fc
                skip_defended_feature_extractor.

            #! densenet has a bug that is not yet fixed for RoLSS defense
            # elif 'densenet' in arch:
            #     if arch == 'densenet121':
            #         weights = densenet.DenseNet121_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = densenet.densenet121(weights=weights, skip=config.defense.skip)
            #     elif arch == 'densenet161':
            #         weights = densenet.DenseNet161_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = densenet.densenet161(weights=weights, skip=config.defense.skip)
            #     elif arch == 'densenet169':
            #         weights = densenet.DenseNet169_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = densenet.densenet169(weights=weights, skip=config.defense.skip)
            #     elif arch == 'densenet201':
            #         weights = densenet.DenseNet201_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = densenet.densenet201(weights=weights, skip=config.defense.skip)
                
            #     skip_defended_feature_extractor.classifier = self.model.classifier

            elif 'resnext' in arch:
                if arch == 'resnext50':
                    weights = resnet.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnext50_32x4d(weights=weights, skip=config.defense.skip)
                elif arch == 'resnext101':
                    weights = resnet.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnext101_32x8d(weights=weights, skip=config.defense.skip)
            
            else:
                raise RuntimeError(
                    f'Model with the name {arch} not currently supported for RoLSS defense'
                )
            
            self.model = skip_defended_feature_extractor # replace our previously loaded model

    RoLSS_defended_model = RoLSS(config)
    return RoLSS_defended_model