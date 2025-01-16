from classifiers.abstract_classifier import AbstractClassifier

def apply_TLDMI_defense(config, model:AbstractClassifier):
    
    class TLDMI(model.__class__): # adapted from TL-DMI repo
        
        def __init__(self, config):
            super(TLDMI, self).__init__(config)

            # num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if config.defense.freeze_layers == 0:
                for param in self.model.parameters():
                    param.requires_grad = False

            elif config.defense.freeze_layers > 0:
                for i, child in enumerate(self.feature_extractor.children()):
                    if i <= config.defense.freeze_layers:
                        for j, param in enumerate(child.parameters()):
                            param.requires_grad = False
                            num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                            if num_trainable < 45000000:
                                pass
                        

            num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"training {num_trainable} out of {total_params} parameters")


        # def forward(self, x): # adapted from DefendMI/BiDO/model.py
        #     # embed_img() and z_to_logits() must be defined in the non-defended classifier in order to apply TLDMI
        #     z = self.embed_img(x)
        #     logits = self.z_to_logits(z)

        #     return logits

    tldmi_defended_model = TLDMI(config)

    return tldmi_defended_model