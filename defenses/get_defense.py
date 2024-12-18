from defenses.label_smoothing import apply_label_smoothing_defense

def get_defense(config, model):
    
    
    if config.defense.name == "label_smoothing":
        model = apply_label_smoothing_defense(config, model)
        
    
    return model