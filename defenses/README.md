# Adding your own defenses
We show how to add your own custom defense through an example. Let's say we have a defense that adds a custom penalty to the loss function during training and inference. Let's name this defense `PenDMI`.

First, in the `/defenses/` folder, make a new file called `pendmi.py`, and start with the following code:
```
def apply_PenDMI_defense(config, model:AbstractClassifier):
    
    class PenDMI(model.__class__):
        
        def __init__(self, config):
            super(PenDMI, self).__init__(config)



    PenDMI_defended_model = PenDMI(config)

    return PenDMI_defended_model
```
We now have a `PenDMI` class that inherits from the class of whichever `AbstractClassifier` model we seek to defend. To add our defense's functionality, we overwrite the `get_loss` function in `AbstractClassifier`:
```
def apply_PenDMI_defense(config, model:AbstractClassifier):
    
    class PenDMI(model.__class__):
        
        def __init__(self, config):
            super(PenDMI, self).__init__(config)

        def get_loss(self, output, target):
            loss = self.criterionSum(output, target) + self.config.defense.alpha * <PENALTY>
            return loss

    PenDMI_defended_model = PenDMI(config)

    return PenDMI_defended_model
```
Notice we have added a defense hyperparameter `alpha`. To make this work, we add a `pendmi.yaml` file in `/configuration/classifier/defense/` with the following body:
```
name: pendmi
alpha: 1
```
The value we set for `alpha` here will be the default value, used when `defense.alpha` is not specified on the command line.
Lastly, we need to add our defense to `/defenses/get_defense.py` by inserting the lines marked by `###` :
```
from defenses.label_smoothing import apply_label_smoothing_defense
from defenses.bido import apply_bido_defense
from defenses.mid import apply_MID_defense
from defenses.tldmi import apply_TLDMI_defense
from defenses.gaussian_noise import apply_gaussian_noise_defense
from defenses.rolss import apply_RoLSS_defense
from defenses.sparse import apply_sparse_defense
from defenses.pendmi import apply_PenDMI_defense ### ADD THIS LINE HERE

def get_defense(config, model):
    
    defense_name = config.defense.name
    
    if defense_name == "label_smoothing":
        model = apply_label_smoothing_defense(config, model)
        
    elif defense_name == "MID":
        model = apply_MID_defense(config, model)

    elif defense_name == "bido":
        model = apply_bido_defense(config, model)

    elif defense_name == "tldmi":
        model = apply_TLDMI_defense(config, model)

    elif defense_name == "gaussian_noise":
        model = apply_gaussian_noise_defense(config, model)

    elif defense_name == "rolss":
        model = apply_RoLSS_defense(config, model)

    elif defense_name == "sparse":
        model = apply_sparse_defense(config, model)

    elif defense_name == "pendmi": ### ADD THESE 2 LINES HERE
        model = apply_PenDMI_defense(config, model) ###

    elif defense_name == "no_defense":
        pass # model stays as is
        
    else:
        raise ValueError(f"Unknown defense: {defense_name}")
    
    return model
```
And that's it! Now, you may run your defense from the command line with
```
python train_classifier.py defense=pendmi defense.alpha=0.5
```
In this way, your defense is automatically ready to use with all target models available in the codebase, and comparison to other defenses is seamless, as both can be attacked within the same framework.

More generally, your `.py` file in the `defenses` folder should overwrite any methods present in `AbstractClassifier` that pertain to your defense. See `/classifiers/abstract_classifier.py` for more information on these methods in order to overwrite them properly.
