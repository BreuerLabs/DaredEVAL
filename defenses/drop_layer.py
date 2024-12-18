import torch
import wandb
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

from classifiers.abstract_classifier import AbstractClassifier
from classifiers.defense_utils import ElementwiseLinear
from utils.plotting import plot_tensor

def apply_drop_layer_defense(config, model:AbstractClassifier):
    
    class DropLayerClassifier(model.__class__):
        
        def __init__(self, config):
            super(DropLayerClassifier, self).__init__(config)
            self.input_defense_layer = self.init_input_defense_layer()
            if self.config.defense.penalty == "skip_lasso":
                self.skip_defense_layer = self.init_skip_defense_layer()            

        def init_input_defense_layer(self):
            if self.config.model.flatten:
                in_features = (self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0],)
            else:
                in_features = (self.config.dataset.input_size[0], self.config.dataset.input_size[1], self.config.dataset.input_size[2])

            defense_layer = ElementwiseLinear(in_features, w_init=self.config.defense.input_defense_init)

            return defense_layer

        def init_skip_defense_layer(self):
            pass # not yet implemented
            # skip = nn.Linear(in_features, self.config.dataset.n_classes, bias=False)

        def post_batch(self):
            super(DropLayerClassifier, self).post_batch()
            if self.config.defense.name == "drop_layer" and self.config.defense.apply_threshold:
               self.apply_threshold()

            track_features = self.config.training.wandb.track_features
            if track_features:
                if isinstance(track_features, ListConfig):
                    feature_norms = self.get_feature_norms()
                    for idx, feature_norm in zip(track_features, feature_norms):
                        wandb.log({f"feature_{idx}" : feature_norm.item(), "train_step": self.train_step})
                wandb.log({"n_features" : self.n_features_remaining, "train_step" : self.train_step})

        def post_epoch(self, epoch):
            super(DropLayerClassifier, self).post_epoch(epoch)
            if epoch % self.config.training.save_defense_layer_freq == 0:
                # save defense layer mask plot
                if self.config.defense.plot_mask:
                    w_first = self.input_defense_layer.weight.data
                    
                    n_channels, x_dim, y_dim = self.config.dataset.input_size
                    if n_channels == 3:
                        w_norms = torch.linalg.norm(w_first, dim=0)
                    else: # n_channels == 1
                        w_norms = w_first.abs() # does the same thing as norm of dim=0 when n_channels is 1, but this is more readable
                        
                    w_norms = w_norms.reshape((1, x_dim, y_dim)) # (x_dim, y_dim) -> (1, x_dim, y_dim)

                    plt = plot_tensor(w_norms.cpu(), self.save_as)
                    if self.config.training.wandb.track:
                        wandb.log({"defense_mask" : plt, "train_step": self.train_step, "epoch": epoch+1})

        def forward(self, x):
            x = self.input_defense_layer(x) # pass through the drop layer first
            x = super(DropLayerClassifier, self).forward(x)
            return x

        def get_loss(self, output, target):
            loss = super(DropLayerClassifier, self).get_loss(output, target)
            if self.config.defense.penalty == "lasso": # add lasso penalty term to loss
                lasso_pen = self.config.defense.lasso.lambda_ * self.lasso_penalty()
                loss = loss + lasso_pen
            
            return loss

        def lasso_penalty(self): # Lasso penalty on one-dimensional weights
        
            w_first = self.input_defense_layer.weight
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels. Performs abs() when n_channels=1, combines RGB values of pixels (group lasso) when n_channels=3
            
            if self.config.defense.lasso.smooth:
                alpha = self.config.defense.lasso.alpha
                smoothed_norms = (w_norms**2 / (2*alpha)) + (alpha/2) # see eqn. (15) in SGLasso paper
                final_smoothed_norms = torch.zeros_like(w_norms)
                final_smoothed_norms[w_norms < alpha] = smoothed_norms[w_norms < alpha]
                final_smoothed_norms[w_norms >= alpha] = w_norms[w_norms >= alpha] # only use smoothed norms if less than alpha
                lasso_pen = final_smoothed_norms.sum()
            else:
                lasso_pen = w_norms.sum()

            return lasso_pen

        def apply_threshold(self):
            thresh = self.config.defense.lasso.threshold
            current_w_first = self.input_defense_layer.weight.data # (n_channels, x_dim, y_dim)
            current_w_norms = torch.linalg.norm(current_w_first, dim=0) # (x_dim, y_dim)
            below_threshold = current_w_norms <= thresh  # (x_dim, y_dim)
            new_w_first = current_w_first * ~below_threshold # (n_channels, x_dim, y_dim) x (x_dim, y_dim) = (n_channels, x_dim, y_dim)
            self.input_defense_layer.weight.data = new_w_first
            self.n_features_remaining = below_threshold.numel() - below_threshold.sum()

        def get_feature_norms(self):
            w_first = self.input_defense_layer.weight.data
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels, shape is (x_dim, y_dim)
            feature_idxs = OmegaConf.to_object(self.config.training.wandb.track_features)
            
            if isinstance(feature_idxs[0], list): # like drop_layer, one weight for each feature index, norm is just abs. value
                w_features = [w_norms[*feature_idx] for feature_idx in feature_idxs]
                return [torch.abs(w_feature) for w_feature in w_features]
            
            elif isinstance(feature_idxs[0], int):
                if len(w_first.shape) == 1: # just one weight for each feature index, norm is just abs. value
                    w_features = [w_norms[feature_idx] for feature_idx in feature_idxs]
                    return [torch.abs(w_feature) for w_feature in w_features]
            else:
                assert 0 == 1, f"feature_idxs[0] is type {type(feature_idxs[0])}, not int or list"
                
                # elif len(w_first.shape) == 2: # like SGLNN, take L2 norm of all weights associated with each feature index
                #     w_features = [w_first[:, feature_idx] for feature_idx in feature_idxs]
                #     return [torch.linalg.norm(w_feature) for w_feature in w_features]

    drop_layer_defended_model = DropLayerClassifier(config)

    return drop_layer_defended_model