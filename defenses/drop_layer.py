import torch
import torch.nn as nn
import wandb
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

from classifiers.abstract_classifier import AbstractClassifier
from classifiers.defense_utils import ElementwiseLinear
from classifiers.get_model import get_model
# from defenses.get_defense import get_defense
from utils import wandb_helpers, load_trained_models
from utils.plotting import plot_tensor

def apply_drop_layer_defense(config, model:AbstractClassifier):
    
    class DropLayerClassifier(model.__class__):
        
        def __init__(self, config):
            super(DropLayerClassifier, self).__init__(config)
            self.mask_layer = self.init_mask_layer()

            try:
                self.use_frozen_custom_mask = self.config.defense.use_frozen_custom_mask
            except Exception as e:
                print("Warning: config.defense.use_frozen_custom_mask not in struct. Default setting to False.")
                self.use_frozen_custom_mask = False
            
            # if using adaptive group lasso, load pre-adapted defense layer weights
            try:
                self.adaptive = config.defense.lasso.adaptive
            except Exception as e:
                print("Warning: config.defense.adaptive not in struct. Default setting to False.")
                self.adaptive = False
            
            if self.adaptive:
                pre_adapted_mask_layer = self.get_pre_adapted_mask_layer() # note this is a Tensor, as opposed to self.mask_layer which is an ElementwiseLinear module
                assert pre_adapted_mask_layer.shape == self.get_mask().data.shape, "pre-adapted mask layer and current mask layer are different shapes"
                self.pre_adapted_mask_layer_norms = torch.linalg.norm(pre_adapted_mask_layer, dim=0).to(self.device) # we only need the norms

                # # if a pre-adapted pixel norm is 0, make the same pixel norm 0 in our current model
                # mask = self.get_mask()
                # mask.data[pre_adapted_mask_layer == 0] = 0
                # self.set_mask(mask)

            # some configuration stuff to make thresholding work
            if self.config.defense.apply_threshold:
                try:
                    self.initial_threshold = self.config.defense.lasso.initial_threshold
                    self.change_threshold_epoch = self.config.defense.lasso.change_threshold_epoch
                except Exception as e:
                    print("Warning: initial_threshold or change_threshold_epoch not found in struct, default setting to 1e-6 and 10 respectively")
                    self.initial_threshold = 1e-6
                    self.change_threshold_epoch = 10
            
                self.threshold = self.initial_threshold
            
            # mask configuration stuff
                
        def init_mask_layer(self):
            if self.config.model.flatten:
                in_features = (self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0],)
            else:
                in_features = (self.config.dataset.input_size[0], self.config.dataset.input_size[1], self.config.dataset.input_size[2])

            defense_layer = ElementwiseLinear(in_features, w_init=self.config.defense.mask_init)

            return defense_layer

        def get_mask(self): # regardless of if DataParallel or not
            if isinstance(self.mask_layer, nn.DataParallel):
                return self.mask_layer.module.weight
            else:
                return self.mask_layer.weight

        def set_mask(self, new_mask: torch.Tensor): # regardless of if DataParallel or not
            if isinstance(self.mask_layer, nn.DataParallel):
                self.mask_layer.module.weight.data = new_mask
            else:
                self.mask_layer.weight.data = new_mask
                

        def post_batch(self):
            super(DropLayerClassifier, self).post_batch()
            if self.config.defense.apply_threshold:
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
                    w_first = self.get_mask().data
                    
                    n_channels, x_dim, y_dim = self.config.dataset.input_size
                    if n_channels == 3:
                        w_norms = torch.linalg.norm(w_first, dim=0)
                    else: # n_channels == 1
                        w_norms = w_first.abs() # does the same thing as norm of dim=0 when n_channels is 1, but this is more readable
                        
                    w_norms = w_norms.reshape((1, x_dim, y_dim)) # (x_dim, y_dim) -> (1, x_dim, y_dim)

                    plt = plot_tensor(w_norms.cpu(), self.save_as)
                    if self.config.training.wandb.track:
                        wandb.log({"defense_mask" : plt, "train_step": self.train_step, "epoch": epoch+1})

            if self.config.defense.apply_threshold:
                if epoch == self.change_threshold_epoch: # change from 'initial_threshold' to 'threshold' when epoch == change_threshold_epoch
                    self.threshold = self.config.defense.lasso.threshold

        def forward(self, x):
            x = self.mask_layer(x) # pass through the mask layer first
            x = super(DropLayerClassifier, self).forward(x)
            return x

        def get_loss(self, output, target):
            loss = super(DropLayerClassifier, self).get_loss(output, target)

            if self.config.defense.penalty: # add penalty term to loss
                lasso_pen, ridge_pen = self.get_penalties()
                loss = loss + (self.config.defense.lasso.lambda_ * lasso_pen) + (self.config.defense.lasso.ridge_lambda * ridge_pen)
            
            return loss

        def get_penalties(self): # Penalty on one-dimensional weights
        
            w_first = self.get_mask()
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels. Performs abs() when n_channels=1, combines RGB values of pixels (group lasso) when n_channels=3

            if self.adaptive: # Calculate GL+AGL (Dinh and Ho, 2020)
                eps = torch.tensor(1e-8) # to avoid division by zero
                adaptive_gamma = torch.tensor(self.config.defense.lasso.adaptive_gamma)
                adaptive_w_norms = torch.div(w_norms, torch.pow(self.pre_adapted_mask_layer_norms, adaptive_gamma)+eps)

            lasso_penalty = adaptive_w_norms.sum() if self.adaptive else w_norms.sum()
            ridge_penalty = torch.pow(torch.linalg.norm(w_norms), 2)

            return lasso_penalty, ridge_penalty

        def apply_threshold(self):
            
            current_w_first = self.get_mask().data # (n_channels, x_dim, y_dim)
            current_w_norms = torch.linalg.norm(current_w_first, dim=0) # (x_dim, y_dim)
            below_threshold = current_w_norms <= self.threshold  # (x_dim, y_dim)
            new_w_first = current_w_first * ~below_threshold # (n_channels, x_dim, y_dim) x (x_dim, y_dim) = (n_channels, x_dim, y_dim)
            self.set_mask(new_w_first)
            self.n_features_remaining = below_threshold.numel() - below_threshold.sum()

        def get_feature_norms(self):
            w_first = self.get_mask().data
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
                
        def train_model(self, train_loader, val_loader):
            freeze_mask = False
            if self.config.defense.load_only_mask:
                self.model = self.init_model() # replace the loaded model with an unloaded model
                freeze_mask = True
            
            elif self.use_frozen_custom_mask:
                custom_mask = torch.load(self.use_frozen_custom_mask) # load custom mask from path name given
                self.set_mask(custom_mask)
                freeze_mask = True
            
            if freeze_mask:
                for param in self.mask_layer.parameters(): # freeze the mask layer during training
                    param.requires_grad = False

            if torch.cuda.device_count() > 1:
                self.mask_layer = nn.DataParallel(self.mask_layer) # self.model will be put on DataParallel in super train_model call, so we just need to put the mask_layer on DataParallel here

            super(DropLayerClassifier, self).train_model(train_loader, val_loader)

        def get_pre_adapted_mask_layer(self):
            # loads the whole pre-adapted model, and keeps only the mask layer weights.
            pre_adapted_config, _ = wandb_helpers.get_config(
                entity=self.config.defense.lasso.pre_adapted_entity,
                project=self.config.defense.lasso.pre_adapted_project,
                run_id=self.config.defense.lasso.pre_adapted_run_id,
            )

            pre_adapted_weights_path = wandb_helpers.get_weights(
                entity=self.config.defense.lasso.pre_adapted_entity,
                project=self.config.defense.lasso.pre_adapted_project,
                run_id=self.config.defense.lasso.pre_adapted_run_id,
            )
            
            pre_adapted_model = get_model(pre_adapted_config)

            pre_adapted_model.mask_layer = self.init_mask_layer() # placeholder mask layer needed to load in the pre-adapted mask layer

            # Load model weights

            pre_adapted_model.load_model(pre_adapted_weights_path)

            return pre_adapted_model.mask_layer.weight.data

        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"
            if isinstance(self.model, nn.DataParallel): # self.model and self.mask_layer are on DataParallel
                state = {
                    "model": self.model.module.state_dict(),
                    "mask_layer": self.mask_layer.module.state_dict()
                }
            else:
                state= {
                    "model": self.model.state_dict(),
                    "mask_layer": self.mask_layer.state_dict()
                }
            torch.save(state, path)

        def load_model(self, file_path, map_location = None):
            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)  

            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state['model'])
                self.mask_layer.module.load_state_dict(state['mask_layer'])
            else:
                self.model.load_state_dict(state['model'])
                self.mask_layer.load_state_dict(state['mask_layer'])



    drop_layer_defended_model = DropLayerClassifier(config)

    return drop_layer_defended_model