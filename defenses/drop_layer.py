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
            self.input_defense_layer = self.init_input_defense_layer()

            # if using adaptive group lasso, load pre-adapted defense layer weights
            try:
                self.adaptive = config.defense.lasso.adaptive
            except Exception as e:
                print("Warning: config.defense.adaptive not in struct. Default setting to False.")
                self.adaptive = False
            
            if self.adaptive:
                pre_adapted_defense_layer = self.get_pre_adapted_defense_layer() # note this is a Tensor, as opposed to self.input_defense_layer which is an ElementwiseLinear module
                assert pre_adapted_defense_layer.shape == self.input_defense_layer.weight.data.shape, "pre-adapted defense layer and current defense layer are different shapes"
                self.pre_adapted_defense_layer_norms = torch.linalg.norm(pre_adapted_defense_layer, dim=0).to(self.device) # we only need the norms

                # if a pre-adapted pixel norm is 0, make the same pixel norm 0 in our current model
                self.input_defense_layer.weight.data[pre_adapted_defense_layer == 0] = 0

            if self.config.defense.apply_threshold:
                try:
                    self.initial_threshold = self.config.defense.lasso.initial_threshold
                    self.change_threshold_epoch = self.config.defense.lasso.change_threshold_epoch
                except Exception as e:
                    print("Warning: initial_threshold not found in struct, default setting to 1e-6")
                    print("Warning: change_threshold_epoch not found in struct, default setting to 10")
            
                self.threshold = self.initial_threshold
            
                
        def init_input_defense_layer(self):
            if self.config.model.flatten:
                in_features = (self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0],)
            else:
                in_features = (self.config.dataset.input_size[0], self.config.dataset.input_size[1], self.config.dataset.input_size[2])

            defense_layer = ElementwiseLinear(in_features, w_init=self.config.defense.input_defense_init)

            return defense_layer

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

            if self.config.defense.apply_threshold:
                if epoch == self.change_threshold_epoch: # change from 'initial_threshold' to 'threshold' when epoch == change_threshold_epoch
                    self.threshold = self.config.defense.threshold

        def forward(self, x):
            x = self.input_defense_layer(x) # pass through the drop layer first
            x = super(DropLayerClassifier, self).forward(x)
            return x

        def get_loss(self, output, target):
            loss = super(DropLayerClassifier, self).get_loss(output, target)

            if self.config.defense.penalty: # add penalty term to loss
                lasso_pen, ridge_pen = self.get_penalties()
                loss = loss + (self.config.defense.lasso.lambda_ * lasso_pen) + (self.config.defense.lasso.ridge_lambda * ridge_pen)
            
            return loss

        def get_penalties(self): # Penalty on one-dimensional weights
        
            w_first = self.input_defense_layer.weight
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels. Performs abs() when n_channels=1, combines RGB values of pixels (group lasso) when n_channels=3

            if self.adaptive: # Calculate GL+AGL (Dinh and Ho, 2020)
                eps = torch.tensor(1e-8) # to avoid division by zero
                adaptive_gamma = torch.tensor(self.config.defense.lasso.adaptive_gamma)
                adaptive_w_norms = torch.div(w_norms, torch.pow(self.pre_adapted_defense_layer_norms, adaptive_gamma)+eps)

            lasso_penalty = adaptive_w_norms.sum() if self.adaptive else w_norms.sum()
            ridge_penalty = torch.pow(torch.linalg.norm(w_norms), 2)

            return lasso_penalty, ridge_penalty

        def apply_threshold(self):
            
            current_w_first = self.input_defense_layer.weight.data # (n_channels, x_dim, y_dim)
            current_w_norms = torch.linalg.norm(current_w_first, dim=0) # (x_dim, y_dim)
            below_threshold = current_w_norms <= self.threshold  # (x_dim, y_dim)
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
                
        def pre_train(self):
            if self.config.defense.load_only_defense_layer:
                self.model = self.init_model() # replace the loaded model with an unloaded model
                for param in self.input_defense_layer.parameters(): # freeze the defense layer during training
                    param.requires_grad = False

        def get_pre_adapted_defense_layer(self):
            # loads the whole pre-adapted model, and keeps only the input defense layer weights.
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

            pre_adapted_model.input_defense_layer = self.init_input_defense_layer() # placeholder defense layer needed to load in the pre-adapted defense layer

            # Load model weights
            if torch.cuda.device_count() > 1:
                pre_adapted_model.model = nn.DataParallel(pre_adapted_model.model) # hacky workaround, fix this eventually
            pre_adapted_model.load_model(pre_adapted_weights_path)

            return pre_adapted_model.input_defense_layer.weight.data


    drop_layer_defended_model = DropLayerClassifier(config)

    return drop_layer_defended_model