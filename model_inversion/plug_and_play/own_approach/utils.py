import torch
from copy import deepcopy
import omegaconf
from model_inversion.plug_and_play.attacks import final_selection, initial_selection, optimize

def create_target_vector(attack_config):
        device = attack_config.training.device

        targets = None
        target_classes = attack_config.dataset.target_classes
        n_candidates = attack_config.dataset.n_candidates
        
        if attack_config.dataset.target_classes == 'all':
            targets = torch.tensor([i for i in range(target_classes)])
            targets = torch.repeat_interleave(targets, n_candidates)
        
        elif type(target_classes) == int:
            targets = torch.full(size=(n_candidates, ),
                                fill_value=target_classes)
        
        elif type(target_classes) == omegaconf.listconfig.ListConfig: # The type when loaded from torch is not a normal list
            targets = torch.tensor(target_classes)
            targets = torch.repeat_interleave(targets, n_candidates)
        
        
        else:
            raise Exception(
                f' Please specify a target class or state a target vector.')

        targets = targets.to(device)
        return targets

#! TODO: Modify
def create_candidates(generator, target_model, targets, attack_config):
        candidate_config = attack_config.model.candidates
        device = attack_config.training.device
        
        #? Maybe not needed for our project
        # if 'candidate_file' in candidate_config:
        #     candidate_file = candidate_config['candidate_file']
        #     w = torch.load(candidate_file)
        #     w = w[:self._config['num_candidates']]
        #     w = w.to(device)
        #     print(f'Loaded {w.shape[0]} candidates from {candidate_file}.')
        #     return w

        if 'candidate_search' in candidate_config:
            search_config = candidate_config.candidate_search
            w = initial_selection.find_initial_w(generator=generator,
                               target_model=target_model,
                               targets=targets,
                               seed=attack_config.training.seed,
                               **search_config)
            
            print(f'Created {w.shape[0]} candidates randomly in w space.')
        else:
            raise Exception(f'No valid candidate initialization stated.')

        w = w.to(device)
        return w

#! TODO: Modify
def create_initial_vectors(G, target_model, targets, attack_config):
    with torch.no_grad():
        w = create_candidates(G, target_model, targets, attack_config).cpu() #? move to device instead of cpu?
        
        # if config.attack['single_w']: #! What is this?
        #     w = w[:, 0].unsqueeze(1)
        
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V

