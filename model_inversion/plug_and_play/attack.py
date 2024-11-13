
from model_inversion.plug_and_play.gan import GAN
from model_inversion.plug_and_play.stylegan import load_discrimator, load_generator
from model_inversion.plug_and_play.utils import create_initial_vectors, create_target_vector

def run(target_model, target_config, attack_config):
    device = attack_config.training.device
    # Load pre-trained StyleGan2 components
    G = load_generator(attack_config.model.stylegan_path, attack_config.training.device)
    D = load_discrimator(attack_config.model.stylegan_path, attack_config.training.device)
    num_ws = G.num_ws

    # Distribute models
    # target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    # target_model.name = target_model_name
    # synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    # synthesis.num_ws = num_ws
    # discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    
    # Create target vectors
    targets = create_target_vector(attack_config)

    
    # Create initial style vectors
    w, w_init, x, V = create_initial_vectors(G, target_model, targets, device)
    del G