import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        img_dim = config.dataset.input_size[0] * config.dataset.input_size[1] * config.dataset.input_size[2]
        
        # Embedding layer for labels
        self.label_emb = nn.Embedding(config.dataset.n_classes, config.model.label_dim)
        
        # Unpack the hyperparameter dict
        hyper = config.model.discriminator
        
        # Model architecture with parameterized layer sizes and dropout rate
        layers = [
                nn.Linear(img_dim + config.model.label_dim, hyper.hidden_dim[0]), 
                nn.LeakyReLU(hyper.negative_slope, inplace=True), 
                nn.Dropout(hyper.dropout)
                ]

        # Add hidden layers based on the hidden_dim list
        for i in range(1, len(hyper.hidden_dim)):
            layers.append(nn.Linear(hyper.hidden_dim[i - 1], hyper.hidden_dim[i]))
            layers.append(nn.LeakyReLU(hyper.negative_slope, inplace=True))
            layers.append(nn.Dropout(hyper.dropout))

        # Output layer
        layers.append(nn.Linear(hyper.hidden_dim[-1], 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, labels):
        x = x.view(x.size(0), -1)  # Flatten image to (batch_size, img_dim)
        c = self.label_emb(labels)  # Embed labels
        x = torch.cat([x, c], 1)  # Concatenate image and label embedding
        out = self.model(x)
        return out.squeeze()
    
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        hyper = config.model.generator
        self.output_shape = config.dataset.input_size

        self.label_emb = nn.Embedding(config.dataset.n_classes, config.model.label_dim)
        img_dim = config.dataset.input_size[0] * config.dataset.input_size[1] * config.dataset.input_size[2]
        
        # Model architecture with parameterized layer sizes
        layers = [nn.Linear(hyper.latent_dim + config.model.label_dim, hyper.hidden_dim[0]), nn.LeakyReLU(hyper.negative_slope, inplace=True)]

        # Add hidden layers based on the hidden_dim list
        for i in range(1, len(hyper.hidden_dim)):
            layers.append(nn.Linear(hyper.hidden_dim[i - 1], hyper.hidden_dim[i]))
            layers.append(nn.LeakyReLU(hyper.negative_slope, inplace=True))

        # Output layer to img_dim with Tanh activation
        layers.append(nn.Linear(hyper.hidden_dim[-1], img_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
    
    def forward(self, z, labels):
        z = z.view(z.size(0), -1)  # Ensure latent vector is (batch_size, latent_dim)
        c = self.label_emb(labels)  # Embed labels
        x = torch.cat([z, c], 1)  # Concatenate latent vector and label embedding
        out = self.model(x)
        # Reshape to the specified output dimensions (batch_size, *output_shape)
        return out.view(x.size(0), *self.output_shape)
    
    
class GAN:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.latent_dim = config.model.generator.latent_dim
        self.n_classes = config.dataset.n_classes
        self.batch_size = config.model.hyper.batch_size

        # Initialize generator and discriminator
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.model.generator.lr)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.model.discriminator.lr)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def generator_train_step(self):
        self.g_optimizer.zero_grad()
        z = Variable(torch.randn(self.batch_size, self.latent_dim)).to(self.device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, self.n_classes, self.batch_size))).to(self.device)
        
        # Generate fake images
        fake_images = self.generator(z, fake_labels)
        
        # Calculate generator loss
        validity = self.discriminator(fake_images, fake_labels)
        g_loss = self.criterion(validity, Variable(torch.ones(self.batch_size)).to(self.device))
        
        # Backward and optimize
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()

    def discriminator_train_step(self, real_images, labels):
        self.d_optimizer.zero_grad()

        # Real images
        real_validity = self.discriminator(real_images, labels)
        real_loss = self.criterion(real_validity, Variable(torch.ones(real_images.size(0))).to(self.device))
        
        # Fake images
        z = Variable(torch.randn(real_images.size(0), self.latent_dim)).to(self.device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, self.n_classes, real_images.size(0)))).to(self.device)
        fake_images = self.generator(z, fake_labels)
        fake_validity = self.discriminator(fake_images, fake_labels)
        fake_loss = self.criterion(fake_validity, Variable(torch.zeros(real_images.size(0))).to(self.device))
        
        # Total loss and optimization
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()

    def train(self, data_loader):
        all_d_loss = []
        for epoch in range(self.config.model.hyper.epochs):
            print(f'Starting epoch {epoch+1}...')
            
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                real_images = Variable(images).to(self.config.training.device)
                labels = Variable(labels).to(self.device)

                # Train discriminator for n_critic steps
                d_loss = 0
                for _ in range(self.config.model.hyper.n_critic):
                    d_loss = self.discriminator_train_step(real_images, labels) #! += instead of just + ? 

                # Train generator
                g_loss = self.generator_train_step()

                # Display step for logging and generating sample images
                if (epoch * len(data_loader) + i + 1) % self.config.training.display_step == 0:
                    print(f'Epoch [{epoch+1}/{self.config.model.hyper.epochs}], Step [{i+1}/{len(data_loader)}], d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
                    
                    # Generate sample images for visual feedback
                    self.generate_sample_images()

                all_d_loss.append(d_loss)
        
        # Print average discriminator loss
        norm_d_loss = self.average(all_d_loss)
        print(f'Average Discriminator Loss: {norm_d_loss:.4f}')
        print('GAN Training is Done!')

    def generate_sample_images(self):
        self.generator.eval()
        with torch.no_grad():
            z = Variable(torch.randn(9, self.latent_dim)).to(self.device)
            sample_labels = Variable(torch.LongTensor(np.arange(9))).to(self.device)
            sample_images = self.generator(z, sample_labels).unsqueeze(1)
            # You can further use make_grid or similar here to visualize the sample_images
            
        self.generator.train()

    @staticmethod
    def average(lst):
        return sum(lst) / len(lst) if lst else 0