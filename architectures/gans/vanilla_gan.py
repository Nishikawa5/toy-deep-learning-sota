import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VanillaGANConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    in_features = 28*28
    latent_dim = 128

    lr = 2e-4
    epochs = 20
    beta1 = 0.5
    beta2 = 0.999
    
    # Gradient clipping for stability
    grad_clip = 1.0
    
    # Label smoothing
    real_label_smooth = 1.0

class Discriminator(nn.Module):

    def __init__(self, config: VanillaGANConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        # Given some image, predict if its real or fake
        x = x.view(x.size(0), -1)
        return self.model(x)



class Generator(nn.Module):

    def __init__(self, config: VanillaGANConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, config.in_features),
            nn.Tanh(),
        )

    def forward(self, z):
        # Given some random noise, generate an image
        x = self.model(z)
        x = x.view(x.size(0), 1, 28, 28)

        return x



class VanillaGAN():

    def __init__(self, config: VanillaGANConfig):
        self.config = config
        
        self.discriminator = Discriminator(config).to(config.device)
        self.generator = Generator(config).to(config.device)

        self.discriminator_optim = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.lr, 
            betas=(config.beta1, config.beta2)
        )
        self.generator_optim = optim.Adam(
            self.generator.parameters(), 
            lr=config.lr, 
            betas=(config.beta1, config.beta2)
        )
        
        # Numerically stable loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_step(self, real_data, fake_data):
        self.discriminator_optim.zero_grad()
        batch_size = real_data.size(0)

        # Forward pass - outputs are logits
        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data.detach())

        # Labels with smoothing for real (0.9 instead of 1.0)
        # The real data has real labels (1)
        real_labels = torch.ones(batch_size, 1, device=self.config.device) * self.config.real_label_smooth
        # The fake data has fake labels (0)
        fake_labels = torch.zeros(batch_size, 1, device=self.config.device)

        # Calculate loss using BCEWithLogitsLoss
        real_loss = self.criterion(real_output, real_labels)
        fake_loss = self.criterion(fake_output, fake_labels)
        loss = real_loss + fake_loss

        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip)
        
        self.discriminator_optim.step()

        return loss.item()

    def generate_noise(self, batch_size):
        return (torch.rand(batch_size, self.config.latent_dim, device=self.config.device) * 2) - 1
            
    def generator_step(self, z):
        self.generator_optim.zero_grad()
        batch_size = z.size(0)

        # Forward pass
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)

        # Generator wants discriminator to think fakes are real
        real_labels = torch.ones(batch_size, 1, device=self.config.device)

        # Calculate loss
        loss = self.criterion(fake_output, real_labels)

        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.grad_clip)
        
        self.generator_optim.step()

        return loss.item()

    def train_step(self, x):
        # Normalize data to [-1, 1] to match generator output range
        x = x * 2 - 1
        
        # Uniform noise [-1, 1]
        batch_size = x.size(0)
        discriminator_noise = self.generate_noise(batch_size)
        generator_noise = self.generate_noise(batch_size)

        discriminator_loss = self.discriminator_step(
            x, 
            self.generator(discriminator_noise)
        )
        generator_loss = self.generator_step(
            generator_noise
        )

        return discriminator_loss, generator_loss

    def inference(self, num_samples=16):
        self.generator.eval()

        with torch.no_grad():
            # Uniform noise [-1, 1]
            z = self.generate_noise(num_samples)
            
            generated_data = self.generator(z)
            generated_data = (generated_data + 1) / 2
            generated_data = torch.clamp(generated_data, 0, 1)

        self.generator.train()
        
        return generated_data.cpu()