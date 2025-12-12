import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class VanillaVAEConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    in_features = 28*28
    latent_dim = 20

    lr = 2e-4
    epochs = 20
    beta = 1.0

    optim_beta1 = 0.5
    optim_beta2 = 0.999
    
    # Gradient clipping for stability
    grad_clip = 1.0
    
    # Label smoothing
    real_label_smooth = 0.9


class MLPEncoder(nn.Module):
    def __init__(self, config: VanillaVAEConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )

        self.mu = nn.Linear(64, config.latent_dim)
        self.logvar = nn.Linear(64, config.latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class MLPDecoder(nn.Module):
    def __init__(self, config: VanillaVAEConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, config.in_features),
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(x.size(0), 1, 28, 28)

        return x


class VanillaVAE():
    def __init__(self, config: VanillaVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = MLPEncoder(config).to(config.device)
        self.decoder = MLPDecoder(config).to(config.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=config.lr, 
            betas=(config.optim_beta1, config.optim_beta2)
        )
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def _parametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = self.criterion(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config.beta * kl_loss

    def train_step(self, x):
        self.optimizer.zero_grad()

        batch_size = x.size(0)

        # Forward pass
        mu, logvar = self.encoder(x)
        z = self._parametrization_trick(mu, logvar)
        x_recon = self.decoder(z)

        # Calculate loss
        loss = self.loss_function(x, x_recon, mu, logvar)

        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.grad_clip)
        
        self.optimizer.step()

        return loss.item()

    def inference(self, num_samples=16):
        self.decoder.eval()

        with torch.no_grad():
            # N(0, 1) latent variable
            z = torch.randn(num_samples, self.config.latent_dim, device=self.config.device)
            
            logits = self.decoder(z)
            
            # Sigmoid to get image pixels
            # the output from decoder are logits (because of loss func)
            generated_data = torch.sigmoid(logits)

        self.decoder.train()
        
        return generated_data.cpu()