import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):

    def __init__(self, in_features, t_dim):
        """
        We need to pass some information about the current time
        t of x. We could add some positional encoding,
        like sinusoidal used in Transformers, but lets keep
        it simple for now.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + t_dim, 64),
            
            nn.ReLU(),

            nn.Linear(64, in_features)
        )

    def get_t_info(self, x_shape, t):
        return torch.ones(x_shape[0], 1) * t

    def forward(self, x, t):

        # This network tries to predict
        # the noise of data?

        # If is a FLow Model no, it tries to predict the velocity
        # If is a Diffusion Model yes.
        t_prepared = self.get_t_info(x.shape, t)

        net_input = torch.cat([x, t_prepared], dim=1)

        return self.net(net_input)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.view(1, -1)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class RobustMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        """
        A more robust MLP using sinuisodal embedding for time encoding
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.mid_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.mid_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, in_features)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x_emb = self.input_layer(x)
        h = self.act(x_emb + t_emb)
        h = self.act(self.mid_layer1(h) + t_emb)
        h = self.act(self.mid_layer2(h) + t_emb)
        return self.final_layer(h)


class FlowODEModel():
    
    def __init__(self, in_features, device='cpu'):
        
        # Given some position x and time t
        # returns the velocity that makes
        # the position x_t closer to x_1 
        # (x_1 is the real data)

        # the model objective is to learn a vector field
        # so it can generalize and produce new data
        # given some random noise, the vector field
        # directs that to some real data
        self.u_t = RobustMLP(in_features).to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.u_t.parameters(), lr=1e-3)


        self.in_features = in_features
    
    def train_step(self, x_1):
        batch_size = x_1.shape[0]

        # sample from p_init
        x_0 = torch.randn_like(x_1)
        
        # uniform from 0 to 1
        t = torch.rand(batch_size, 1, device=x_1.device)

        # calculate x_t by interpolation
        x_t = (1 - t) * x_0 + t * x_1

        # target and prediction
        velocity_target = x_1 - x_0
        velocity_prediction = self.u_t(x_t, t)

        # the model objective is to predict
        loss = F.mse_loss(velocity_target, velocity_prediction)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples, steps):
        # sample from p_init
        # the only non deterministic part
        x = torch.randn(n_samples, self.in_features, device=self.device)

        # the size of steps
        dt = 1.0 / steps

        for i in range(steps):
            t_value = i * dt

            # predict the velocity
            velocity = self.u_t(x, torch.tensor(t_value, device=x.device))
            # step
            x = x + velocity * dt

        return x




class FlowSDEModel():

    def __init__(self, sigma_min, sigma_max, in_features, device='cpu'):
        
        # Given some position x and time t
        # returns the velocity that makes
        # the position x_t closer to x_1 
        # (x_1 is the real data)

        # the model objective is to learn a vector field
        # so it can generalize and produce new images
        # given some random noise, the vector field
        # directs that to some real data
        self.u_t = RobustMLP(in_features).to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.u_t.parameters(), lr=1e-3)

        # Now we add a probabilistic factor
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.in_features = in_features

    def get_sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def train_step(self, x_1):
        batch_size = x_1.shape[0]
        device = x_1.device

        # uniform from 0 to 1
        t = torch.rand(batch_size, 1, device=device)

        # calculate sigmas
        sigmas = self.get_sigma(t)

        # sample from p_init
        x_0 = torch.randn_like(x_1)
        x_t = x_1 + sigmas * x_0       

        # target and prediction
        score_prediction = self.u_t(x_t, t)
        score_target = -x_0 # / sigmas

        # the model objective is to predict the score
        loss = F.mse_loss(score_target, score_prediction)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    @torch.no_grad()
    def sample(self, n_samples, steps):
        device = next(self.u_t.parameters()).device

        # sample from p_init
        # the only non deterministic part
        # start from pure noise (sigma_max)
        x = torch.randn(n_samples, self.in_features, device=device) * self.sigma_max

        # go backwards from t=1 to t=0
        t_steps = torch.linspace(1, 0, steps, device=device)
        dt = 1.0 / steps

        for i in range(steps - 1):
            t_current = t_steps[i].view(1, 1).repeat(n_samples, 1)
            sigma_current = self.get_sigma(t_current)

            # get the variance scaling g(t)
            # for VE SDE: g(t) = sigma(t) * sqrt(2 * ln(sigma_max/sigma_min))
            g_t = sigma_current * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
            
            # get score prediction from model
            # our model predicts (score * sigma), so divide by sigma
            score_pred = self.u_t(x, t_current) / sigma_current

            # dx = [ -g(t)^2 * score ] dt + g(t) * dW
            drift = - (g_t ** 2) * score_pred * dt
            diffusion = g_t * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
            
            # subtract drift because we are going backwards in time
            x = x - drift + diffusion
        
        return x

if __name__ == "__main__":
    # Create simple 2D circle data
    theta = torch.rand(128) * 2 * math.pi
    r = 1.0
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    real_data = torch.stack([x, y], dim=1)

    flow_SDE = FlowSDEModel(0.5, 1, in_features=2)
    flow_ODE = FlowODEModel(in_features=2)

    print("Training...")
    for i in range(1000):
        loss_sde = flow_SDE.train_step(real_data)
        loss_ode = flow_ODE.train_step(real_data)
        if i % 100 == 0:
            print(f"Step {i}: SDE {loss_sde:.4f}, ODE {loss_ode:.4f}")

    out_sde = flow_SDE.sample(5, 100)
    out_ode = flow_ODE.sample(5, 100)
    print(f"Sample output:\nSDE {out_sde}\nODE {out_ode}")