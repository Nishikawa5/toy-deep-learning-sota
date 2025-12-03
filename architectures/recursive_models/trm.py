import torch
import torch.nn as nn


"""
This model reuses its weights, so it forces the model to
not cheat by memorizing patterns. It learns a general
operation for all passes.

"regularization through compression"


"""

class ToyTinyRecursiveModel(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            d_model,
            think_steps,
            answer_steps,
        ):

        super(ToyTinyRecursiveModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.think_steps = think_steps
        self.answer_steps = answer_steps
        
        # Project the problem and answer to latent space
        self.x_encoder = nn.Linear(input_dim, d_model)
        self.y_encoder = nn.Linear(output_dim, d_model)

        # Recursive Thinking
        # [current_z, encoded_input, encoded_current_answer] -> new_z
        self.reasoning = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.answer_head = nn.Linear(d_model, output_dim)

        
    def forward(self, x, target=None):
        batch_size = x.shape[0]

        # "Problem statement"
        x_emb = F.relu(self.x_encoder(x))

        # the initial state of Latent Thought
        z = torch.zeros(batch_size, self.d_model, device=x.device)
        
        # the initial answer
        y_logits = torch.zeros(batch_size, self.output_dim, device=x.device)

        all_step_losses = []
        # The model "thinks" about the problem
        # First it guesses an answer, then think about it
        # Then it updates its answer and think about it again
        # This is repeated for a answer_steps
        for _ in range(self.answer_steps):
            
            # Prepare the answer to the model's brain
            # for thinking
            y_probs = F.softmax(y_logits, dim=-1)
            y_emb = self.y_encoder(y_probs)

            # The model "thinks" about the problem
            for _ in range(self.think_steps):
                # Latent thought, problem, answer
                combined_input = torch.cat([z, x_emb, y_emb], dim=-1)

                # Based on the latent thought, problem and answer
                # the model updates its latent thought (thinks)
                delta_z = self.reasoning(combined_input)
                z = z + delta_z
                z = F.layer_norm(z, (self.d_model,))
            
            # Update the current answer
            y_logits = self.answer_head(z)

            if target is not None:
                # Calculate loss inside forward
                # Since the model is refining the answer at each step
                # It doesn't magically get the final answer correct at last step
                # This creates a new branch in the backprop graph
                # it gets connected at the end when summing them
                step_loss = F.cross_entropy(y_logits, target)
                all_step_losses.append(step_loss)

        final_loss = sum(all_step_losses) / len(all_step_losses) if all_step_losses else None

        return y_logits, final_loss

class ToyTinyRecursiveModelWarmup(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            d_model,
            think_steps,
            warmup_steps,
            train_steps,
        ):

        super(ToyTinyRecursiveModelWarmup, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.think_steps = think_steps
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps

        # Project the problem and answer to latent space
        self.x_encoder = nn.Linear(input_dim, d_model)
        self.y_encoder = nn.Linear(output_dim, d_model)

        # Recursive Thinking
        # [current_z, encoded_input, encoded_current_answer] -> new_z
        self.reasoning = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.answer_head = nn.Linear(d_model, output_dim)

    def _run_cycle(self, z, x_emb, y_logits):
        # Prepare the answer to the model's brain
        # for thinking
        y_probs = F.softmax(y_logits, dim=-1)
        y_emb = self.y_encoder(y_probs)

        # The model "thinks" about the problem
        for _ in range(self.think_steps):
            # Latent thought, problem, answer
            combined_input = torch.cat([z, x_emb, y_emb], dim=-1)

            # Based on the latent thought, problem and answer
            # the model updates its latent thought (thinks)
            delta_z = self.reasoning(combined_input)
            z = z + delta_z
            z = F.layer_norm(z, (self.d_model,))
        
        # Update the current answer
        y_logits = self.answer_head(z)
        return z, y_logits
        
    def forward(self, x, target=None):
        batch_size = x.shape[0]

        # "Problem statement"
        x_emb = F.relu(self.x_encoder(x))

        # the initial state of Latent Thought
        z = torch.zeros(batch_size, self.d_model, device=x.device)
        
        # the initial answer
        y_logits = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Warmup
        # Here the model doesn't calculate gradients
        # So it doesn't store the gradients for backprop in memory
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                z, y_logits = self._run_cycle(z, x_emb, y_logits)

        # The thought of the model after warmup
        z = z.detach()
        y_logits = y_logits.detach()

        # Now we need to calculate gradients
        # for the train steps
        z.requires_grad_(True)
        y_logits.requires_grad_(True)

        all_step_losses = []
        # The model "thinks" about the problem
        # First it guesses an answer, then think about it
        # Then it updates its answer and think about it again
        # This is repeated for a answer_steps
        for _ in range(self.train_steps):
            z, y_logits = self._run_cycle(z, x_emb, y_logits)
            
            if target is not None:
                # Calculate loss inside forward
                # Since the model is refining the answer at each step
                # It doesn't magically get the final answer correct at last step
                # This creates a new branch in the backprop graph
                # it gets connected at the end when summing them
                step_loss = F.cross_entropy(y_logits, target)
                all_step_losses.append(step_loss)

        final_loss = sum(all_step_losses) / len(all_step_losses) if all_step_losses else None

        return y_logits, final_loss