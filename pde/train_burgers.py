import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fourier_initial_condition(x_size, k, L):
    A = 1.0
    modes = []
    for i in range(1, k + 1):
        dist_min = -A / (i * i)
        dist_max = A / (i * i)
        a_k = np.random.uniform(dist_min, dist_max)
        b_k = np.random.uniform(dist_min, dist_max)
        modes.append((i, a_k, b_k))

    spacing = L / x_size
    domain = np.zeros(x_size)
    for i in range(x_size):
        x = i * spacing
        value = 0
        for (k_mode, a_k, b_k) in modes:
            inside = 2 * np.pi * k_mode * x / L
            value += a_k * np.sin(inside) + b_k * np.cos(inside)
        domain[i] = value
    return domain

def burgers_forward_euler_step(u, delta_x, delta_t, nu):
    N = len(u)
    u_next = np.zeros(N)
    for i in range(N):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        du_dx = (u[ip1] - u[im1]) / (2 * delta_x)
        d2u_dx2 = (u[ip1] - 2 * u[i] + u[im1]) / (delta_x * delta_x)
        du_dt = -u[i] * du_dx + nu * d2u_dx2
        u_next[i] = u[i] + delta_t * du_dt
    return u_next

def generate_dataset(num_samples, x_size, L, delta_t, nu, timesteps):
    k = 3
    delta_x = L / x_size
    data = []

    for _ in range(num_samples):
        u = fourier_initial_condition(x_size, k, L)
        for _ in range(timesteps):
            u_prev = u.copy()
            u_next = burgers_forward_euler_step(u, delta_x, delta_t, nu)
            u = u_next
            data.append((u_prev, u_next))

    return data

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, mlp_ratio=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        # x: (B, seq_len, 1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

def train():
    # params
    x_size = 64
    L = 6 * np.pi
    delta_t = 0.05
    nu = 0.6
    num_samples = 100
    timesteps = 400
    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    # generate data
    print("Generating dataset...")
    data = generate_dataset(num_samples - 1, x_size, L, delta_t, nu, timesteps)
    test_data = generate_dataset(1, x_size, L, delta_t, nu, timesteps)

    # to tensors
    inputs = torch.from_numpy(np.array([d[0] for d in data])).float()
    targets = torch.from_numpy(np.array([d[1] for d in data])).float()

    test_inputs = torch.from_numpy(np.array([d[0] for d in test_data])).float()
    test_targets = torch.from_numpy(np.array([d[1] for d in test_data])).float()

    # residual targets
    residuals = targets - inputs

    dataset = torch.utils.data.TensorDataset(inputs, residuals)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model - try MLP first
    # model = ResidualMLP(x_size, hidden_dim=256, num_layers=4).to(device)

    # or transformer (treating each point as a token)
    model = SimpleTransformer(input_dim=1, embed_dim=32, num_heads=4, num_layers=2).to(device)
    use_transformer = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (batch_in, batch_res) in enumerate(loader):
            batch_in = batch_in.to(device)
            batch_res = batch_res.to(device)

            if use_transformer:
                batch_in = batch_in.unsqueeze(-1)  # (B, seq, 1)
                batch_res = batch_res.unsqueeze(-1)

            optimizer.zero_grad()
            pred_res = model(batch_in)
            loss = criterion(pred_res, batch_res)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Iter {i} | Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs} done, Avg Loss: {avg_loss:.6f}")

    # validation - autoregressive
    print("\nValidation (autoregressive)...")
    model.eval()
    preds = []
    targs = []

    with torch.no_grad():
        current_input = test_inputs[0:1].to(device)
        for i in range(len(test_data)):
            if use_transformer:
                inp = current_input.unsqueeze(-1)
                pred_res = model(inp).squeeze(-1)
            else:
                pred_res = model(current_input)

            pred = current_input + pred_res
            target = test_targets[i:i+1].to(device)

            loss = criterion(pred, target)
            print(f"Step {i}, Loss: {loss.item():.6f}")

            preds.append(pred.cpu().numpy().flatten())
            targs.append(target.cpu().numpy().flatten())

            current_input = pred

    # save animation
    print("\nSaving animation...")
    fig, ax = plt.subplots(figsize=(10, 4))

    def update(frame):
        ax.clear()
        ax.plot(targs[frame], 'b-', label='Target')
        ax.plot(preds[frame], 'r-', label='Prediction')
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f'Burgers Equation - Step {frame}')
        ax.legend()
        return ax,

    anim = FuncAnimation(fig, update, frames=min(100, len(preds)), interval=50)
    anim.save('burgers_pytorch.gif', writer='pillow', fps=20)
    print("Saved burgers_pytorch.gif")
    plt.close()

if __name__ == "__main__":
    train()
