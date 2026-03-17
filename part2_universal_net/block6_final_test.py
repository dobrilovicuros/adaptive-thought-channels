"""
BLOCK 6: Final Generalization & Pruning Test
======================================
The ultimate test of the Universal Adaptive Neural Network.
Goal: Achieve 100% accuracy on all 16 Boolean functions (including 6 unseen),
while aggressively pruning useless neurons.

Key Mechanisms:
  - Extended burn-in (5000 epochs) to ensure perfect generalization before pruning.
  - Importance score + L1 regularization to kill useless neurons.
  - Channel death and Revival mechanisms to prevent catastrophic forgetting.
  - Starts with 64 neurons -> dynamically reduces to essential core.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, List, Dict

# Truth tables definition
ALL_FUNCTIONS = {
    "ZERO": [0,0,0,0], "AND":[0,0,0,1], "A_NOT_B": [0,0,1,0], "A":[0,0,1,1],
    "NOT_A_B":[0,1,0,0], "B": [0,1,0,1], "XOR":[0,1,1,0], "OR": [0,1,1,1],
    "NOR":[1,0,0,0], "XNOR": [1,0,0,1], "NOT_B":[1,0,1,0], "A_OR_NB":[1,0,1,1],
    "NOT_A": [1,1,0,0], "NA_OR_B":[1,1,0,1], "NAND":[1,1,1,0], "ONE":[1,1,1,1],
}

INPUTS = [[0, 0], [0, 1],[1, 0], [1, 1]]
INPUT_TO_IDX = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

def make_sample(x: List[int], func_name: str) -> Tuple[np.ndarray, float]:
    truth_table = ALL_FUNCTIONS[func_name]
    y = truth_table[INPUT_TO_IDX[(x[0], x[1])]]
    return np.array(x + truth_table, dtype=float), float(y)

def make_dataset(func_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [],[]
    for name in func_names:
        for x in INPUTS:
            xi, yi = make_sample(x, name)
            X.append(xi)
            Y.append(yi)
    return np.array(X), np.array(Y).reshape(-1, 1)

# ─────────────────────────────────────────────
# Adaptive Pruning Channel
# ─────────────────────────────────────────────

class AdaptiveChannel:
    def __init__(self, input_size=6, hidden_size=16, l1_lambda=0.002,
                 death_threshold=0.008, death_patience=500,
                 importance_window=300, min_alive=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1_lambda = l1_lambda
        self.death_threshold = death_threshold
        self.death_patience = death_patience
        self.importance_window = importance_window
        self.min_alive = min_alive

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))

        # Pruning states
        self.alive = np.ones(hidden_size, dtype=bool)
        self.imp_history = [[] for _ in range(hidden_size)]
        self.low_imp_count = np.zeros(hidden_size, dtype=int)
        
        self.death_events = []
        self.alive_history =[]

    def _he_init(self):
        """Reinitialize dead channel (Revival)."""
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.abs(np.random.randn(1, self.hidden_size)) * 0.5 # Positive bias for ReLU
        self.W2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2 / self.hidden_size)
        self.b2 = np.zeros((1, 1))
        self.alive[:] = True
        self.imp_history = [[] for _ in range(self.hidden_size)]
        self.low_imp_count[:] = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.a1[:, ~self.alive] = 0.0 # Dead neurons output 0
        self.out = self.a1 @ self.W2 + self.b2
        return self.out

    def backward(self, d_out: np.ndarray, lr: float, l2: float):
        sign_W2 = np.sign(self.W2)
        dW2 = self.a1.T @ d_out + l2 * self.W2 + self.l1_lambda * sign_W2
        db2 = np.sum(d_out, axis=0, keepdims=True)
        
        d_a1 = d_out @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0)
        d_z1[:, ~self.alive] = 0.0 # No gradient for dead neurons
        
        dW1 = self.X.T @ d_z1 + l2 * self.W1
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def update_importance(self, epoch: int) -> bool:
        killed = False
        alive_count = np.sum(self.alive)
        for i in range(self.hidden_size):
            if not self.alive[i]: continue
            
            imp = float(np.abs(self.W2[i, 0])) * float(np.mean(np.abs(self.a1[:, i])))
            self.imp_history[i].append(imp)
            if len(self.imp_history[i]) > self.importance_window:
                self.imp_history[i].pop(0)
                
            avg_imp = np.mean(self.imp_history[i])
            if avg_imp < self.death_threshold:
                self.low_imp_count[i] += 1
            else:
                self.low_imp_count[i] = 0
                
            if self.low_imp_count[i] >= self.death_patience and alive_count > self.min_alive:
                self.alive[i] = False
                self.W1[:, i] = 0.0
                self.W2[i, 0] = 0.0
                self.death_events.append((epoch, i))
                alive_count -= 1
                killed = True
                
        self.alive_history.append(int(np.sum(self.alive)))
        return killed

    @property
    def n_alive(self):
        return int(np.sum(self.alive))

# Compositor remains the same basic logic as Block 2 (skipped class definition repetition for brevity, assuming standard MLP combining channel outputs).
class Compositor:
    def __init__(self, n_channels=4, hidden_size=8):
        self.n_channels = n_channels
        self.W1 = np.random.randn(n_channels, hidden_size) * np.sqrt(2/n_channels)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, 1))

    def forward(self, c_outs):
        self.concat = np.concatenate(c_outs, axis=1)
        self.z1 = self.concat @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = 1 / (1 + np.exp(-np.clip(self.z2, -500, 500)))
        return self.out

    def loss(self, y_pred, y_true):
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self, y_true, lr, l2):
        n = len(y_true)
        d_out = (self.out - y_true) / n
        dW2 = self.a1.T @ d_out + l2 * self.W2
        db2 = np.sum(d_out, axis=0, keepdims=True)
        d_a1 = d_out @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0)
        dW1 = self.concat.T @ d_z1 + l2 * self.W1
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        d_concat = d_z1 @ self.W1.T
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return[d_concat[:, i:i+1] for i in range(self.n_channels)]

class AdaptiveChannelNetwork:
    def __init__(self, n_channels=4, channel_hidden=16, comp_hidden=8,
                 lr=0.03, l2=0.001, l1_lambda=0.002, death_threshold=0.008, 
                 death_patience=500, importance_window=300, min_neurons_alive=3,
                 burn_in=5000, kill_interval=400, channel_death_patience=2000,
                 loss_window=300, revival_factor=1.25, revival_cooldown=1000):
        
        self.n_channels = n_channels
        self.lr, self.l2 = lr, l2
        self.burn_in = burn_in
        self.kill_interval = kill_interval
        self.channel_death_patience = channel_death_patience
        self.loss_window = loss_window
        self.revival_factor = revival_factor
        self.revival_cooldown = revival_cooldown

        self.channels =[AdaptiveChannel(6, channel_hidden, l1_lambda, death_threshold, 
                         death_patience, importance_window, min_neurons_alive) for _ in range(n_channels)]
        self.compositor = Compositor(n_channels, comp_hidden)

        self.channel_alive = np.ones(n_channels, dtype=bool)
        self.channel_dead_count = np.zeros(n_channels, dtype=int)

        self.loss_history =[]
        self.loss_avg = None
        self.last_kill_epoch = -kill_interval
        self.last_revival_epoch = -revival_cooldown
        self.revival_events =[]
        self.total_epochs = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        c_outs =[]
        for i, k in enumerate(self.channels):
            out = k.forward(X)
            if not self.channel_alive[i]: out = np.zeros_like(out)
            c_outs.append(out)
        return self.compositor.forward(c_outs)

    def backward(self, y_true: np.ndarray):
        c_grads = self.compositor.backward(y_true, self.lr, self.l2)
        for i, (channel, grad) in enumerate(zip(self.channels, c_grads)):
            if self.channel_alive[i]:
                channel.backward(grad, self.lr, self.l2)

    def step(self, epoch_loss: float, epoch: int):
        self.total_epochs = epoch
        self.loss_history.append(epoch_loss)
        if len(self.loss_history) > self.loss_window: self.loss_history.pop(0)
        self.loss_avg = float(np.mean(self.loss_history))

        # Pruning Logic
        if epoch >= self.burn_in and (epoch - self.last_kill_epoch) >= self.kill_interval:
            killed_any = any(k.update_importance(epoch) for i, k in enumerate(self.channels) if self.channel_alive[i])
            if killed_any: self.last_kill_epoch = epoch
        else:
            for k in self.channels: k.alive_history.append(k.n_alive)

        # Channel Death Logic
        for i, k in enumerate(self.channels):
            if not self.channel_alive[i]: continue
            if k.n_alive == 0:
                self.channel_dead_count[i] += 1
                if self.channel_dead_count[i] >= self.channel_death_patience:
                    self.channel_alive[i] = False
                    print(f"  [Epoch {epoch:5d}] ✦ CHANNEL K{i} killed.")
            else:
                self.channel_dead_count[i] = 0

        # Revival Logic
        if (epoch >= self.burn_in and self.loss_avg and 
            epoch_loss > self.loss_avg * self.revival_factor and 
            (epoch - self.last_revival_epoch) >= self.revival_cooldown):
            
            dead =[i for i in range(self.n_channels) if not self.channel_alive[i]]
            if dead:
                target = dead[0]
                self.channels[target]._he_init()
                self.channel_alive[target] = True
                self.last_revival_epoch = epoch
                self.revival_events.append((epoch, target))
                print(f"  [Epoch {epoch:5d}] ✦ REVIVAL K{target} triggered!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(int).flatten()

    def evaluate_function(self, func_name: str) -> Tuple[int, int]:
        correct = 0
        for x in INPUTS:
            xi, yi = make_sample(x, func_name)
            if self.predict(xi.reshape(1, -1))[0] == int(yi): correct += 1
        return correct, 4

def run_experiment():
    np.random.seed(42)
    funcs = list(ALL_FUNCTIONS.keys())
    train_funcs, test_funcs = funcs[:10], funcs[10:]

    print("=" * 65)
    print("BLOCK 6: Final Generalization & Pruning Test")
    print("=" * 65)

    X_train, Y_train = make_dataset(train_funcs)
    net = AdaptiveChannelNetwork()

    print("--- Training ---")
    n = len(X_train)
    batch_size = 16
    for epoch in range(30000):
        idx = np.random.permutation(n)
        X_shuf, Y_shuf = X_train[idx], Y_train[idx]
        epoch_loss = 0.0
        
        for i in range(0, n, batch_size):
            Xb, Yb = X_shuf[i:i+batch_size], Y_shuf[i:i+batch_size]
            pred = net.forward(Xb)
            epoch_loss += net.compositor.loss(pred, Yb) * len(Xb)
            net.backward(Yb)
            
        avg_loss = epoch_loss / n
        net.step(avg_loss, epoch)
        
        if (epoch + 1) % 5000 == 0:
            alive_n = sum(k.n_alive for k in net.channels)
            print(f"  Epoch {epoch+1:5d} | Loss: {avg_loss:.4f} | Alive Neurons: {alive_n}/64")

    print("\n--- Final Statistics ---")
    train_correct = sum(net.evaluate_function(f)[0] for f in train_funcs)
    test_correct = sum(net.evaluate_function(f)[0] for f in test_funcs)
    
    alive_n = sum(k.n_alive for k in net.channels)
    print(f"  Training Accuracy: {train_correct/40 * 100:.1f}%")
    print(f"  Test Accuracy:     {test_correct/24 * 100:.1f}%")
    print(f"  Neurons Alive:     {alive_n}/64 ({64-alive_n} successfully pruned)")
    
    if train_correct == 40 and test_correct == 24 and alive_n < 64:
        print("\n  ✓ SUCCESS: 100% accuracy achieved with massive resource optimization!")

if __name__ == "__main__":
    run_experiment()
