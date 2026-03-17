# Development Log — Adaptive Neural Network with Neuron Pruning

**Project Goal:** Develop a neural network where neurons dynamically die over time if they do not contribute to information processing, and revive when the network requires more capacity. Long-term goal: a network that can learn multiple logical functions simultaneously through specialized "thought channels".

---

## Version 1 — First Implementation (Sigmoid + Activation as Metric)

### Idea
Track the average activation of each neuron. If a neuron maintains a low activation for a long time, prune (kill) it.

### Implementation
- Activation function: **Sigmoid** in the hidden layer
- Pruning metric: Average absolute neuron activation
- Parameters: `death_threshold=0.02`, `death_patience=300`

### Problem
Not a single neuron was pruned.

### Diagnostics
We ran an analysis of actual activations after training:
### Conclusion and Root Cause
The Sigmoid activation **never outputs exactly 0** — it is always in the range (0, 1). Because of this, all neurons appeared "active" even when they contributed nothing useful. The network evenly distributed the workload across all neurons because it was mathematically the path of least resistance for gradient descent.

Adjusting the threshold to ~0.34 would have pruned neurons, but that was a superficial fix — the problem was the metric itself, not the parameters.

*(Code Fix: `TypeError` in `evaluate()` method. `float(yi)` failed because `yi` was an array of shape `(1,)`. Fixed by using `float(yi.flat[0])`)*

---

## Version 2 — Redesign (ReLU + Importance Score + L1)

### What we changed and why:

**1. ReLU instead of Sigmoid in the hidden layer**  

A neuron might be highly active, but if its outgoing `W2` weight is tiny, it doesn't contribute to the output. The Importance Score combines both factors.

**3. L1 Regularization on W2**  
Useful neurons "defend" themselves via strong error gradients. Useless ones offer no resistance, so L1 gradually zeroes out their weights.

**4. Burn-in Phase**  
For the first 3000 epochs, pruning is disabled. The network learns the function first, and only then do we start pruning redundant neurons.

**5. Kill Interval**  
Minimum of 400 epochs between pruning events — giving the network time to structurally recover.

### Result
The network successfully learned XOR and pruned excess neurons, leaving only the essential ones.

---

## Version 3 — Revival Mechanism (In Development)

### Motivation
The next step is training the same network on multiple logical functions (XOR, AND, OR) sequentially to form specialized "thought channels".
The prerequisite problem to solve: **catastrophic forgetting**. When we train a new function, the gradients alter the very weights that learned the previous one.

The Revival Mechanism is part of the solution: when the network receives a new task and the error spikes, **new neurons spawn** to handle the new task, leaving the old ones intact.

### Conceptual Design
- Track the **moving average of the loss** over the last N epochs.
- When the current loss **spikes significantly** above the average → trigger a capacity demand signal.
- Find a **dead neuron** candidate for revival.
- Reinitialize it **randomly** (He initialization) — do not restore old weights, as that would inject obsolete information.
- Assign a `birth_epoch` marker to prevent immediate pruning (a mini burn-in per neuron).

### Bug Fix: ReLU Bias Initialization
**Problem:** Newborn neurons were receiving `bias = 0.0`. ReLU activation with zero weights and zero bias always outputs 0, meaning the importance score stays at 0 from epoch 1. The network instantly treated the newborn neuron as dead again.
**Fix:** `self.b1[0, idx] = float(np.abs(np.random.randn()) * 0.5)`
`np.abs` guarantees a positive bias, ensuring ReLU passes a signal even when `W1` weights are weak.

---

## Version 4 — Autonomous Task Change Detection

### Motivation
In v3, the network didn't autonomously detect task changes. The overall error didn't spike dramatically enough for the revival to trigger, so the network "quietly" overwrote XOR neurons with AND logic (catastrophic forgetting).
**Key Observation:** A human shouldn't have to tell the brain "change the task" — the brain senses it.

### Idea: Error Vector Cosine Similarity
Instead of total scalar error, we track the **error vector per sample**:
Cosine similarity measures the *angle* between the current vector and the historical average, ignoring magnitude. We care about the RELATIVE PATTERN of errors. XOR and AND might have the same total error, but completely different patterns.

### Iterative Calibration of the Adaptive Threshold (v4.1 - v4.2c)
Using a fixed threshold (e.g., 0.83) required manual tuning for every pair of tasks. We switched to a **statistical adaptive threshold**:
**Tuning the parameters:**
- `sigma=2.5` → 39 false detections (Too sensitive).
- `sigma=4.0` → 9 false detections.
- `sigma=5.5` + `sim_history` limit (last 500) + `cooldown=5000` → **2 perfect detections.**

*Insight:* Sigma controls sensitivity, cooldown prevents chain-triggering, and limiting history prevents early noisy epochs from ruining the statistics.

---

## Version 5 — EWC (Elastic Weight Consolidation)

### Concept
To prevent catastrophic forgetting, EWC computes the **Fisher Information** for every weight when a task change is detected. During the new task, a penalty is added to the gradient: `penalty = ewc_lambda × F_i × (W_current - W_anchor)`.

### Results & Failures
- **Problem 1:** `std_dev = 0` caused the detection threshold to equal 1.0, triggering false alarms. *Fixed by clamping std_dev minimum to 0.005.*
- **Problem 2:** Fisher was calculated when gradients were already zero (network had converged). *Fixed by anchoring at the end of the burn_in phase.*
- **Problem 3:** Lambda=500 blocked learning entirely. *Lowered to 100.*
- **Core Limitation Revealed:** Only 1 neuron remained alive at the end of the AND phase. **One scalar output neuron mathematically cannot represent XOR and AND simultaneously.**

### Architectural Conclusion (Reconsolidation)
The entire EWC saga led to a fundamental realization: **retention is not a regularization problem — it is an architectural problem.**
A single output path must overwrite itself. We needed a multi-output architecture where channels can spontaneously form in the hidden layer. EWC was trying to solve a mathematically impossible problem.

---

## Version 6 — Two-Level Output: Gate + Head

### Solution: Gate & Head Matrices
- `W_gate` (hidden, n_channels): Learns WHICH channel is active (Softmax competition).
- `W_head` (hidden, n_channels): Learns WHAT each channel outputs (Sigmoid).

Channel `k` receives a gradient ONLY if it wins the Softmax gate. Neurons contributing to different channels receive conflicting gradients and gradually specialize into specific channels.

### Results (SUCCESS ✓)
- XOR learned: 4/4
- AND learned: 4/4
- XOR retained: 4/4
Spontaneous specialization was confirmed: XOR mapped to K0, AND to K1, without explicit labels telling the hidden layer what to do.

---

## Version 7 & 8 — Routing Agent & Final Thoughts

### Contextual Gate Failure (v7)
We tried feeding a sequence of previous `(input, answer)` pairs directly into the gate for few-shot task recognition. It failed to converge due to EWC conflicts and sequence misalignment.

### Routing Agent (v8)
Since autonomous routing based solely on confidence fails for ambiguous inputs (e.g., `[1,1]` is 0 for XOR but 1 for AND), we built an external Q-learning Agent.
- **State:** Short-term memory of 4 pairs + channel accuracy stats.
- **Action:** Select channel K0-K3.
- **Reward:** +1 correct, -1 incorrect.
The Agent successfully learned the meta-rule linking input patterns to the correct channel without altering the base network's weights.

### Final Project Conclusion
1. **Pruning works:** We successfully reduced network size dynamically.
2. **Autonomous task detection works:** Using error vector cosine similarity.
3. **Thought channels form spontaneously:** The Gate+Head architecture forces neurons to group.
4. **Fundamental insight:** True autonomous routing for ambiguous logical tasks requires contextual memory, which led us directly to the concept of the Universal Von Neumann Architecture in Part 2.
Every epoch, L1 pushes all `W2` weights slightly closer to zero:ReLU outputs exactly 0 when the input is negative. Neurons that don't contribute literally pass a zero forward into the network — making them visibly "dead" without needing a special metric.

**2. Importance Score instead of Activation**
