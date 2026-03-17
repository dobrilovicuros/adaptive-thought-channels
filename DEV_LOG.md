# Development Log
**Adaptive Neural Network → Universal Network**  

**Project Goal:** Develop a neural network where neurons dynamically die over time if they do not contribute to information processing, and revive when the network requires more capacity. 
**Long-term Goal:** A network that can learn multiple logical functions simultaneously through specialized "thought channels"—ultimately leading to a universal interpreter that executes any logical operation based solely on its structural description.

---

## PART 1 — Adaptive Network with Thought Channels (v1–v8)
This section covers the development timeline from the first naive implementation to the final architecture featuring gate+head routing, EWC protection, and autonomous task change detection.

### Version 1 — First Implementation
**Idea:** Track the average activation of each neuron. If a neuron maintains a low activation for a long time—kill it.
**Implementation:**
* Activation function: Sigmoid in the hidden layer
* Pruning metric: Average absolute neuron activation
* Parameters: `death_threshold=0.02`, `death_patience=300`

**Problem and Diagnostics:**
Not a single neuron was pruned. Real activation analysis showed:
* Neuron 18: 0.1698
* Neuron 0: 0.8341

*Root Cause:* The Sigmoid function never outputs exactly 0; it's always in the range (0,1). All neurons appear 'active' even when contributing nothing. Adjusting the threshold would only be a surface-level fix—the metric itself was fundamentally flawed.
**✗ Result:** Zero neurons pruned — Sigmoid architecture prevents dead neuron detection.

### Version 2 — Redesign (ReLU + Importance Score + L1)
**Changes made and why:**
1. **ReLU instead of Sigmoid:** ReLU yields exactly 0 for negative inputs—useless neurons literally pass zero forward.
2. **Importance Score:** `importance[i] = |W2[i]| × mean(|a1[i]|)`. A neuron might be active but have a tiny outgoing weight. The importance score combines both factors.
3. **L1 Regularization on W2:** Pushes useless W2 weights towards zero. Useful neurons 'defend' themselves via strong gradients; useless ones offer no resistance.
4. **Burn-in Phase:** Pruning disabled for the first 3000 epochs to let the network learn the function first.
5. **Kill Interval:** Minimum 400 epochs between pruning events to allow structural recovery.

**✓ Result:** Network learned XOR and pruned excess neurons (20 → ~10 living neurons).

### Version 3 — Revival Mechanism
**Motivation:** The same network needs to learn XOR, AND, OR sequentially. Problem: Catastrophic forgetting (new gradients overwrite learned weights).
**Revival Design:**
* Track the moving average of the loss over the last N epochs.
* If the loss spikes above the average → trigger revival signal.
* Find a dead neuron → reinitialize (He init, not old weights).
* Add a `birth_epoch` marker to prevent immediate pruning of newborns.

*Bug Fix:* Newborn neurons were receiving `bias=0.0`. ReLU + zero weights + zero bias → importance=0 from epoch 1 → instantly treated as dead. Fixed by assigning a positive random bias so ReLU passes the signal.
**→ Status:** Revival infrastructure set up, but catastrophic forgetting remains unsolved.

### Version 4 — Autonomous Task Change Detection
**Key Observation:** A human shouldn't have to tell the brain "change the task"—the brain senses it from the data. The elegant approach is for the network to detect task shifts autonomously.
**Idea:** Cosine similarity of error vectors.
Instead of total scalar error, we track the error vector per sample:
`error_vec =[|pred_0-y_0|, |pred_1-y_1|, |pred_2-y_2|, |pred_3-y_3|]`
Cosine similarity measures the *angle* (the relative pattern of errors), not the magnitude. 

*Iterative Calibration (v4 to v4.2c):*
* `sigma=2.5` → 39 detections (✗ False positives)
* `sigma=4.0` → 9 detections (✗)
* `sigma=5.5` + `cooldown=5000` + `sim_history` limit → 2 detections (✓ Stable)

**✓ Result:** Adaptive detection calibrated — 2 correct detections for the XOR→AND transition.

### Version 5 — EWC (Elastic Weight Consolidation)
**Concept:** Upon detecting a task change, the network:
* Computes Fisher Information for every weight (measuring its importance for the current task).
* Freezes current weights as an 'anchor'.
* Adds an EWC penalty during the new task: `λ × F_i × (W_current - W_anchor)`.

**✓ Result:** EWC worked functionally, but revealed a fundamental architectural limitation. A single output neuron cannot mathematically represent XOR and AND simultaneously. EWC was attempting to solve an impossible problem.

### Version 6 — Gate + Head Architecture
**Key Insight:** A "Channel" isn't a predefined synapse. It's an activation pattern that spontaneously forms. We need a multi-output architecture where each channel has its own output dimension.
* `W_gate`: Learns WHICH channel is active (softmax competition).
* `W_head`: Learns WHAT each channel answers (sigmoid per channel).

Channel `k` receives a gradient ONLY if it wins the softmax. Neurons contributing to different channels receive conflicting gradients and gradually specialize.
**✓ Result:** Spontaneous specialization confirmed. XOR→K0, AND→K1, OR→K2 without explicit labels.

### Version 7 & 8 — Contextual Gate & Routing Agent
**Failure in v7:** Contextual gating failed to converge due to EWC conflicts and sequence misalignment.
**Success in v8 (Agent):** Built an external Q-learning Routing Agent.
* State: Short-term memory of the last 4 inputs + channel accuracy stats.
* Action: Select channel {0, 1, 2, 3}.
* Reward: +1 correct, -1 incorrect.

**Fundamental Conclusion of Part 1:** Autonomous channel selection without task signals is mathematically impossible for ambiguous inputs (e.g.,[1,1] means 0 in XOR, but 1 in AND/OR). True autonomy requires context. This perfectly motivates Part 2.

---

## PART 2 — Universal Adaptive Neural Network
**Core Idea:** Instead of the network learning *which* tasks it is solving (XOR channel, AND channel), it learns *how* to execute any logical operation based on its description.

**The Von Neumann Analogy:** 
The Von Neumann architecture is universal because it separates the processor from the program. Our network makes the same separation:
* **Processor** = Primitive Channels and Composition mechanism.
* **Program** = Truth table provided as an input.

### Block 1 — Universal Logic Interpreter ✓
**Goal:** The network receives `(input, truth_table)` and executes the operation.
* Format: `Input(6) = [x0, x1, t00, t01, t10, t11]`
* Split: 10 Boolean functions for training, 6 completely unseen functions for testing.

**Result:** 
* Training Accuracy: 100%
* Generalization (Unseen): 100%
* *Insight:* The network successfully learned the *meta-rule* of indexing a truth table based on binary inputs.

### Block 2 — Primitive Channels ✓
**Goal:** Prove that multiple parallel channels (`K=4`) can spontaneously specialize in interpreting different types of truth tables without explicit labels.
**Result:** 100% generalization. Channels spontaneously divided the workload based on the density of "1"s in the truth tables.

### Block 3 — Compositional Mechanism ✓
**Goal:** Demonstrate that the learned network can serve as a building block for complex operations (ALU behavior).
* **Half-Adder (Parallelism):** Executed `Sum = Net(A, B, instr=XOR)` and `Carry = Net(A, B, instr=AND)` simultaneously on the same hardware.
* **Chaining (Sequential):** Output of `(A XOR B)` passed directly as a float input to `(Y1 OR C)`. Both succeeded flawlessly (8/8).

### Block 4 — Complexity Estimator (Oracle) ✓
**Goal:** Energy efficiency. A system that predicts the difficulty of a task by only looking at the truth table. Simple tasks get 1 neuron; complex tasks get the full network.
**Result:** Oracle network correctly classified all 16/16 Boolean functions based on linear separability. 

### Block 5 & 6 — Pruning/Revival Integration & Final Test ✓
**Goal:** Combine the pruning/revival from Part 1 with the Universal Architecture from Part 2. Demand 100% accuracy on all 16 functions while aggressively pruning the network.

**Block 6 Setup:** 
* `burn_in` extended to 5000 epochs (ensure 100% generalization *before* pruning starts).
* Started with 64 neurons (4 channels × 16 neurons).

**Final Results:**
* Training accuracy: 100%
* Generalization on unseen functions: 100%
* **Neurons alive: 12/64** (81% reduction). Exactly 3 essential neurons survived per channel.
* Revival events: 0 (the network remained stable, pruning coexisted peacefully with generalization).

**Project Status:** SUCCESS. We successfully built a lean, dynamically allocating, universal neural network.
