# 🚀 vLLM & Ray Parallelism & Deployment Simulator
### Interactive Multi-Architecture Enterprise AI Serving Playground
*Covering DeepSeek-V4-Flash on 8× NVIDIA B200 & Qwen 3.8 27B on 24× NVIDIA A100/A40 vGPUs*

An interactive, visual simulation suite explaining **The 4 Knives (Parallelism Methods: TP, TEP, PP, DEP/DP)** and **Deployment Strategies** for large frontier models running **vLLM** and **Ray Serve**, based on the Ray Summit 2026 Recap (*"Serving a frontier open model with Ray and vLLM — DeepSeek-V4-Flash on Nscale"* by Eyal Lantzman & Natdhanai Praneenatthavee) and enterprise production capacity planning for **Qwen 3.8 27B**.

---

## 🧭 The 3 Interactive Versions in this Suite

| Version | Live GitHub Pages Link | Primary Model | Hardware Target | Purpose & View Mode |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | [`index.html`](https://yuttananewj-nbtc.github.io/LLM_NBTC_OPTIMIZATION/index.html) | **DeepSeek-V4-Flash (284B MoE)** | **8× NVIDIA B200 (180GB HBM3e)** | **8-GPU Cluster & Topology View**: NVLink mesh, 4 Knives slicing matrix, Ray Summit benchmark metrics, and live packet stream. |
| **v2** | [`index_v2.html`](https://yuttananewj-nbtc.github.io/LLM_NBTC_OPTIMIZATION/index_v2.html) | **DeepSeek-V4-Flash (284B MoE)** | **8× NVIDIA B200 (180GB HBM3e)** | **3D Transformer Explainer View**: Isometric interactive model architecture, token flow through Attention + 256 Routed Experts, and 8-step guided tutorial. |
| **v3** | [`index_v3.html`](https://yuttananewj-nbtc.github.io/LLM_NBTC_OPTIMIZATION/index_v3.html) | **Qwen 3.8 27B (Dense Transformer)** | **24× NVIDIA A100 / A40 vGPUs** | **Enterprise Fleet Capacity Planner (2,000 Users · 30 Concurrency)**: Live VRAM SLA calculator, 24-vGPU fleet topology map, GQA sharding stage, and 4 production deployment strategies. |

---

## ⚡ Version 3 Deep Dive: Qwen 3.8 27B for 2,000 Enterprise Users

### 1. Enterprise Sizing & Workload Profile
- **Organization Size:** 2,000 total active enterprise employees.
- **Peak Concurrency:** **30 concurrent active inference requests** (1.5% peak concurrency rule for internal enterprise portals / chatbots).
- **Target SLA:**
  - TTFT (Time-To-First-Token) p50 < 120 ms
  - Streaming Speed > 45 tok/s (exceeds human reading speed of ~10–15 tok/s)
  - Zero queue drops under 30 concurrent load.

### 2. Model Profile: Qwen 3.8 27B Dense vs DeepSeek 284B MoE
| Dimension | DeepSeek-V4-Flash (v1 & v2) | Qwen 3.8 27B (v3) |
| :--- | :--- | :--- |
| **Model Type** | Sparse MoE (1 shared + 256 routed experts) | **Dense Transformer** (All layers execute every token) |
| **Active Parameters** | 13B active / 284B total | **27B active / 27B total** |
| **Attention Mechanism** | Multi-Head Latent Attention (MLA) + CSA/HCA | **Grouped-Query Attention (GQA)**: 48 Query Heads, 8 KV Heads (ratio 6:1) |
| **Layer Count** | 60 layers | **64 layers** |
| **Memory Footprint** | Checkpoint: 167 GB (FP4/FP8) | **BF16 = 54 GB**, **W8A16 = 27 GB**, **INT4 = 15 GB** |
| **Max Clean TP** | TP8 | **TP8** (constrained by 8 KV heads; 1 KV head per GPU) |

### 3. Why Slide 24 Rules of Thumb Flip for Dense 27B
In Slide 24 of the Ray Summit presentation, the authors gave rules of thumb for choosing parallelism knives:
1. **TP (Tensor Parallelism):** 
   - *Slide 24:* Latency-sensitive serving inside NVLink domain; dense layers shard cleanly.
   - *For Qwen 27B:* **TP is the undisputed latency champion!** Because all 27B weights are active on every token, sharding the dense GEMM across 2 or 4 GPUs cuts TTFT from ~180ms down to ~65ms.
2. **TEP / EP (Expert Parallelism):**
   - *Slide 24:* "Think twice when: Dense models (nothing to shard)".
   - *For Qwen 27B:* **Does not apply!** Qwen 27B has no router or 256 expert bank, so expert parallelism is not used.
3. **DP (Data Parallelism / Whole Replicas):**
   - *Slide 24:* Replicates weights across nodes/GPUs.
   - *For Qwen 27B:* **Throughput king!** With W8A16 (27GB) or INT4 (15GB), 1 single A40 (48GB) or A100 (80GB) holds the entire model with 15–53GB left for KV cache. Running 24 independent replicas (`24× TP1`) yields **0% cross-GPU communication overhead**!
4. **PP (Pipeline Parallelism):**
   - *Slide 24:* Crossing backend network between NVLink domains; bubbles dominate at low batch.
   - *For Qwen 27B:* Unnecessary on 48GB A40 or 80GB A100. However, on **24GB vGPU slices** with BF16 weights (54GB), PP or TP is mandatory to avoid OOM.

### 4. Hardware Spec Comparison across 24 vGPUs
| Hardware Option | VRAM / GPU | Interconnect & Bandwidth | Best For | 30 Concurrency Fit |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A40 48GB** | 48 GB GDDR6 | PCIe Gen4 (64 GB/s / 696 GB/s mem) | Enterprise On-Prem Private Cloud | ⭐ 12 Replicas × TP2 (2.5 streams/rep) |
| **NVIDIA A100 80GB** | 80 GB HBM2e | NVLink-3 (600 GB/s / 2,039 GB/s mem) | High-Performance Cloud Cluster | 🚀 6 Replicas × TP4 or 24 Replicas × TP1 |
| **vGPU 24GB Slice** | 24 GB Virtual | PCIe Shared / Hypervisor | Cost-Optimized Virtualization | W8A16 with TP2, or INT4 with TP1 |

### 5. The 4 Deployment Strategies on 24 vGPUs
- **Strategy A: 12 Replicas × TP2 (🏆 The Golden Enterprise Balance)**
  - 12 independent vLLM engines, each spanning 2 GPUs with TP2.
  - Concurrency per replica: 30 ÷ 12 = **2.5 concurrent users**.
  - On A40 (48GB): Model takes 13.5 GB/GPU (W8A16), leaving **32.5 GB VRAM for KV cache** per GPU (~496,000 tokens pooled KV cache per replica!).
  - TP2 minimizes PCIe bus contention on A40 while cutting TTFT to 88 ms.
- **Strategy B: 6 Replicas × TP4 (Deep RAG Context Pool)**
  - 6 independent engines, each spanning 4 GPUs with TP4.
  - Concurrency per replica: 30 ÷ 6 = **5 concurrent users**.
  - TTFT p50 drops to 65 ms. Deep pooled KV cache (~120–250 GB) easily handles 64K–128K token documents.
- **Strategy C: 24 Replicas × TP1 (Maximum Throughput)**
  - 24 independent whole-model replicas. Zero inter-GPU communication!
  - 30 concurrency ÷ 24 = **1.25 concurrent users per replica**.
  - Total cluster throughput reaches **1,000–1,250 tok/s**.
- **Strategy D: Prefill/Decode Disaggregation or Pipeline Parallelism**
  - Disaggregation: 8 vGPUs dedicated to high-batch Prefill (2× TP4) + 16 vGPUs dedicated to low-latency Decode (8× TP2) over NIXL.
  - Pipeline Parallelism: 12× PP2 or 8× PP3 assembly lines for low-bandwidth interconnects.

---

## 🌟 DeepSeek-V4-Flash Analysis (Versions 1 & 2)

### 1. 🧠 Model Context: DeepSeek-V4-Flash
- **284B Total / 13B Active Parameters** (Sparse Mixture-of-Experts with 1 shared + 256 routed experts; only 6 routed experts fire per token).
- **1,000,000 (1M) Token Context Window** kept affordable by hybrid **CSA** (Compressed Sparse Attention) + **HCA** (Heavily Compressed Attention).
- **167 GB FP4+FP8 Checkpoint** (74 files) shipping quantization-aware from day one.
- **DSpark Speculative Decoding Draft Built-in**: Draft model ships inside the same checkpoint (`num_speculative_tokens: 7`), providing an immediate **+1.20× (+20%) throughput boost** without downloading a separate model.
- **Hardware Tested**: 8× NVIDIA B200 (180GB HBM3e per GPU, 1.8 TB/s NVLink-5 interconnect domain).

---

## 🔪 The 4 Knives (Parallelism Methods — Slide 23 & 24)

| Knife | How it Slices | Interconnect / Comm | Reach for it when (Slide 24) | Think twice when (Slide 24) | Role in Demo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TP** *(Tensor Parallel)* | Slices **EVERY** layer (Attention & Experts) across GPUs | Constant high-frequency **All-Reduce** across GPUs | Latency-sensitive serving inside one NVLink domain; dense layers shard cleanly. | Long context KV hungry; links slower than NVLink; heads don't divide cleanly. (KV moved 5.03M vs 5.05M). | Attention slicing in Strat A; Prefill pool in Strat C (TP4). |
| **TEP** *(Tensor + Expert Parallel)* | Slices **Attention** (TP) + assigns **whole intact experts** (32/GPU) | Lightweight TP All-Reduce + local/NVLink expert execution | Sparse MoE like V4-Flash (6 of 256 active); intact experts run at full speed without sharding matmul. | Dense models; expert routing crossing beyond NVLink domain; imbalanced load without EPLB. | **Strategy A (Winner)**: 2,209 tok/s on single 8× B200 node! |
| **PP** *(Pipeline Parallel)* | Cuts model vertically into layer stages (assembly line) | Point-to-point activations over **RoCE / InfiniBand** | Crossing backend network between NVLink domains; freeing HBM once TP is maxed out. | Low-batch, latency-critical traffic: **Pipeline Bubbles** dominate at small batch sizes. | Out of scope for single model: Strategy D scales across machines with Whole Replicas instead! |
| **DEP** *(Data + Expert Parallel)* | Replicates **Attention** (DP8) + shards 256 experts (32/GPU) | **All-to-All** token dispatch & combine across all 8 GPUs | Throughput-per-dollar at massive scale; decode pool in P/D setups. | Single-user latency (All-to-All bound); replicating attention wastes HBM. (*TEP8 beat DP8×EP 2,209 vs 1,634 tok/s*). | **Strategy B** (Reference recipe) & Strategy C Decode pool. |
| **DCP** *(Decode Context Parallelism)* | Shards decode KV cache across TP ranks | Intra-node TP sync | Decode-heavy, very long context (1M tokens). | Prefill-heavy traffic; DCP size must divide TP size. | Special knob for ultra-long context decode. |

---

## 🚀 The 4 Tested Deployment Shapes (Strategies A–D — Slide 28)

| Strategy | Architecture Shape | Aggregate Throughput (32 con) | Single Speed | TTFT p50 (single) | TPOT p50 (single) | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | **TEP8** (TP8 Attention + 32 Experts/GPU) | **2,209 tok/s** ★ | **140.0 tok/s** | **0.141 s** | **6.62 ms** | 1 Node (8× B200) |
| **Strategy B** | **DP8×EP + DSpark (k=7)** (Published recipe) | **1,634 tok/s** | **65.9 tok/s** | 0.193 s | 14.27 ms | 1 Node (8× B200) |
| *Strategy B plain* | DP8×EP (Speculator OFF) | 1,358 tok/s | 49.5 tok/s | - | - | 1 Node (8× B200) |
| **Strategy C** | **P/D Disaggregation** (TP4 Prefill + DP4×EP Decode over NIXL) | ~1,700 tok/s *(Bug on 2+2)* | *Garbage Out* ⚠️ | - | - | 1 Node (4+4 GPUs) |
| **Strategy D** | **Whole Replicas** (2× TEP8 Nodes behind Ray Serve) | **4,647 tok/s** 🏆 *(2.10×)* | **140.0 tok/s** | **0.145 s** | **7.45 ms** | **2 Nodes (16× B200)** |

### 🚨 Critical Takeaways from the Talk
1. **Match the strategy to the wiring:** Keep chatty parallelism (expert routing, cache handoffs) inside the 1.8 TB/s NVLink domain. Between machines (RoCE), send only light pipeline activations or complete HTTP requests.
2. **Disaggregation is a pattern, not a cluster size:** P/D works even on a single machine (2 pools + NIXL connector). But on a single 8-GPU node, splitting 4+4 forced TP4, which hit a silent bug in vLLM v0.27.1 where `"what is 2 + 2"` returned gibberish! In production, span P/D across 2 nodes so both pools use known-good TP8.
3. **The cache is still the product at 1M:** Slide 26 dials:
   - Pair `--kv-cache-dtype fp8` with `--block-size 256` always.
   - Use `--attention-config '{"use_fp4_indexer_cache":true}'` to shrink sparse attention indexer to 4-bit.
   - Cap long thinking tokens (up to 64K output).
   - At 1M context, KV cache holds 15.6M tokens (vs 5.0M at 256K), costing 1.3× in concurrency (19 vs 26 streams).
4. **The flags ARE the architecture:** Every deployment shape is one `vllm serve` argument away. Switching strategies is a single `kubectl apply` against a shared ReadWriteMany (RWX) 167 GB PVC.

---

## 🌐 1-Minute GitHub Pages Deployment Guide

To host this interactive simulation directly on your **`https://<username>.github.io/<repo-name>/`**:

### Step 1: Initialize Git and Commit
```bash
cd "d:\06_งานปี 69\18_vLLM Simulation"
git init
git add .
git commit -m "feat: Add interactive vLLM & Ray Parallelism Simulation (v1, v2, v3)"
```

### Step 2: Push to GitHub
```bash
# Create a new repository on github.com (e.g. vllm-ray-simulation)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

### Step 3: Enable GitHub Pages
1. Go to your repository on GitHub: `https://github.com/<your-username>/<your-repo-name>`
2. Click **Settings** (แถบเมนูด้านบน)
3. In the left sidebar, click **Pages**
4. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: **`main`** / Folder: **`/ (root)`**
5. Click **Save**
6. Within 1 minute, your site will be live at:
   - **Version 1 (Default):** `https://<your-username>.github.io/<your-repo-name>/`
   - **Version 2:** `https://<your-username>.github.io/<your-repo-name>/index_v2.html`
   - **Version 3:** `https://<your-username>.github.io/<your-repo-name>/index_v3.html`! 🎉

---

## 📁 Repository Structure
```
├── index.html       # Version 1: 8-GPU Cluster View & Topology Mesh (DeepSeek-V4-Flash)
├── index_v2.html    # Version 2: 3D Transformer Explainer View (DeepSeek-V4-Flash MoE)
├── index_v3.html    # Version 3: Enterprise Capacity Planner (Qwen 3.8 27B on 24× A100/A40 vGPUs)
├── .nojekyll        # Disables Jekyll processing for full compatibility on GitHub Pages
├── README.md        # Comprehensive bilingual documentation & presentation recap
└── natdhanai-praneenatthavee-ray-summit-2026-recap-opening.pdf  # Source presentation deck
```

---

*Presented by Eyal Lantzman (Nscale) & Natdhanai Praneenatthavee at Ray Summit 2026 Recap, extended with Qwen 3.8 27B Enterprise Capacity Planning.*

---

*Presented by Eyal Lantzman (Nscale) & Natdhanai Praneenatthavee at Ray Summit 2026 Recap.*
