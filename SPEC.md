# LLM 量化实验 SPEC

## 1. 实验概述

### 1.1 实验目标
对比三种中文小型 LLM（Qwen3-1.7B、MiniCPM-2B、InternLM2.5-1.8B）在不同位宽量化下的困惑度(PPL)和生成质量。
以及 Gemma 3（如可用）。

### 1.2 参考实现（本仓库维护）
本 SPEC 对应的可复用实现脚本为：
- `scripts/dorefa_spec_experiments.py`：运行 BF16 / W-only / W+A（含 outlier ablation），输出 `summary.json`
- `scripts/plot_layer_quant_error.py`：输出逐层误差累计图（默认不计首尾层）
- `scripts/generate_spec_html_report.py`：输出 `report.html`（包含续写样例与指标汇总，并集成 layer error）

复现命令与 PDF 导出见：`docs/spec_report.md`。

---

## 2. 环境要求

### 2.1 硬件要求
- GPU: NVIDIA GPU (推荐 24GB+ 显存)
- 内存: 32GB+

### 2.2 软件依赖
```bash
# 核心依赖（不同模型需要不同版本）
torch>=2.0.0
transformers==4.51.0  # Qwen3-1.7B 需要
# 或 transformers==4.44.2  # InternLM2.5 可用
# 或 transformers==4.36.2  # MiniCPM 推荐

numpy>=1.24.0
```

### 2.3 模型路径
```
Qwen/Qwen3-1.7B/           # 28 layers
MiniCPM-2B-sft-bf16/   # 40 layers
models--internlm--internlm2_5-1_8b/  # 24 layers (脚本会自动解析 snapshots)
MiniCPM4-0.5B/         # MiniCPM4 0.5B
gemma-3-1b-it/         # Gemma 3 1B IT
```

### 2.4 数据集路径
```
swift_train/lgqm.txt  # 《临高启明》文本
```

---

## 3. 量化策略："除两层都量化"

### 3.1 策略说明
本 SPEC 的“除两层都量化”在实现上指：
- **不量化第 0 个 transformer block** 与 **最后一个 transformer block**（仍用 BF16）。
- **不量化 embedding 与 lm_head**（参考实现只量化 transformer block 内的若干线性层权重）。
- 量化对象默认为 **weight-only (W)**；可选扩展为 **weight + activation (W+A)**（见 3.4）。
- 本实验的量化为 **per-tensor** 且主要用于评估误差与效果：权重张量仍以 BF16 存储，仅将其取值限制到离散量化 levels（并非 GPTQ/AWQ 的 int packing）。

### 3.2 层配置
| 模型 | 总层数 | 量化层 | 保留层 | 说明 |
|------|--------|--------|--------|------|
| Qwen3-1.7B | 28 | 1-26 (26层) | 0, 27 | 输入+输出层保留 |
| MiniCPM-2B | 40 | 1-38 (38层) | 0, 39 | 输入+输出层保留 |
| InternLM2.5-1.8B | 24 | 1-22 (22层) | 0, 23 | 输入+输出层保留 |
| MiniCPM4-0.5B | 24 | 1-22 (22层) | 0, 23 | 输入+输出层保留 |

### 3.3 DoReFa-like Outlier-Aware 量化算法（W）

对每个待量化张量（权重或激活）$x$（展平为向量）：

1) **异常值阈值（按百分位）**  
设 outlier 百分位为 $p$（例如 1.0 表示 1%），令 $q=p/100$：

$$
a = Q_q(x), \quad b = Q_{1-q}(x)
$$

其中 $Q_t(\cdot)$ 表示 $t$ 分位数。异常值 mask：

$$
m_i = \mathbb{1}[x_i < a \ \lor\ x_i > b]
$$

注意：$p\%$ 表示**两端各保留 $p\%$**，因此实际“保留原值”的比例通常约为 $2p\%$（分布不对称时略有偏差）。  
当 $p=0\%$ 时，$a=\min(x), b=\max(x)$，从而 $m_i=0$，即**不做异常值特殊处理**（0% outlier ablation）。

2) **量化整数范围（对称）**  
位宽为 $B$：

$$
q_{\min}=-(2^{B-1}-1),\quad q_{\max}=2^{B-1}-1
$$

3) **缩放因子（与参考实现一致）**  

$$
s = \begin{cases}
\frac{b}{q_{\max}}, & b>0 \\\\
1, & b \le 0
\end{cases}
$$

4) **量化（仅对非异常值）**  

$$
\tilde{x}_i = \mathrm{clip}\big(\mathrm{round}(x_i/s), q_{\min}, q_{\max}\big)\cdot s
$$

最终输出：

$$
y_i = \begin{cases}
x_i, & m_i=1 \\\\
\tilde{x}_i, & m_i=0
\end{cases}
$$

5) **权重覆盖范围（参考实现）**  
仅量化每个 transformer block 内的若干线性层权重（不量化 layernorm/embedding/lm_head）：
- Qwen / MiniCPM / Gemma 风格：`q_proj,k_proj,v_proj,o_proj` 与 `gate_proj,up_proj,down_proj`
- InternLM2 风格：`wqkv,wo` 与 `w1,w2,w3`

### 3.4 Activation 量化（W+A）

W+A 中，除了 3.3 的权重量化外，还对 **每个中间 transformer block 的输出 hidden state** 做同样的 outlier-aware 量化。
为控制实验规模，参考实现默认仅在 **W>=6bit** 时启用 activation 量化。

1) **按层校准 bounds（BF16 baseline）**  
选取一段校准文本，前向一次得到所有层的 hidden states $\{h_\ell\}$。对每个目标层 $\ell$（通常 $\ell=1,\dots,L-2$）：

$$
a_\ell = Q_q(h_\ell),\quad b_\ell = Q_{1-q}(h_\ell)
$$

其中 $h_\ell$ 展平成向量后计算分位数。

2) **推理时按层量化**  
在第 $\ell$ 个 block 输出处应用 3.3 的量化算子（用该层的 $a_\ell,b_\ell,s_\ell$），得到 $\hat{h}_\ell$ 并继续向后传播。  
因为 $\hat{h}_\ell$ 会被下一层作为输入，所以这类误差会沿深度“累积”，见 5.3。

### 3.5 分位数估计与缓存（实现细节）

- 为减少计算开销，分位数计算可对张量做**等步长抽样**，最多取 `quantile_sample_size` 个元素，再在 CPU 上计算分位数。
- 权重 bounds 缓存：`<run_dir>/quantile_cache/*.json`
- activation bounds 缓存：`<run_dir>/activation_cache/*.json`（包含校准文本长度与序列长度等元信息）

---

## 4. 实验配置

### 4.1 Qwen3-1.7B (transformers==4.51.0)
```python
TARGET_LAYERS = list(range(1, 27))  # 28层，保留0和27
OUTLIER_PERCENTILE = 1.0  # 保留1%异常值

CONFIGS = [
    ("BF16 baseline", None),
    ("4-bit W (1.0%)", 4),
    ("5-bit W (0.1%)", 5),  # 使用0.1%异常值
    ("6-bit W (1.0%)", 6),
    ("6-bit W (0.1%)", 6),  # 使用0.1%异常值
    ("8-bit W (1.0%)", 8),
]
```

**结果:**
| 配置 | PPL | vs BF16 | 状态 |
|------|-----|---------|------|
| BF16 | 47.7926 | - | ✅ |
| 8-bit W (1.0%) | 47.6847 | -0.23% | ✅ 推荐 |
| 5-bit W (0.1%) | 48.4749 | +1.43% | ✅ |
| 6-bit W (1.0%) | 48.6693 | +1.83% | ✅ |
| 6-bit W (0.1%) | 50.3949 | +5.44% | ⚠️ |
| 4-bit W (1.0%) | 55.8537 | +16.87% | ⚠️ |

### 4.2 MiniCPM-2B (transformers>=4.36.0)
```python
TARGET_LAYERS = list(range(1, 39))  # 40层，保留0和39

CONFIGS = [
    ("BF16 baseline", None),
    ("4-bit W", 4),
    ("6-bit W", 6),
    ("8-bit W", 8),
]
```

**结果:**
| 配置 | PPL | vs BF16 | 状态 |
|------|-----|---------|------|
| BF16 | 38.3995 | - | ✅ |
| 8-bit W | 38.4077 | +0.02% | ✅ 最佳 |
| 6-bit W | 38.4576 | +0.15% | ✅ 推荐 |
| 4-bit W | 39.7531 | +3.53% | ✅ |

### 4.3 InternLM2.5-1.8B (transformers==4.44.2)
```python
TARGET_LAYERS = list(range(1, 23))  # 24层，保留0和23

CONFIGS = [
    ("BF16 baseline", None),
    ("4-bit W", 4),
    ("6-bit W", 6),
    ("8-bit W", 8),
]
```

**结果:**
| 配置 | PPL | vs BF16 | 状态 |
|------|-----|---------|------|
| BF16 | 42.2972 | - | ✅ |
| 8-bit W | 42.3069 | +0.02% | ✅ 推荐 |
| 6-bit W | 42.6515 | +0.84% | ✅ |
| 4-bit W | 73.5115 | +73.80% | ❌ 不可用 |

---

## 5. 评估指标

### 5.1 困惑度 (PPL)
定义 token 序列为 $x_1,\dots,x_N$，语言模型给出的条件概率为 $P(x_t\mid x_{<t})$。  
困惑度定义：

$$
\mathrm{PPL}=\exp\Big(\frac{1}{N-1}\sum_{t=2}^{N}-\log P(x_t\mid x_{<t})\Big)
$$

实现上使用**滑动窗口**在长文本上累计 NLL（避免一次性超长上下文导致 OOM）。每个窗口仅对“新引入”的 token 计入 loss（等价于对重叠部分做 mask）。

参考伪代码：
```python
def calculate_ppl(model, tokenizer, text, n_ctx=2048, stride=1024):
    device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + n_ctx, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        attn_chunk = attention_mask[:, begin_loc:end_loc] if attention_mask is not None else None
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        outputs = model(input_chunk, attention_mask=attn_chunk, labels=target_ids)
        nlls.append(outputs.loss * trg_len)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
```

### 5.2 生成质量 (续写示例)
使用4个中文提示词：
```python
SAMPLE_PROMPTS = [
    "萧子山看着眼前的虫洞，心中充满了",  # 叙事场景
    "文总说道：",                          # 对话场景
    "穿越到明朝之后，他们首先要解决的是",  # 历史场景
    "临高启明计划的核心是",                # 主题场景
]

def generate_sample(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    # 只解码新增 token，避免用字符串切片导致对齐不稳定（参考实现一致）。
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    continuation = tokenizer.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return continuation
```

**对齐要求（参考实现）**  
- 每个实验配置（即每条 PPL 记录）都使用同一组 `SAMPLE_PROMPTS` 生成续写，保证横向可比。  
- 参考实现会把每条记录的续写样例写入 JSON：`samples=[{"prompt":..., "continuation":...}, ...]`。  
- `report.html / report.pdf` 会为 **每条 PPL 记录** 展示对齐的续写样例（与该行配置一一对应）。  

### 5.3 逐层误差指标（Layer Error）

逐层误差用于解释“量化误差如何随深度演化”。参考实现输出两类曲线（通常以 layer index 为 x 轴，y 轴用 log scale）：

1) **权重量化误差（按层聚合，rel L2）**  
对第 $\ell$ 层（transformer block）中被量化的一组权重张量集合 $\mathcal{W}_\ell$：

$$
e_W(\ell)=\sqrt{\frac{\sum_{W\in \mathcal{W}_\ell}\lVert W^{(q)}-W\rVert_F^2}{\sum_{W\in \mathcal{W}_\ell}\lVert W\rVert_F^2}}
$$

2) **hidden-state “误差累计”（按层输出对比，rel L2）**  
令 BF16 baseline 在第 $\ell$ 层输出的 hidden state 为 $h_\ell$，量化推理（W 或 W+A）下同位置输出为 $\hat{h}_\ell$：

$$
e_A(\ell)=\frac{\lVert \hat{h}_\ell-h_\ell\rVert_2}{\lVert h_\ell\rVert_2+\varepsilon}
$$

之所以称为“累计”，是因为 $\hat{h}_\ell$ 已包含从前面各层传递过来的量化扰动；因此 $e_A(\ell)$ 反映了**截至深度 $\ell$ 的整体偏离程度**（不需要再对层做显式求和）。

绘图默认**不计首尾层**（不包含第 0 与最后一个 transformer block），与实验的“首尾保留 BF16”一致。

---

## 6. 可复现实现（推荐）

本实验的可复现/可复用实现已整理为脚本（见本文件 1.2 节），并配套生成：
- `summary.json`：各配置 PPL 与续写结果
- `report.html` / `report.pdf`：主报告（开头有各模型指标汇总，包含续写结果）
- `layer_error/*`：逐层误差累计图与索引页

复现流程与命令见：`docs/spec_report.md`。

---

## 7. 模型特定注意事项

### 7.1 Qwen3-1.7B
- **必需**: `transformers==4.51.0+`
- **加载参数**: `use_fast=False` (tokenizer)
- **架构**: `Qwen2ForCausalLM`
- **特殊层名**: `model.layers`, `self_attn.q_proj`, `mlp.gate_proj`

### 7.2 MiniCPM-2B
- **必需**: `trust_remote_code=True`
- **transformers**: `>=4.36.0`
- **架构**: 自定义 (需信任远程代码)
- **注意**: 需调用 `model.to("cuda")` 而非 `device_map`

### 7.3 InternLM2.5-1.8B
- **推荐**: `transformers==4.44.2`
- **架构**: `InternLM2ForCausalLM`
- **特殊**: 4-bit 量化会导致灾难性退化，避免使用

---

## 8. 结果分析模板

### 8.1 成功标准
- **PPL 增长 < 0.5%**: 几乎无损 ✅
- **PPL 增长 < 2.0%**: 可接受 ⚠️
- **PPL 增长 > 5.0%**: 不推荐 ❌

### 8.2 生成质量检查点
1. **语言一致性**: 续写应保持中文
2. **连贯性**: 语义连贯，不跳脱
3. **重复性**: 无严重重复 (如 InternLM 4-bit 的重复模式)

---

## 9. 常见问题

### Q1: 显存不足
**解决**: 减小 `n_ctx` 或增加 `stride`
```python
calc.calculate(n_ctx=1024)  # 默认 2048
```

### Q2: Transformers版本冲突
**解决**: 使用虚拟环境或按需切换
```bash
# Qwen3
pip install transformers==4.51.0

# InternLM
pip install transformers==4.44.2
```

### Q3: 结果不 reproducible
**解决**: 固定随机种子
```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## 10. 参考资源

- DoReFa 论文: https://arxiv.org/abs/1606.06160
- Qwen3: https://huggingface.co/Qwen/Qwen3-1.7B
- MiniCPM: https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
- InternLM2.5: https://huggingface.co/internlm/internlm2_5-1_8b
