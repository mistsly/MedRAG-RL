# Codex 项目任务：MedSearch-RL —— 医疗问答 Agent 的 RAG 与 Agentic RL 学习仓库

## 一、项目背景与目标

我是一名 AI 领域科研人员，正在系统学习 Retrieval-Augmented Generation（RAG）、LLM Agent 和 Agentic Reinforcement Learning（Agentic RL）。

我希望你帮助我从零构建一个**学习导向、科研导向、可运行、可复现实验的开源项目**，暂时命名为 `MedSearch-RL`。

项目的具体任务是：构建一个能够检索外部医学知识库、通过多轮工具调用完成医疗问题回答，并使用强化学习优化检索策略的 LLM Agent。

**本项目最重要的目标不是做出一个功能齐全的医疗聊天机器人，而是帮助我理解并亲自实践 RAG、Agentic RAG 和 Agentic RL 的核心技术。**

计划使用五天完成一个最小但完整的系统，要求每个阶段可以独立运行、验证与学习。

最终应该具备：

1. 基础医学 RAG 系统，支持 BM25、Dense Retrieval、Hybrid Retrieval。
2. 基于 LangGraph 的 Agentic RAG，支持自主工具调用、多轮检索与回答。
3. 基于 GRPO 的 Agentic RL，能够真实更新 LLM 参数以优化其检索与回答策略。
4. 完整的实验评估模块，能够比较 No-RAG、RAG、Agentic RAG 和 RL-Agent。
5. 一套与源码对应的学习文档，解释关键算法、工程实现和实验设计。

## 二、技术栈与参考项目

优先使用以下开源基座，不要在已有优秀实现之外重复造轮子，但也不要把所有代码封装到无法理解的第三方接口中。

### 1. 医学 RAG：MedRAG

仓库：https://github.com/gzxiong/MedRAG

参考重点：

- 医学语料构建与管理。
- BM25、MedCPT、Contriever 等检索方法。
- 文档检索、排序、上下文构造与答案生成。
- i-MedRAG 的迭代检索设计。
- 医疗问答 Benchmark 与评估方式。

MedRAG 是主要的医学 RAG 学习基座。请先阅读它的实际代码与数据接口，明确哪些模块适合复用，哪些模块适合独立实现以帮助理解原理。

不要求部署 MedRAG 的全部大型知识库。第一版应优先实现小规模、单机可运行的 RAG。

### 2. Agent 框架：LangGraph

仓库：https://github.com/langchain-ai/langgraph

负责实现 Agent 的状态管理、条件分支、工具调用、循环检索和结束判断。

重点理解：

- StateGraph
- Agent State
- Tool Calling
- Action / Observation
- Conditional Edge
- Multi-turn Trajectory

尽量采用简洁、显式的状态转移，不要为了架构而引入多余的 Agent、抽象类和复杂中间层。

### 3. Agentic RL 参考：Search-R1

仓库：https://github.com/PeterGriffinJin/Search-R1

这是本项目最重要的强化学习研究参考。

重点研究：

- LLM 如何通过多轮交互使用搜索工具。
- 如何构建 Agentic RL 的训练数据。
- 如何采集和管理多轮 Rollout。
- GRPO 如何优化包含工具调用的 LLM 策略。
- 工具观察结果与 LLM 生成 Token 如何区分。
- 最终奖励如何作用于完整检索轨迹。

不要求直接迁移整个 Search-R1 训练系统。请重点借鉴其训练范式，使用更轻量的框架实现。

### 4. RL 训练框架：Hugging Face TRL

仓库：https://github.com/huggingface/trl

优先使用 TRL 的 GRPOTrainer 及相应的 LoRA/PEFT 训练方案。

训练必须包含真正的策略参数更新，而不是模拟奖励或仅使用 Prompt Engineering 优化 Agent。

请首先确认当前安装版本支持的 GRPO API、工具调用方式和多轮环境接口，再按照实际 API 设计训练模块，不要假设接口始终与旧教程一致。

### 5. 暂时不引入的框架

- Agent Lightning
- 完整 verl 训练栈
- RAG-Gym

这些项目可以写入扩展阅读文档，但五天内不作为必须依赖。

## 三、硬件与基础模型

当前主要计算资源：

- RTX 4090 24GB：用于项目开发、RAG、Agent 推理和小模型训练。
- A100 80GB：用于后续更大规模的训练实验。

模型选择：

- `Qwen/Qwen3-0.6B`：首先跑通完整 Agentic RL 训练链路。
- `Qwen/Qwen3-1.7B`：作为主要实验模型。
- `Qwen/Qwen3-4B`：作为可选扩展，不属于五天必做范围。

优先使用 BF16 + LoRA。只有在实际显存不足时，再评估 QLoRA、量化、梯度检查点等方案。

需要明确区分推理显存、训练显存、Rollout 峰值显存与检索服务的资源占用。不要假设 4090 24GB 一定能够运行任何长度、批量的 GRPO 训练。

项目应提供适合 24GB GPU 的轻量配置，并支持通过配置文件迁移到 A100。

## 四、数据集与知识库

### 主要数据集

PubMedQA：

https://huggingface.co/datasets/qiaojin/PubMedQA

优先使用 `pqa_labeled`，其回答标签为：

- yes
- no
- maybe

该任务便于使用确定性规则评估答案，也适合构建初步 RL Reward。

请首先核实数据集字段与官方划分，严格区分训练、验证和测试样本。官方测试集不得参与训练、奖励调参或模型选择。

### 知识库构建

第一版使用约 1 万篇 PubMed 摘要构建本地知识库，包括问题对应论文和额外干扰文献。

每篇文献尽可能保存：

- Document ID
- PMID
- Title
- Abstract
- Source

严禁把测试答案、标注解释或其他直接泄露正确标签的信息放入检索文档。

需要实现检索数据与训练数据的独立管理，并说明如何检查信息泄漏。

扩展数据集：

- MedQuAD：https://github.com/abachaa/MedQuAD
- MedMCQA：https://huggingface.co/datasets/openlifescienceai/medmcqa

这两个数据集暂时只作为扩展选项。

## 五、系统架构

目标工作流：

```text
Medical Question
       |
       v
Agent Policy (Qwen3)
       |
       +---- BM25 Search
       |
       +---- Dense Search
       |
       +---- Hybrid Search
       |
       +---- Query Reformulation
       |
       v
Retrieved Documents
       |
       v
Agent Observation & Reasoning
       |
       +---- Continue Searching
       |
       +---- Finish Answer
       |
       v
Final Answer + Evidence + Trajectory
```

训练阶段：

```text
Training Question
       |
       v
Policy Rollout
       |
       v
Multi-turn Tool Interactions
       |
       v
Final Answer + Search Trajectory
       |
       v
Reward Computation
       |
       v
GRPO + LoRA
       |
       v
Updated Agent Policy
```

RAG 和 RL 模块应当解耦。

尤其需要注意：普通 RAG、LangGraph Agent 和 RL 训练环境应尽可能复用同一套 Retriever、知识库和工具语义。

## 六、分阶段实施计划

### Day 1：基础 RAG 与检索器

学习目标：

理解文本 Embedding、稀疏检索、密集检索、索引构建与 Top-K 检索。

实现内容：

1. 下载并解析 PubMedQA 数据。
2. 构建本地医学摘要语料库。
3. 实现 BM25 检索。
4. 使用 MedCPT 或其他合适的 Embedding 模型实现 Dense Retrieval。
5. 建立向量索引。
6. 提供统一的 Retriever 接口。

交付结果：

能够输入问题并返回 Top-K 文献，以及其 ID、标题、相关性得分和文本内容。

学习文档需要解释 BM25 与 Dense Retrieval 的原理、区别、优势和局限。

### Day 2：完整 RAG 与检索评测

学习目标：

理解 Hybrid Retrieval、RRF、Reranking、Context Construction 和 RAG Evaluation。

实现内容：

1. 基于 Day 1 的检索器实现 Hybrid Retrieval。
2. 使用 RRF 融合不同检索结果。
3. 将检索内容提供给 Qwen3，生成医学问题答案。
4. 输出相关文献引用。
5. 实现基础评价指标。

需要支持：

- Recall@K（有 Gold Document ID 时）
- Answer Accuracy
- Citation Validity
- Retrieval Latency

比较 No-RAG、BM25-RAG、Dense-RAG、Hybrid-RAG。

### Day 3：Agentic RAG

学习目标：

理解 Agent 与普通 RAG 的本质区别，重点掌握状态管理、工具调用和自主决策。

使用 LangGraph 实现：

1. Agent 可以选择使用 BM25、Dense 或 Hybrid Search。
2. Agent 可以修改搜索 Query。
3. Agent 可以根据已获取证据决定是否继续搜索。
4. Agent 可以自主结束检索并输出答案。
5. 最多执行 3 轮检索，避免无界循环。
6. 记录完整的 Agent Trajectory。

每一步至少记录：

- 当前问题
- Agent Action
- Tool Name
- Tool Arguments
- Tool Observation
- 当前检索轮数
- 最终答案

需要提供一个可以直观看到 Agent 执行过程的调试脚本。

不要把 Agent 实现为固定执行的“先搜一次，再搜一次”流程，必须允许模型自主决策。

### Day 4：Agentic RL 与 GRPO

这是本项目的技术重点。

学习目标：

理解 Policy、Reward、Trajectory、Rollout、Advantage、GRPO 和 LoRA 训练之间的关系。

参考 Search-R1，结合 TRL 实现一个最小可运行的多轮工具调用训练环境。

#### Agent 状态与动作

State：

- 用户问题
- 已获取证据
- 检索历史
- 剩余工具调用预算

Action：

- 调用某个 Retriever
- 生成或重写检索 Query
- 提交最终答案

Observation：

- 检索工具返回的文献

Reward：

- 最终回答正确性
- 引用证据有效性
- 搜索次数惩罚
- 无效动作惩罚

初始奖励函数可以参考：

```python
reward = (
    answer_correctness
    + 0.2 * valid_evidence
    - 0.03 * search_count
    - 0.2 * invalid_action
)
```

该公式只作为初始方案，需要在代码中明确定义每个变量，并支持通过配置调整权重。

必须注意：

1. 最终答案正确性应当以标准标签为依据。
2. 不能仅凭模型自己声称“证据充分”就给予检索奖励。
3. 不允许 Agent 直接读取正确答案。
4. 检索工具返回的文本不能被错误地当成模型自己生成的 Token 参与策略梯度。
5. 奖励既要考虑回答准确率，也要避免无意义的重复检索。
6. RL 必须训练模型的工具调用和回答行为，而不只是训练一个独立的最终答案生成器。
7. 如果所用 TRL API 无法直接支持预期交互，需要说明原因，并采用其支持的最简单正确方案。

#### 训练配置

优先考虑：

- Qwen3-0.6B
- GRPO
- LoRA
- Group Size = 4
- 最多 3 轮检索
- 小规模训练集
- 单张 4090 24GB

先通过极小规模 Smoke Test 验证：

- Rollout 正常生成。
- 工具调用正常。
- Reward 正常计算且能够区分不同轨迹。
- Policy Loss 正常反向传播。
- 可训练参数真实更新。
- 模型权重能够保存并重新加载。

再尝试 Qwen3-1.7B 与更长训练。

最终应保存 Checkpoint、Reward 曲线、训练日志和示例轨迹。

### Day 5：整合与对比实验

最终比较：

| 组别 | RAG | Agent | RL |
|---|---|---|---|
| A | 无 | 无 | 无 |
| B | 固定 Top-K | 无 | 无 |
| C | 自主检索 | 有 | 无 |
| D | 自主检索 | 有 | GRPO |

主要指标：

- Answer Accuracy
- Gold Evidence Recall@K
- Citation Validity
- Average Search Count
- Invalid Search Rate
- Inference Cost

比较 RL 前后的工具调用轨迹，分析是否真的学到了更好的检索策略。

注意控制变量，包括模型、知识库、检索器、测试问题和推理预算。

最后提供一个简单的 Streamlit Demo，展示输入问题、检索过程、最终答案和引用文献。

## 七、推荐仓库结构

```text
MedSearch-RL/
├── configs/
├── data/
├── docs/
│   ├── 01_rag_fundamentals.md
│   ├── 02_retrieval_methods.md
│   ├── 03_agentic_rag.md
│   ├── 04_grpo_fundamentals.md
│   ├── 05_agentic_rl.md
│   └── 06_experiments.md
├── src/
│   └── medsearch_rl/
│       ├── data/
│       ├── rag/
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   └── pipeline.py
│       ├── agent/
│       │   ├── graph.py
│       │   ├── state.py
│       │   └── tools.py
│       ├── rl/
│       │   ├── environment.py
│       │   ├── rewards.py
│       │   └── train_grpo.py
│       └── evaluation/
├── scripts/
├── tests/
├── app.py
├── pyproject.toml
└── README.md
```

目录可以根据实际实现调整，不要求机械遵循。但应确保主要模块职责明确，且项目支持分阶段运行。

## 八、非常重要：这是一个教学性质的仓库

除了实现功能，你需要帮助我理解这些技术。

### 1. 代码需要具有教学价值

核心算法不能全部隐藏在黑盒接口中。

例如 BM25 可以使用成熟库负责实际检索，但应在文档中解释其评分过程、参数和实现位置。

GRPO 可以使用 TRL，但必须解释：

- 训练数据如何组织。
- 一个 Rollout 如何执行。
- 什么是完整的 Agent Trajectory。
- Reward 在哪里计算。
- Group Relative Advantage 如何得到。
- 哪些 Token 会参与梯度更新。
- LoRA 更新了哪些参数。
- Checkpoint 如何保存和恢复。

必要时使用公式、流程图和简短的代码示例。

### 2. 每个阶段必须可以独立运行

应该为每个阶段提供：

- 环境配置方法。
- 数据准备命令。
- 最小执行命令。
- 预期输出样例。
- 评价方法。
- 常见错误及其排查方式。

不要出现“必须执行完整五天项目才能测试第一天代码”的情况。

### 3. 注重工程可读性

遵循以下原则：

- 优先保证核心业务逻辑正确。
- 不进行过度抽象或过度防御性编程。
- 不引入不必要的配置框架、服务拆分和中间层。
- 不编写大量与核心功能无关的门控测试。
- 保留必要的参数校验、错误处理和实验可复现机制。
- 使用类型标注，核心复杂逻辑提供简洁注释。
- 避免过于庞大的单文件实现。
- 对技术设计的关键取舍说明理由。

### 4. 强调科研复现

至少支持：

- 固定随机种子。
- 保存实验配置。
- 保存 Agent Trajectory。
- 保存评估结果。
- 比较实验版本。
- 训练前后使用一致的评测逻辑。

禁止伪造实验指标、训练结果或模型文件。

无法在当前环境运行的实验，应明确区分“代码已实现”和“实际已验证”。

### 5. 允许小规模教学实验，但不能偷换概念

如果训练资源不足，允许降低模型规模、样本数量、检索轮数或生成长度。

但不能将以下行为称为完成 Agentic RL：

- 仅修改 Prompt。
- 仅对最终答案做 SFT。
- 只计算 Reward，没有梯度更新。
- 用固定规则假装模型自主检索。
- 伪造工具调用或训练日志。

医学答案应展示证据来源和不确定性，整个系统定位为研究原型，不作为临床诊断或治疗工具。

## 九、工作方式

你不仅是 Coding Agent，也承担科研学习助手的职责。

请采用以下开发流程：

**第一步：调研与规划**

先检查当前工作目录、Python/PyTorch/CUDA 环境，并阅读选定基座的真实代码和官方文档，确认关键依赖版本与接口。

输出简洁的架构设计、技术选型、五天任务表和预计的显存风险。

**第二步：建立最小可运行版本**

按照 Day 1 → Day 5 的顺序逐步实现，而不是先一次性生成所有文件的空壳。

每完成一个核心模块，运行对应的最小测试，确认实际可用。

**第三步：同步完善学习文档**

每个阶段必须留下相应文档，包含：

1. 该阶段解决的问题。
2. 重要理论知识。
3. 对应的核心源码。
4. 数据流或执行流程。
5. 如何运行。
6. 应当观察什么结果。
7. 有哪些值得继续研究的问题。

**第四步：阶段总结**

每个阶段结束时，简洁报告：

- 已经实现的模块。
- 实际通过的验证。
- 尚未完成的内容。
- 推荐我重点阅读的源码文件。
- 下一阶段要做的工作。

不要在有能力实现的情况下，仅输出设计建议而不编写代码。

## 十、最终验收标准

项目完成时，我应当能够：

1. 解释 BM25、Dense Retrieval 与 Hybrid Retrieval 的区别。
2. 独立构建一个小型医学 RAG 知识库。
3. 理解 LangGraph 的状态转移与工具调用机制。
4. 查看和分析完整的 Agent 检索轨迹。
5. 理解 Agentic RL 中 Policy、Action、Observation、Reward 的对应关系。
6. 使用 GRPO 真正训练一个能够调用检索工具的 Qwen3 Agent。
7. 对比普通 RAG、Agentic RAG 和 RL-Agent 的效果。
8. 将这一套代码改造成后续的 Agentic RL 科研实验基座。

**请从检查当前仓库环境、核实参考项目及其适用接口开始，制定实施计划，然后直接开始构建 Day 1 的代码和学习文档。优先保证第一阶段完整可运行，不要只生成一个五天项目的空目录结构。**
