<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/agenthatch/.github/main/profile/assets/logo-dark.svg">
    <img alt="agenthatch" src="https://raw.githubusercontent.com/agenthatch/.github/main/profile/assets/logo-light.svg" width="600">
  </picture>
</p>

<p align="center">
  <strong>将任意 SKILL.md 编译为独立可运行的 AI Agent。</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/v/agenthatch?color=blue" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/pyversions/agenthatch" alt="Python versions">
  </a>
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/dm/agenthatch" alt="PyPI downloads">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://discord.gg/uSgU2HUD4">
    <img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord">
  </a>
  <a href="https://x.com/EterRights">
    <img src="https://img.shields.io/badge/X-@EterRights-1DA1F2" alt="X">
  </a>
</p>

---

## SKILL.md 的困境

SKILL.md 的承诺很美好：写一个 markdown 文件，告诉 Agent 该做什么，然后它就会照做。
但任何在 Claude Code、Codex CLI 或 OpenClaw 里用过三个以上 skill 的人，都知道真实体验是怎样的：

| 痛点 | 实际情况 |
|---|---|
| **无隔离** | 多个 skill 共享同一个上下文窗口，指令互相污染。文件整理 skill 和 Git 操作 skill 混在一起，Agent 搞不清哪条指令属于谁。 |
| **参考书，不是操作手册** | Agent 把 SKILL.md 当成建议而非契约。面对一个长 skill，模型像人读文档一样略读——挑出看起来相关的部分，忽略其余。 |
| **Token 浪费** | 每个 SKILL.md 都塞在 system prompt 里。加 5 个各 3KB 的 skill，对话还没开始就烧掉了 15KB 上下文。长任务场景下复利惊人。 |
| **零校验** | 工具名拼错、参数遗漏、指令歧义——Agent 到运行时才发现。而此时对话已经走了 20 轮。 |
| **规模衰减** | 1-3 个 skill 还行。10 个以上完全失控。没有依赖图，没有冲突检测，不知道谁覆盖了谁。 |

核心问题不是格式。是 SKILL.md 本质上属于 **prompt 工程**，不是**软件工程**。
你在让 LLM 在运行时、每次对话、零编译、零类型检查、零契约的情况下，解释人类自然语言。

---

## agenthatch 做了什么

agenthatch 把 SKILL.md 当作**源代码**——而不是 prompt。它通过一条确定性流水线将其编译为独立的 Python Agent，你可以导入、分发、部署到任何地方。

```
SKILL.md  →  解析  →  6-Harness LLM 推理  →  代码生成  →  可运行的 Agent
  (输入)    (阶段1)     (阶段2: AI 推理)      (阶段3: Jinja2)    (输出)
```

产出是一个自包含的 Python 包：有自己的 `pyproject.toml`、CLI 入口、带类型标注的工具定义、
MCP 集成以及运行时配置。它不是对 skill 的包装——它**就是** skill，编译成了代码。

---

## 快速开始

```bash
# 安装
pip install agenthatch

# 初始化 LLM 提供商
agenthatch init

# 添加 SKILL.md
agenthatch skills add ./my-skill/SKILL.md

# 孵化为 Agent
agenthatch hatch my-skill

# 运行
agenthatch run my-skill
```

三步从 markdown 到可运行 Agent。孵化后的 Agent 保存在 skillhouse 中，随时可重新运行。

---

## SKILL.md vs agenthatch

| | SKILL.md (原始) | agenthatch (孵化后) |
|---|---|---|
| **执行方式** | LLM 运行时解释 | 编译为独立 Python 包 |
| **隔离** | 所有 skill 共享一个上下文窗口 | 每个 Agent 有独立的运行时、工具和配置 |
| **校验** | 无——拼写错误和歧义到运行时才发现 | 代码生成前经过 AHSSPEC schema 校验 |
| **Token 开销** | 每轮对话注入完整 skill 正文 | ~150 字节运行时配置 |
| **工具定义** | 自然语言描述，LLM 猜测如何调用 | 带类型标注的 Python 函数 + JSON Schema |
| **MCP** | 每个 Agent 手动配置 | 自动检测，自动配置 |
| **确定性** | LLM 每次解读不同 | 同一份 SKILL.md → 同一份 AHSSPEC 结构（低温度推理） |
| **多 skill 扩展** | 3-5 个以上退化 | 无上限——每个 Agent 是独立进程 |
| **调试** | 读 LLM 思维链，祈祷 | 标准 Python 调试、日志、测试 |

---

## 架构

agenthatch 运行 **3 阶段流水线**，内含 6 个 AI Harness：

### 阶段 1：确定性解析（无 AI）

解析 SKILL.md 的 frontmatter、正文和目录文件。全程不涉及 AI——纯文件系统操作。
输出为 `ContextPack`，零语义转换。

### 阶段 2：6-Harness LLM 推理

6 个专用 AI Harness 按序处理，每个有独立的人格和温度配置：

| Harness | 职责 | 温度 |
|---|---|---|
| **A — 身份** | 从 frontmatter 提取名称、版本、描述 | 0.1 |
| **B — 意图** | 推断触发短语和用户意图 | 0.5 |
| **C — 接口** | 设计工具签名、参数和返回类型 | 0.5 |
| **D — 基座** | 检测运行时基类和指令结构 | 0.3 |
| **E — 装配** | 交叉校验所有 Harness 输出，生成 AHSSPEC | 0.2 |
| **F — MCP** | 检测并配置 MCP 服务器连接 | 0.3 |

每个 Harness 内部执行 **分析 → 推理 → 自校验 → 修正** 循环，最多 2 次重试。
Harness E 对其他五个输出进行交叉验证，产出统一的 AHSSPEC（Agent Hatch 标准规范）。

### 阶段 3：代码生成

Jinja2 模板将 AHSSPEC 渲染为完整的 Python Agent 包：

```
hatched-agent/
├── pyproject.toml          # pip 可安装包
├── runtime.toml            # LLM 提供商、模型、API Key
├── README.md               # 自动生成的使用文档
├── agenthatch.yaml         # AHSSPEC 清单
└── src/{package_name}/
    ├── __init__.py
    ├── agent.py            # Agent 类（继承 AHCoreAgent）
    ├── tools.py            # 带类型标注的工具实现
    └── references.py       # AI 提取的结构化数据
```

### 运行时：PlanLayer

生成的 Agent 使用 **PlanLayer 状态机**——一个 6 状态规划引擎，路径为
启动 → 规划 → 执行 → 验证 → 重新规划 → 完成。
它能在任务中途自适应：合并已完成步骤、失败时分支、工具超时时优雅降级。

---

## 工作明细

<details>
<summary>点击展开：完整流水线详解</summary>

### 第 1 步：`agenthatch init`

在 `~/.agenthatch/` 目录下配置 LLM 提供商。支持 OpenAI、DeepSeek、Anthropic
及任何 OpenAI 兼容接口。配置文件为 TOML 格式——可读、可版本化、易于分享。

### 第 2 步：`agenthatch skills add <path>`

将 SKILL.md 及其目录复制到 skillhouse 索引中。skillhouse 追踪你添加的每个 skill、
孵化状态以及生成 Agent 的存储位置。

### 第 3 步：`agenthatch hatch <name>`

完整流水线运行：

```
阶段 1 (确定性): 解析 SKILL.md → ContextPack
阶段 2 (AI): 6 个 Harness → HarnessOutput → 装配 → AHSSPEC
阶段 3 (Jinja2): AHSSPEC → Agent 包
```

可选参数：
- `--no-generate` — 跳过阶段 3，先审查 AHSSPEC
- `--force` — 覆盖已有孵化 Agent
- `--dry-run` — 预览，不写入文件

### 第 4 步：`agenthatch run <name>`

以交互式 TUI 模式启动孵化后的 Agent。Agent 加载运行时配置，连接 LLM 提供商，
启动对话循环，支持工具调用、上下文压缩和 PlanLayer 驱动执行。

</details>

---

## CLI 参考

| 命令 | 功能 |
|---|---|
| `agenthatch init` | 初始化配置和提供商设置 |
| `agenthatch skills add <path>` | 注册 SKILL.md 到 skillhouse |
| `agenthatch skills list` | 列出所有已注册 skill |
| `agenthatch skills delete <name>` | 从 skillhouse 移除 skill |
| `agenthatch hatch <name>` | 运行完整流水线（解析 → 推理 → 生成） |
| `agenthatch run <name>` | 以交互式 TUI 启动孵化后的 Agent |
| `agenthatch search <query>` | 搜索 skillhouse 索引 |
| `agenthatch doctor` | 诊断环境和依赖 |
| `agenthatch assemble` | 重新装配已有 skillhouse Agent |

---

## 安装

```bash
pip install agenthatch
```

要求 Python 3.11 及以上。

开发环境：

```bash
git clone https://github.com/agenthatch/agenthatch.git
cd agenthatch
pip install -e ".[dev]"
```

---

## 文档


| 文档 | 链接 |
|---|---|
| 贡献指南 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 安全策略 | [SECURITY.md](SECURITY.md) |
| 支持 | [SUPPORT.md](SUPPORT.md) |
| 路线图 | [ROADMAP.md](ROADMAP.md) |
| 行为准则 | [CODE_OF_CONDUCT.md](https://github.com/agenthatch/.github/blob/main/CODE_OF_CONDUCT.md) |
| 更新日志 | [CHANGELOG.md](CHANGELOG.md) |

---

## 社区

- [GitHub Discussions](https://github.com/agenthatch/agenthatch/discussions) — 问答、想法、路线图讨论
- [GitHub Issues](https://github.com/agenthatch/agenthatch/issues) — Bug 和功能需求
- [Discord](https://discord.gg/uSgU2HUD4)
- [X (Twitter)](https://x.com/EterRights)

---

## 参与贡献

agenthatch 目前由个人维护，正在寻找第一批贡献者。
Issue、Pull Request、文档、设计——任何形式的参与都欢迎。

详见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、质量门禁
（`hatch run quality:check`）和 PR 规范。

AI 辅助的贡献同样欢迎。提交前通过质量门禁即可——这是唯一的要求。

---

## 常见问题

### 面向谁？

任何维护超过 3 个 SKILL.md 文件并感受到摩擦的人。Claude Code、Codex CLI、OpenClaw
用户——如果你曾想过"要是这个 skill 是个真正的程序就好了"，这就是为你准备的。

### 能和 Claude Code / Codex / OpenClaw 一起用吗？

可以。孵化后的 Agent 是独立的 Python 包。你可以作为 CLI 运行、作为库导入，
或封装为 MCP 服务器。它不依赖任何特定 Agent 平台。

### 支持哪些 MCP 服务器？

任何使用标准 MCP 协议的服务器。Harness F 会自动检测 SKILL.md 中引用的 MCP 服务器，
并在生成 Agent 的运行时配置中自动配置。

### 这会取代 SKILL.md 吗？

不会。SKILL.md 是输入格式，agenthatch 是编译器。你仍然用 markdown 编写 skill——
agenthatch 将它们转化为 Agent。

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

---

<sub>📖 English version: [README.md](README.md)</sub>