# Personal Assistant 项目 — Git 工作流说明

> 写给 Yukang317 自己看的。你现在是"会 git add/commit/push 三板斧"的新手阶段，
> 这份文档帮你理解**正常项目开发该怎么用 Git**。

---

## 一、你现在的 Git 状态诊断

### 实际情况

```
main 分支，11 次提交，全部线性排列，没有其他分支。
```

```
413939c  阶段6.1初步测试
2de6914  feat(agent): 阶段6.1 Agent框架骨架完成
4023367  feat(agent): 阶段6.1 Agent框架骨架生成
3bf4ced  feat: 完成阶段5.5 PG迁移 + 阶段6技术方案设计
478bb8f  feat(rag): 完成路线2知识检索全链路 + RAGAS评估
a0cc583  feat(rag): 完成路线1文档摄入管线 + 路线2检索起步
3442e4d  feat: 初始化个人AI助手项目第一版
```

### 先说结论：**不奇怪，但可以做得更好**

| 维度 | 你现在的做法 | 正常 solo 项目 | 评价 |
|------|-------------|---------------|------|
| 提交次数 | 11 次 | 几十到几百次都正常 | ✅ 完全正常 |
| 提交粒度 | 每次提交对应一个功能模块 | 同左 | ✅ 合理 |
| Commit message | 用了 feat/docs 前缀 | Conventional Commits 规范 | ✅ 意外地规范 |
| 分支策略 | **只有 main** | 至少 2-3 个活跃分支 | ⚠️ 可以改进 |
| 推送频率 | 每次改完立刻 push | 不一定立刻 push | ⚠️ 可以改进 |

**11 次提交一点都不多**。一个中等规模的项目有几百次提交是常事。你看到的大项目可能有几千甚至几万次提交——这才是正常的。

---

## 二、为什么只有 main 分支不够好

### 你现在的工作方式

```
改代码 → git add . → git commit → git push
```

这就像一个游戏**只有一个存档位，还自动上传云端**：
- 中途写崩了？回不去了（除非用 reset/revert，那也麻烦）
- 想同时做两个功能？混在一起，没法分开提交
- 想实验一个大胆的想法？不敢，改坏了 main 就坏了

### 分支的类比

```
main 分支 = 出版的书（稳定、可读）
feature 分支 = 你的草稿本（随便写、随便撕）
```

**分支的核心价值：隔离风险。** 在 feature 分支上：
- 随便改、随便 commit、写崩了就扔掉这个分支
- 改完了、测好了 → 合并到 main → 删除 feature 分支
- main 分支永远保持"能跑"的状态

---

## 三、推荐的 Git 工作流（solo 项目版）

### 3.1 分支命名规范

```
main              ← 永远可运行、可部署的稳定版本
feature/xxx       ← 开发新功能
fix/xxx           ← 修 bug
experiment/xxx    ← 做实验，可能不合并
refactor/xxx      ← 重构代码（不改功能）
docs/xxx          ← 纯文档改动
```

### 3.2 日常开发流程

以一个典型的功能开发为例：

```bash
# 第 1 步：从最新的 main 创建一个功能分支
git checkout main
git pull origin main          # 确保 main 是最新的
git checkout -b feature/memory-system

# 第 2 步：在 feature 分支上写代码，随时 commit
# 注意：这里可以频繁提交！不用追求一次完美
git add agent/memory.py
git commit -m "feat(memory): 添加 Markdown 文件记忆存储"

git add agent/nodes/load_memory.py
git commit -m "feat(memory): 实现上下文加载时读取记忆"

git add tests/test_memory.py
git commit -m "test(memory): 添加记忆系统单元测试"

# 第 3 步：改完了，推送到 GitHub（仍然是 feature 分支）
git push -u origin feature/memory-system

# 第 4 步：在 GitHub 上创建 PR（Pull Request），自己审查一遍 diff
# gh pr create --title "feat(memory): 添加 Markdown 文件记忆系统"

# 第 5 步：确认没问题，合并到 main
git checkout main
git merge feature/memory-system

# 第 6 步：推送合并后的 main
git push origin main

# 第 7 步：删除 feature 分支（本地和远程）
git branch -d feature/memory-system
git push origin --delete feature/memory-system
```

### 3.3 关键心态转变

| 旧习惯 | 新习惯 | 为什么 |
|--------|--------|--------|
| 改完一大块才 commit | **小步提交**，改一个逻辑点就 commit | 回退时粒度更细，丢失更少 |
| commit 完立刻 push | push 是"发布"动作，**可以 commit 很多次再 push** | 本地 commit 是存档，push 是上传云端 |
| 所有改动混在一起提交 | **一次 commit 只做一件事** | 以后回头看日志，一眼知道每次改了什么 |
| 不敢 commit（怕"把坏的也存了"） | **大胆 commit**，反正 feature 分支上可以随便搞 | commit 不是"发布"，只是"存档" |

---

## 四、关于提交频率：多频繁算正常？

### 看两个真实例子

**小型功能分支**（你接下来要做的 memory 系统）：
```
feature/memory-system
├── a1b2c3d feat(memory): 定义 MemoryItem 数据模型
├── e4f5g6h feat(memory): 实现 Markdown 文件读写
├── i7j8k9l feat(memory): 在 load_context 节点中注入记忆
├── m0n1o2p refactor(memory): 抽取记忆格式化逻辑
├── q3r4s5t test(memory): 添加记忆读取测试
└── u6v7w8x docs: 更新 README 记忆系统状态
```
6 次提交，合入 main 后变成 1 个合并点。

**大型功能分支**（比如你将来做的"多平台接入"）：
```
feature/multi-platform
├── ...（20-30 次提交）
├── 可能在这个分支上还有子分支
└── 最终合并到 main
```

### 判断标准

| 提交频率 | 评价 |
|---------|------|
| 每改一行就 commit | 😅 太碎了，合并起来噪音大 |
| 每个逻辑点 commit 一次 | ✅ 最佳粒度 |
| 改了一整天才 commit | ⚠️ 太粗了，丢代码风险大 |
| 改了一周才 commit | 🔴 风险极高，而且冲突难处理 |

**一句话：commit 的频率 = 你愿意"回到这个存档点"的频率。**

---

## 五、你现在 11 次提交，回头看合理吗？

假设用了分支策略，你的历史可能长这样：

```
main
├── 3442e4d feat: 初始化项目
│
├── (merge) feature/rag-doc-ingest ──────────────┐
│   ├── a0cc583 feat(rag): 完成路线1文档摄入管线  │  这个阶段约
│   └── ...（解析器、分块器、存储等子提交）       │  3-5 次 commit
│                                                 │
├── (merge) feature/rag-retrieval ────────────────┤
│   ├── 478bb8f feat(rag): 检索全链路 + 评估      │  这个阶段约
│   └── ...（BM25、向量、RRF、评估等子提交）      │  3-5 次 commit
│                                                 │
├── (merge) feature/pg-migration ─────────────────┤
│   └── 3bf4ced feat: PG 迁移 + 阶段6方案         │
│                                                 │
├── (merge) feature/agent-framework ──────────────┤
│   ├── 4023367 feat(agent): 骨架生成             │
│   ├── 2de6914 feat(agent): 骨架完成             │
│   └── 413939c 初步测试                          │
│                                                 │
├── (merge) feature/agent-upgrade ────────────────┤
│   ├── f468b71 feat(agent): 意图路由升级         │
│   └── 035e7e5 feat(agent): 网页搜索工具         │
│                                                 │
└── (merge) docs/readme ──────────────────────────┤
    └── 286ed92 docs: 添加 README                 │
```

**看起来更清晰，对吧？** 每个大功能是一个分支，分支内的多次 commit 展示开发过程，合并后 main 保持整洁。

---

## 六、实际操作建议（从现在开始）

### 6.1 你不需要记住所有命令

你已经有了 git-helper skill，以后直接对我说：
- "创建一个分支做 memory 系统"
- "提交代码"
- "合并到 main"

我会引导你一步步操作，同时解释每个命令的含义。

### 6.2 记住三条铁律就够了

```
1. main 永远是能跑的代码
2. 新功能 = 新分支，改完了才合并回 main
3. 每次 commit 只做一件事，commit message 写清楚
```

### 6.3 下一步实践

你的项目接下来要做**阶段 6.2（工具扩展 + 记忆系统）**，这就是练习分支策略的好机会：

```
实际步骤（我来引导）：
  1. git checkout -b feature/memory-system    ← 创建分支
  2. 写代码，每完成一个小模块就 commit       ← 小步提交
  3. 全改完、测通了                            ← 确认稳定
  4. git checkout main && git merge ...        ← 合并回 main
  5. git push && 删掉 feature 分支            ← 收尾
```

---

## 七、常见疑问

### Q: 我一个人写代码，用分支是不是多此一举？
**不是。** 分支保护的是"你自己不被自己坑"——你昨天写的一段代码今天想推翻重来，但昨天那段已经被 merge 到 main 了，你就得 revert。如果还在 feature 分支上，直接 `git reset --hard` 回到三天前的状态就行。

### Q: 分支太多会不会很乱？
**不会。** 合并后删除 feature 分支，它就从历史中消失了（合并 commit 会保留，但分支名不占地方）。你会看到 `git branch` 里通常只有 `main`。

### Q: 多久 push 一次合适？
**看情况。** 本地 commit 可以很频繁（每 10 分钟一次都行），push 可以攒几个 commit 一起推。简单规则：
- 当天工作结束 → push
- 要切电脑/环境 → push
- 做到一半怕丢 → push（GitHub 就是你的云备份）

### Q: 不小心把坏的代码 commit 了怎么办？
**在 feature 分支上，完全无所谓。** 你可以继续改、继续 commit、也可以 `git reset` 回退。只要不 merge 到 main，就不影响稳定版本。

---

## 八、总结

```
你现在的位置：    会 git add/commit/push ✅
下一步要学的：    用分支隔离风险，小步提交
最终目标：        main 永远稳定，feature 分支拿来造
```

**你的 11 次提交历史完全正常，不奇怪。** 这只是开始——等你的项目做到阶段 7、8 的时候，100+ 次提交才是常态。关键是**每次提交都值得回头看**，而不是"改了一堆，随便写个 'update' 就交了"。

从现在开始，每个新功能都用分支来做。习惯了之后，你会觉得"没有分支才不习惯"。

---

**参考资源：**
- [Git 官方文档 — 分支基础](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%AE%80%E4%BB%8B)
- [Conventional Commits 规范](https://www.conventionalcommits.org/zh-hans/v1.0.0/)
- 项目的 git-helper skill：跟我说"提交代码"/"创建分支"/"看看状态"就能触发
