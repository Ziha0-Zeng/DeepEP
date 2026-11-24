# DeepEP CI 部署与门禁

本流水线参考 flash-linear-attention-npu 的管理员评论触发、可信 CI 定义、独立设备执行及结果汇总方式。
当前只覆盖基础 DeepEP 的 CPU、打包、Runtime 和单机 A5 HT，不宣称覆盖 LL、MegaMoE 或 EpGmmFused。

## 自动 CPU 检查

每次 PR、main push 和手动触发在 GitHub-hosted runner 上执行以下检查。

- `DeepEP / Lint`：固定版本的 pre-commit 全仓检查。
- `DeepEP / CPU`：Python 3.11、g++、CPU 版 PyTorch 2.12.0 和非 framework pytest。
- `DeepEP / CI Contract`：独立检查 Python 测试矩阵和报告契约、Node 状态回传逻辑、Bash 语法和工作流配置。

CPU 版 PyTorch 用于运行 CPU tensor 和 runtime mock 测试。A5 package 阶段使用匹配的 Torch/torch_npu
执行安装后的框架检查，设备精度与通信由后续 HT 阶段验证。

## A5 runner 的准备

准备独立的 Linux A5 测试环境，安装 GitHub Actions runner，添加自定义标签 `ascend950`、`deepep-ci`；
GitHub 自动添加的 `self-hosted`、`linux` 标签须保留。不要手动固定 ARM64/x64 标签，本 workflow 不限制主机架构。
runner 注册入口为仓库 Settings → Actions → Runners → New self-hosted runner，按页面给出的当前版本命令安装。

在 runner 服务使用的 Python 环境准备：

- Python 3.11、配套 Torch/torch_npu、CANN 9.2.0 和驱动、URMA 及 SHMEM 所需运行时；
- g++（C++17）、CMake、Make/Ninja、Bisheng、Git、flock；
- `requirements-build.txt` 和 `requirements-dev.txt` 中固定版本依赖；
- 从 `dependencies.lock.json` 固定提交构建、已登记 `deepep-sdk.json` 的 SHMEM SDK。

此版直接使用专用 runner 原生环境，不自动安装驱动、注册 runner、构建 Docker 镜像或重建 SDK。
SHMEM 版本不匹配时明确失败；更新锁文件后先由管理员维护 SDK。执行作业只调用 `record_shmem_sdk.py --check`。
离线环境可配置完整 wheelhouse，包含构建、测试依赖及传递依赖。

由管理员创建 `/etc/deepep-ci/runner.env`，并确保 runner 用户不能修改它。以下路径和端口仅为示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SHMEM_ROOT=/opt/deepep/shmem-sdk
source "$SHMEM_ROOT/set_env.sh"

export DEEPEP_CI_PYTHON=/opt/deepep/python/bin/python
# 必须明确预留 2、4 或 8 张卡；没有默认设备，不能空置。
export DEEPEP_CI_DEVICES=0,1
export DEEPEP_CI_MASTER_PORT=19090
export DEEPEP_CI_SHMEM_PORT=19091
# 预先创建 runner 可写的锁文件，所有本机 DeepEP runner 共用同一路径。
export DEEPEP_CI_LOCK=/var/lock/deepep-ci-machine.lock
# 可选；配置后 package 验证仅从该目录安装依赖。
# export DEEPEP_CI_WHEELHOUSE=/opt/deepep/wheelhouse
```

整次作业持有机器锁，设备数从列表推导，支持 EP2/4/8。设备和端口同时要从日常训练、调试任务中预留；
文件锁只约束遵守该锁的进程。配置变化后重新运行，报告会记录本次实际 EP。
同一 required context 允许这三种规模，必须由管理员维护固定覆盖策略；若要求 EP8，固定预留八卡并保护此配置。

自托管 runner 会编译和执行待检 PR 代码。使用专用账号和可重置的测试节点，隔离生产网络、凭据和私有源码。
有组织 runner group 时，将其限制到默认分支上的 `.github/workflows/npu-ci.yml`。
公共仓库的设备节点应采用每作业重置的环境，并在部署时验证只有预期的工作流可以访问 runner。
写状态的 prepare/finalize 在 GitHub-hosted runner 上执行，设备 job 只授予 contents read，checkout 不保留 Git 凭据。
作业脚本回收超时、取消或异常退出的子进程。宿主机故障或强制杀死 runner 后，复用设备前检查残留进程。

## 触发方式及执行范围

workflow 首次合入默认分支后，管理员可在 PR 评论：

```text
/run-npu-ci quick
/run-npu-ci full
```

也可以从 Actions 的 `DeepEP A5 HT CI` 手动填写 PR 编号。
只有仓库 Admin 可触发。工作流使用默认分支的 CI 定义，接受指向默认分支的非草稿、未关闭 PR。
触发前，把当前默认分支合入 PR 分支或将 PR 分支变基到当前默认分支。
评论 workflow 需要存在于默认分支才会接收 issue_comment 事件。

| 阶段 | 内容 |
| --- | --- |
| environment | 固定 SHMEM SDK 的版本、头文件和库校验和检查 |
| package | `validate_package.py` 的源码测试、wheel/sdist、sdist 重建、隔离安装、安装后测试及实际 Runtime probe |
| dispatch-server | server 去重，所有路由、BF16/FP8、权重切换、fresh/cached、输出生命周期 |
| combine-server-bf16/fp8 | 显式 `--with-dispatch --with-weights`，验证实际 Dispatch/Notify/Combine |
| combine-server-no-weights | 实际 Dispatch/Combine 的无权重路径 |
| EP8 的 rank 阶段 | EP8 额外重复上述 Dispatch 和三种 Combine 配置，显式使用 rank 去重 |

quick 使用 T33，full 使用 T257；都为 H7168/K6/E256、所有路由、重复三轮、无 profiler。
两种模式都满足本版 HT 门禁，full 增加跨 chunk 边界覆盖，不代表全规模验证。
Dispatch 和 Runtime probe 沿用 ElasticBuffer 的默认 1 GiB huge-page 分配，部署时必须为预留设备准备对应的 SHMEM huge-page 资源。
Combine 显式关闭 1 GiB huge-page 工作区要求，验证普通分配路径；其 huge-page 路径须另设用例。
任一阶段失败立即结束，不会运行后续阶段；缺少必需阶段的报告不能发布成功。
不能将 EP2 通过视为 EP8、跨机 HT 或 LL 已验证。

`scripts/validate_package.py --runtime-ranks` 只验证资源/事件生命周期，因此另执行 HT ST。
所有设备 ST 使用验证目录里的已安装 wheel 和复制后的测试，工作目录位于源码树之外。
每个命令有超时，Linux subreaper 及进程组清理回收本作业的子进程。

## 状态和产物

默认分支上的 PR 就绪、新建、更新、重开或修改时，元数据 workflow 为当前 head SHA 补写缺少的 pending 状态。
它使用默认分支的状态脚本。管理员触发后，prepare 固定 head/base/CI SHA，设备 job 检出精确 head 并验证它包含该 base。
同一 PR 的运行串行，正在运行的任务可完成，本机不同 PR 通过共享机器锁串行占用资源。

| PR 检查名 | 内容 |
| --- | --- |
| `DeepEP / A5 / Environment and Package` | 环境、wheel/sdist 安装和 Runtime probe |
| `DeepEP / A5 / Dispatch` | 本次 EP 规模要求的 Dispatch 用例 |
| `DeepEP / A5 / Combine` | 本次 EP 规模要求的 Combine 用例 |
| `DeepEP / A5 HT` | 总门禁，验证所有阶段及同一次运行的报告和产物 |

前三项用于定位失败，Actions Summary 同时显示各项结果。阶段失败后，后续阶段会显示已完成数量。

finalize 在 GitHub-hosted runner 上使用可信脚本，仅在以下条件都满足时发布成功：

- A5 job 和 artifact 下载都成功；
- 所有预期阶段成功、顺序正确，汇总任务独立读取并复核逐 rank 报告、检查数、源码身份及 Dispatch/Notify 校验标记；
- head/base/CI SHA、run ID、attempt、模式及 PR 编号一致；
- 归档 wheel 的实际 SHA256 与执行报告、package validation 一致；
- 发布前 PR head 和目标分支 commit 均未变化。

旧作业不会给新 commit 写成功。作业中 base 更新会使本次验证失败，更新 PR 后重新触发。
上传 result、逐阶段日志、package validation、JUnit XML、wheel/sdist 和 HT rank JSON，保留七天。
setup、checkout 或硬取消可能没有 result，这时最终门禁保持 pending 或发布 failure，绝不能放行。
日志含机器路径和测试信息，请按仓库可见性管理 artifact。定期维护 runner 工作区和临时产物的磁盘空间。

## GitHub 分支保护

1. 通过引导 PR 将 workflow 合入默认分支，确认 `DeepEP / Lint`、`DeepEP / CPU`、`DeepEP / CI Contract` 成功运行。
2. 在目标仓库 **Settings → Rulesets** 创建启用的默认分支规则，勾选 **Require a pull request before merging** 和
   **Require status checks to pass**，将上述三个自动检查设为必过。review 数量、Code Owners 和 bypass 按团队权限配置。
3. 配置 A5 runner 后，用最新默认分支上的测试 PR 发送 `/run-npu-ci quick`，确认三个分项、总门禁和归档产物完整。
   将 `DeepEP / A5 HT` 加入必过项，并启用 **Require branches to be up to date before merging**。
4. 在测试 PR 推送新提交，确认 A5 状态回到 pending 且 GitHub 阻止合入。重新触发后检查通过，再验证合入放行。

按 GitHub 展示的实际状态名称配置必过项。A5 分项可在 PR 和 Actions Summary 查看，总门禁负责强制校验整次 A5 运行。
本版没有 merge_group 触发，不接入 GitHub merge queue；需要 merge queue 时另行实现合并队列提交验证。

## 后续扩展

LL 要求 EP16+，独立接入多机调度后设置单独的 required context；不能用本单机 HT 状态替代。
MegaMoE、EpGmmFused 分别增加构建开关、依赖和设备精度矩阵。
性能先收集可重复基线，后续独立设置回归阈值，不把当前的正确性 quick/full 当作性能门禁。

参考：[GitHub runner 部署](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)、
[安全边界](https://docs.github.com/en/actions/reference/security/secure-use)、
[分支保护](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)。
