# LL Combine 回传顺序与暂存流水实验

## 为什么测试这两项

EP32、H3584、TopK 8、384 个专家、BF16 的现有二级打点结果如下。两版各预热 10 轮、采样 10 轮，使用 Host MPI Barrier。表中数值为每轮 rank-MAX Event 的平均时延。

| BS | 原流程打点版，28 个接收核 | 36 个接收核 | 时延变化 |
| --- | ---: | ---: | ---: |
| 32 | 107.649 μs | 105.062 μs | −2.4% |
| 48 | 109.247 μs | 112.606 μs | +3.1% |
| 64 | 122.875 μs | 119.872 μs | −2.4% |
| 72 | 125.485 μs | 126.758 μs | +1.0% |

增加接收核数的收益尚不稳定。原流程 BS72 的最慢轮次中，一个接收核的主体耗时为 140,184 tick，其中未就绪轮询累计 83,123 tick。另一台机器的中位轮次中，一个接收核回传 3 包，提交累计 31,806 tick，其中 GM 暂存完成等待占 25,570 tick。两条记录来自不同样本，分别说明等待现象。

源码中，EP32 先处理本机回传，再处理跨机回传；每个跨机包暂存到 GM 后立即等待、提交。新实验分别调整这两个环节。暂存等待可能包含前序搬运与计算，不能直接作为纯 GM 拷贝耗时。当前数据也未量化设备启动偏斜。

## 优化开关

| 编译参数 | 默认值 | 修改范围 | 需要验证的现象 |
| --- | --- | --- | --- |
| `--remote-first ON` | OFF | DIRECT 和 FIRST_HOP 回传先遍历远端主机，最后遍历本机；远端主机之间保持原顺序 | RDMA 首包、末包是否提前，最终未就绪等待是否缩短 |
| `--defer-return ON` | OFF | 使用不同输出槽准备下一包，再提交上一包；每组末包立即提交 | 暂存等待是否减少，RDMA 提交时间和完整 Event 是否改善 |

`remote-first` 对单机 G2/G4 保留原顺序。EP32 使用 G16，每组对应一个物理主机。任务仍按轮转分给固定提交核；遍历变化也会改变具体任务所属的核和 QP，因此这是回传顺序与任务分配的组合实验。

`defer-return` 只在输出槽数大于 1、当前组还有下一条本核任务时启用。只有一个任务的组和每组末包立即提交。GM 暂存区、UB 容量和包格式保持原值。上一包提交后，SDK 才能复用它的 UB scratch；下一包使用另一个输出槽。满批继续按原规则 quiet，尾批在 kernel 结束前回收。FIRST_HOP 仍先确认组内 ready，再 Get、归约和回传。

延后提交可能减少等待，也可能推迟首包。需要同时检查等待耗时和实际提交时刻。默认关闭两个开关，冻结的 base kernel 保持原样。

## Dispatch 分支的参考价值

[metadata 分支的固定版本](https://github.com/TangYunfei147/DeepEP-LL/tree/3276f3a) 包含以下候选优化：

- 任务索引跨 count 记录累积，凑满 64 条后搬运，减少短记录的小 DMA 和同步。
- 用整数向量加法调整前缀偏移，偏移为 0 时跳过。
- 用 `(region + q) % 接收核数` 分配转发任务，分散稀疏 `q=0` 记录。
- 取得前缀后只扫描有效且待处理的 slot，复用已有 bitmap。

Combine 已有有效 token 的待处理列表和可选 FIRST_HOP 任务缓存。本次优先修改回传时机，保持生成测试输入的 Dispatch 实现一致。元数据分支还调整过 SHMEM 版本，这次继续使用当前锁定的 SDK。

[small-bs 分支的固定版本](https://github.com/TangYunfei147/DeepEP-LL/tree/65770a7) 在 `0 < BS < AIV数 − 1` 时设置一 token 一发送核，剩余核处理接收。56 核下，BS32 为 32/24，BS48 为 48/8，BS64 和 BS72 恢复 28/28。它还配套修改了 count 控制区布局。Combine 的发送工作量取决于专家输出和路由，继续保留独立的 `--recv-cores` 对照开关。

## 编译与运行

两台机器更新同一分支后，运行无参数编译脚本。它默认编译干净的 base，以及本节的 4 个打点版本；此前的任务缓存、分核和提前归约实验改为按需编译。脚本复用已安装的配套 SDK，无需安装 Python 包。构建列表和默认环境路径见 [C++ 用例说明](README.md#一次编译并运行当前实验)。

```bash
git pull --ff-only origin perf/ll-combine-small-batch
bash tests/ll_cpp_bench/build_all.sh
```

默认双机地址为 `90.90.93.84` 和 `90.90.93.184`。第一台执行：

```bash
bash tests/ll_cpp_bench/run_all.sh
```

第一台打印 `RUN_ID` 和第二台的加入命令。第二台只需复制该命令：

```bash
bash tests/ll_cpp_bench/run_all.sh 第一台打印的RUN_ID
```

默认对比本节的 4 个打点版本，测试 BS32、48、64、72 和 256。形状为 H3584、TopK 8、384 个专家、BF16、随机路由、seed 0；每个 BS 预热 10 轮、采样 10 轮，启用二级打点。脚本自动打印各 BS 的时延和打点汇总。每次从第一台启动都会生成新的 `RUN_ID`，保留此前数据。

默认顺序为对照、优先跨机回传、暂存流水和组合版。第二轮可以反序运行，检查先后顺序对结果的影响。第一台执行以下命令，第二台复制脚本打印的加入命令：

```bash
bash tests/ll_cpp_bench/run_all.sh \
    --variants return-profile,pipeline-profile,remote-profile,ref-profile
```

用 `--ep 4 --devices 6,7,14,15` 或 `--ep 16` 可以先做单机冒烟。主机地址、端口、编译并行度和形状参数都可覆盖，详见两个脚本的 `--help`。确认有收益后，用原有 `build.sh --profile OFF` 编译对应组合，再比较完整 Event 性能。

## 新增打点怎么看

schema 3 继续使用每核 1536 B 记录，新增字段如下。

| 字段 | 含义 |
| --- | --- |
| `rdma_first_stage` / `rdma_last_stage` | 首包开始暂存、末包暂存指令下发后的时刻 |
| `rdma_first_put` / `rdma_last_put` | 首次调用 RDMA Put 前、最后一次调用返回后的时刻 |
| `rdma_deferred_packets` | 实际延后提交的包数；单任务或单槽时为 0 |
| `rdma_defer_gap_ticks` | 暂存指令下发后，到开始等待暂存完成的累计间隔；包括下一包准备，单独展示 |
| `rdma_stage_wait_max_ticks` / `rdma_stage_wait_max_peer` | 单次暂存等待最大值及目标 peer |

`direct_first_submit` 等旧字段继续表示回传 helper 的调用位置。实际跨机下发查看新增 `rdma_*put` 字段。Put 返回表示请求已提交，数据到达与源端消费仍分别看接收 ready 和最终输出完成。

终端新增 `rdma_return` 行，并在选中轮次、rank 内展示 `RETURN largest_last_put_offset`。它按各核自身入口计算相对间隔，用于找候选拖尾核；跨核绝对完成先后还需时钟与入口校准。`defer_gap_ticks` 包含有用工作，程序将它单列，不作为热点耗时排名。

报告工具可以读取旧 schema 2 数据。新旧打点记录的字段和开销不同，性能对比使用相同构建版本和相同打点级别。Host 模型用于验证覆盖、槽位复用、事件和记录格式；CANN 编译、NPU 精度与性能仍需目标机器验证。

本次本地专项检查共 74 项通过，包含冻结原版逐文件校验、任务覆盖、延迟 DMA/NIC 的缓冲复用、全部 Combine 模板编译、打点解析和构建缓存。扩展运行 `test_build.py` 时，8 个已有测试因 Windows 的 Shell 可执行文件识别或符号链接权限失败；相关编译器发现逻辑保持原样。
