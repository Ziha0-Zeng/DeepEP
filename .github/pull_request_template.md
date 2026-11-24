## 变更说明

- 背景和关联 Issue
- 变更内容和影响范围

## 验证结果

- CPU 测试命令和结果
- 涉及真机的变更请填写机型、EP 规模、测试命令和结果

## 合入检查

- PR 自动运行 `DeepEP / Lint`、`DeepEP / CPU` 和 `DeepEP / CI Contract`。
- 更新提交后，A5 检查等待管理员在本 PR 评论 `/run-npu-ci quick` 或 `/run-npu-ci full`。
- `DeepEP / A5 HT` 汇总环境与打包、Dispatch、Combine 和结果校验。设备数和运行范围以本次 Actions 报告为准。
- A5 runner 部署、触发和分支保护配置见仓库的 `docs/ci.md`。
