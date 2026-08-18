# 冻结输入说明

这个目录保存可以直接提交的校准证据和配置清单，但不复制大型数据集或原始 SFT 轨迹。

## 文件

- `calibration-gate.json`：C0/C1/C2 已通过的 calibration/smoke 门禁及指纹。
- `docker-image.json`：当次 Docker 镜像和隔离能力预检记录。
- `protocol-selected.yaml`：被选中的 P1 协议及来源证明。
- `shared-sft-e3-seed42.json`：已有 shared-SFT state/weights 的只读来源记录，主要供 `--gate-only` 校验。

## 外部依赖

`configs/experiment.yaml` 期望本仓库旁边存在 `07-retool-lcb-mini`，并通过以下相对路径读取：

```text
../07-retool-lcb-mini/datasets/formal-v6/
../07-retool-lcb-mini/artifacts/training/shared-cold-start/source/manifest.json
```

在本仓库内创建本地数据链接：

```bash
ln -s ../../07-retool-lcb-mini/datasets/formal-v6 inputs/formal-v6
```

`inputs/formal-v6` 被 `.gitignore` 排除，因为它只应是本地链接，数据约 2.3 GB。正式运行前，代码会用 manifest、SHA-256、题目数量和实验指纹验证这些输入，路径指向错误或内容漂移时会直接停止。
