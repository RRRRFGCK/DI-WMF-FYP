"""Generate the repair completion ledger after both independent QA passes."""
from pathlib import Path
import csv
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).parent
OUTPUT = ROOT / "outputs_correncoder_patient46_20260908"


def main():
    integrity = json.loads((WORK / "patient46_integrity_summary.json").read_text())
    replay = json.loads((WORK / "new_patient_cuda_replay_summary.json").read_text())
    assert integrity["complete"] and not integrity["issues"]
    assert integrity["new_fits_checked"] == 39
    assert integrity["reused_fits_checked"] == 559
    assert integrity["record_outputs_checked"] == 689
    assert replay["all_passed"] and replay["checkpoints_replayed"] == 13
    global_manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert len(global_manifest["experiments"]) == 13
    rows = []
    configuration_rows = []
    for experiment in global_manifest["experiments"]:
        assert experiment["status"] == "complete"
        assert experiment["records_completed"] == 53
        assert experiment["patients_completed"] == 46
        assert experiment["reused_patient_models"] == 43
        assert experiment["new_patient_models"] == 3
        label = experiment["experiment"]
        for patient in ("s03386", "s11342", "s25323"):
            path = OUTPUT / label / "groups" / patient / "group_manifest.json"
            group = json.loads(path.read_text())
            assert group["status"] == "complete" and not group["reused"]
            assert group["epochs_completed"] == 80
            rows.append({
                "experiment": label, "patient_id": patient,
                "test_records": ";".join(map(str, group["test_record_folds"])),
                "training_records": group["train_records"],
                "training_patients": group["train_patients"],
                "seed": group["group_seed"], "epochs": group["epochs_completed"],
                "initialisation_seconds": group["initialisation_seconds"],
                "training_seconds": group["training_seconds"],
                "wall_seconds": group["wall_seconds"],
                "gpu_peak_allocated_bytes": group["gpu_peak_allocated_bytes"],
                "gpu_peak_reserved_bytes": group["gpu_peak_reserved_bytes"],
                "checkpoint_sha256": group["checkpoint_sha256"],
                "group_manifest": str(path),
            })
        configuration_rows.append({
            **experiment,
            "record_metrics": str(OUTPUT / label / "record_metrics.csv"),
            "patient_metrics": str(OUTPUT / label / "patient_metrics.csv"),
            "manifest": str(OUTPUT / label / "manifest.json"),
        })
    for filename, data in (("new_fit_completion_ledger.csv", rows),
                           ("completed_configuration_ledger.csv", configuration_rows)):
        with (WORK / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": global_manifest["protocol_version"],
        "configurations_completed": 13, "patient_models": 598,
        "new_patient_models": 39, "reused_singleton_models": 559,
        "record_outputs": 689, "new_epochs_total": 3120,
        "summed_training_seconds": sum(r["training_seconds"] for r in rows),
        "summed_group_wall_seconds": sum(r["wall_seconds"] for r in rows),
        "peak_allocated_bytes_max": max(r["gpu_peak_allocated_bytes"] for r in rows),
        "peak_reserved_bytes_max": max(r["gpu_peak_reserved_bytes"] for r in rows),
        "integrity": integrity, "cuda_checkpoint_replay": replay,
        "output_manifest": str(OUTPUT / "manifest.json"),
        "frozen_source_manifest": str(WORK / "frozen_source_manifest.json"),
        "sampling_policy": "record-balanced training; patient-disjoint testing and equal-patient reporting",
        "original_outputs_overwritten": False,
    }
    (WORK / "completion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# 患者级回归修复完成记录

13 个配置全部完成：39 个新患者组模型 × 80 epochs；559 个单记录患者模型有条件复用；共 598 个患者模型、689 份记录级波形。

## 身份与复用条件

按原始 MAT 的 `fix.id` 分组，53 份记录来自 46 名患者。多记录患者 s03386（原始零基 folds 5–8）、s11342（19–22）、s25323（37–38）各排除全部所属记录，分别使用 seed 60、74、92，训练集包含 49、49、51 份记录且均来自另 45 名患者。

每配置其余 43 个患者均只有一份记录。其原训练记录顺序、逐记录预处理、seed 和批次协议不变，checkpoint/history/waveforms 按来源 SHA-256 逐份验证复制。因此 39 次补训足够的结论仅适用于保留原 record-balanced 训练协议；并非 patient-balanced 训练，也不适用于重新编号种子或改变训练窗采样。

新模型固定 final epoch 80、Adam lr 0.001、batch 30，保留原损失、初始化、外部 CapnoBase checkpoint 和校准配置。不使用测试指标择优或挑选 epoch。缺失历史配置元数据以显式 repair defaults 标注，而不是伪称恢复了历史启动参数。

## 验证

- 完整身份/来源/指标检查：598 个患者组、689 份波形，issues 为空；所有新旧组均为 45 个训练患者且与测试患者无交集。
- 559 个复用组的 checkpoint、history 和 waveforms 与原来源 SHA-256 相同。
- 39 个新模型均完成 80 epochs；配置、训练身份、代码 SHA-256、外部权重/数据来源及时间/显存记录均已落盘。
- 每配置一个新患者组 checkpoint 在原 CL2 CUDA 环境独立 reload/inference，共 13 个；最大预测差值 {replay['max_prediction_difference']:.12g}，最大 MSE 差值 {replay['max_mse_difference']:.12g}，最大 MAE 差值 {replay['max_mae_difference']:.12g}。这不是对全部新 checkpoint 的逐一回放，全部 39 个均完成了哈希与结果完整性检查。
- 全部 689 份 target 均与原 loader 输出的 14,388 样本参考波形逐元素一致；历史 FFT RR 数组及零/最大滞后相关均由保存波形重新计算通过。

## 输出及保留的限定

新结果只写入 `outputs_correncoder_patient46_20260908`；未覆盖历史结果。入口 `run_correncoder_patient_loso.py`，协议和 schema 见本目录 `README.md`，源代码冻结副本及其哈希见 `frozen_source_manifest.json`。

记录级 `test_mse`/`test_mae` 仍是 471 个重叠窗口上的逐样本误差，相关和 RR 使用融合波形。患者统计先平均患者内记录，再对 46 名患者等权平均。这里的 t 区间仅为描述性；LOSO 训练集重叠、同一预训练 checkpoint 和单种子限制不因身份修复消失。

本 runner 保留历史同 FFT 峰值的 RR 参考以隔离身份修复。人工事件参考、修正 classical baseline、敏感性和统一多重比较另由独立评估模块生成，不应将这里的旧参考汇总误称为人工标注 RR。全记录归一化/离线预处理和继承的网络/损失定义未在此次补训中改变。

所有新组 training time 累计 {summary['summed_training_seconds']:.3f} 秒，组内 wall time 累计 {summary['summed_group_wall_seconds']:.3f} 秒。累计值不是单次任务端到端耗时；PyTorch allocator peak 不包含驱动上下文与其他进程。各配置及患者逐组记录见 `new_fit_completion_ledger.csv`。
"""
    (WORK / "completion_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
