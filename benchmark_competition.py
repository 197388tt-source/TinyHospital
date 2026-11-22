"""One-command, offline competition benchmark and Markdown report generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from measure_decision_transfer import measure as measure_binary_decision
from performance_baseline import measure as measure_th_acp
from text_baseline import measure as measure_text


ROOT = Path(__file__).resolve().parent


def _pct(old: float, new: float) -> float:
    return round((old - new) * 100 / old, 2) if old else 0.0


def benchmark(case_count: int = 10, *, include_process: bool = True) -> dict:
    """Run comparable pure-text and TH-ACP baselines without external services."""
    if case_count != 10:
        raise ValueError("当前 TH-ACP 固定基线使用前 10 个病例；case_count 必须为 10")
    text = measure_text(case_count)
    structured = measure_th_acp()
    structured_view = {
        "mode": "th_acp_structured",
        "method": structured["method"],
        "case_count": structured["case_count"], "case_ids": structured["case_ids"],
        "message_count": structured["packet_count"],
        # Compare business-message content to text messages.  Packet-envelope
        # bytes are retained separately because a natural-language message bus
        # has no directly equivalent TH-ACP header.
        "total_message_bytes": round(sum(step["mean_message_bytes"] for step in structured["steps"]) * structured["case_count"]),
        "total_message_tokens_estimate": round(sum(step["mean_message_tokens_estimate"] for step in structured["steps"]) * structured["case_count"]),
        "total_packet_bytes": round(structured["mean_case_packet_bytes"] * structured["case_count"]),
        "mean_case_elapsed_ms": structured["mean_case_execution_ms"],
        "exact_matches": structured["exact_matches"],
        "non_text_transfer_count": 0,
        "non_text_transfer_bytes": 0,
        "note": "This baseline measures structured TH-ACP packets. The separate decision-vector result below measures actual non-text Pipe transfer.",
    }
    if text["case_ids"] != structured_view["case_ids"] or text["exact_matches"] != structured_view["exact_matches"]:
        raise RuntimeError("两种模式的病例或确定性推荐结果不一致，拒绝生成不公平对比")
    result = {
        "benchmark_version": 1,
        "reproducibility": {
            "dataset": "DIALMED traincut.txt 前 10 例",
            "network": "not used",
            "model": "deterministic local stub",
            "quality_equivalence_checked": True,
        },
        "pure_text": text,
        "th_acp_structured": structured_view,
        "comparison": {
            "same_cases": True,
            "same_exact_matches": True,
            "message_bytes_saved": text["total_message_bytes"] - structured_view["total_message_bytes"],
            "message_bytes_saved_percent": _pct(text["total_message_bytes"], structured_view["total_message_bytes"]),
            "communication_tokens_saved": text["total_message_tokens_estimate"] - structured_view["total_message_tokens_estimate"],
            "communication_tokens_saved_percent": _pct(text["total_message_tokens_estimate"], structured_view["total_message_tokens_estimate"]),
        },
    }
    result["non_text_state_transfer"] = measure_binary_decision() if include_process else {
        "status": "skipped", "reason": "--skip-process was selected"
    }
    return result


def format_report(result: dict) -> str:
    text, structured, comparison = result["pure_text"], result["th_acp_structured"], result["comparison"]
    rows = [
        "# Tiny Hospital 赛题统一评测报告",
        "",
        "固定运行 DIALMED `traincut.txt` 前 10 例。两种模式使用相同的八步医疗协作与确定性角色策略，"
        "不访问网络或模型 API；因此可重复比较通信成本，不能将本报告的本机耗时外推为真实 LLM 服务时延。",
        "",
        "## 纯文本协作与结构化协议对比",
        "",
        "| 指标 | 纯文本协作 | TH-ACP 结构化协作 | 改进 |",
        "| --- | ---: | ---: | ---: |",
        f"| 病例数 | {text['case_count']} | {structured['case_count']} | 一致 |",
        f"| 完全匹配病例数 | {text['exact_matches']} | {structured['exact_matches']} | 一致 |",
        f"| Agent 间消息数 | {text['message_count']} | {structured['message_count']} | — |",
        f"| 通信字节 | {text['total_message_bytes']} | {structured['total_message_bytes']} | 节省 {comparison['message_bytes_saved']} ({comparison['message_bytes_saved_percent']:.2f}%) |",
        f"| 通信 Token 估算 | {text['total_message_tokens_estimate']} | {structured['total_message_tokens_estimate']} | 节省 {comparison['communication_tokens_saved']} ({comparison['communication_tokens_saved_percent']:.2f}%) |",
        f"| TH-ACP 完整报文字节 | 不适用 | {structured['total_packet_bytes']} | 含协议头与关联字段 |",
        f"| 每病例本机执行时间 | {text['mean_case_elapsed_ms']:.3f} ms | {structured['mean_case_elapsed_ms']:.3f} ms | 仅供本机参考 |",
        "",
        "纯文本模式把任务、病例、结果和药物说明写入自然语言消息；TH-ACP 模式用 capability、结构化参数、关联 ID 和协议校验路由相同业务步骤。"
        "两者推荐结果一致，通信开销差异因此可归因于通信表示，而非病例或决策质量变化。",
        "",
        "## 非文本状态传递：Doctor → Expert 二进制决策向量",
        "",
    ]
    state = result["non_text_state_transfer"]
    if state.get("status") == "skipped":
        rows.append(f"本次跳过：{state['reason']}。")
    else:
        rows += [
            f"- 覆盖病例：{state['cases']}；实际二进制交接帧：{state['handoff_frames']}；输出一致：{state['outputs_identical']}。",
            f"- JSON 重建帧：{state['json_frames_bytes_reconstructed']} 字节；二进制实测帧：{state['binary_frames_bytes_measured']} 字节；"
            f"节省 {state['saved_bytes']} 字节（{state['saved_percent']:.2f}%）。",
            f"- 原始二进制状态载荷：{state['raw_binary_payload_bytes']} 字节；编码均值 {state['encode_mean_us']:.2f} μs；"
            f"解码均值 {state['decode_mean_us']:.2f} μs。",
        ]
    rows += [
        "",
        "## 说明",
        "",
        "- Token 为稳定的字符规则估算，不等同于任一模型 API 的实际计费 Token。",
        "- 纯文本模式和结构化模式均执行八个业务交接；TH-ACP 的 `packet_count` 统计业务协议报文。",
        "- 二进制状态传递单独在四 Agent 进程 Runtime 中测量，包含 Pipe 帧头，不包含 LLM 推理时间。",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Tiny Hospital 赛题统一通信评测报告")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--skip-process", action="store_true", help="跳过多进程二进制状态传递测量")
    args = parser.parse_args()
    report = benchmark(include_process=not args.skip_process)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "competition_report.json"
    markdown_path = args.output_dir / "competition_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(format_report(report), encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {markdown_path}")


if __name__ == "__main__":
    main()
