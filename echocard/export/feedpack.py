"""AI 投喂包导出：投喂AI.md（反偏见 Prompt + 半结构化协议）+ zip + 解压目录。

- 单日导出：EchoCard_YYYY-MM-DD_投喂包/（内含 投喂AI.md + photos/）
- 多日合并：EchoCard_START_to_END_投喂包/
- 同级同时保留 zip 与解压文件夹；禁止在 Markdown 中嵌入 Base64 图片；
- 图片引用按相对路径 photos/HH-MM-SS_序号.jpg。
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import zipfile
from datetime import date, datetime
from pathlib import Path

from ..core.config import AppConfig
from ..core.db import Database
from ..core.utils import now_iso

log = logging.getLogger(__name__)

DEFAULT_PROMPT = """\
你是一位善于倾听与洞察的个人记忆整理师。下面是我在 {days} 的全天候采集记录，
包含：本地语音转写（带说话人与客观音频标注）、说话人聚类编号、检测到的音乐片段、
以及按时间戳命名的照片。请你为我生成一份深度复盘。

## 一、反偏见纠错声明（最高优先级）
以下文本、说话人标签与音频标注全部由本地模型自动生成，可能存在以下误差：
同音错字、说话人身份标错、把正常对话标为低置信度、把环境外放误录为对话、
把音乐/掌声等事件漏标或误标。请你始终以**语义连贯性、对话逻辑与照片线索**为
最高判断标准；发现标签与语义冲突时，直接按你的判断自行纠正，并在文末
「纠错记录」里简述你修正了哪些标记。**不要因为标注可疑而丢弃内容。**

## 二、说话人称呼说明
- 「说话人 N」为当日无监督聚类中性编号；「疑似-XXX (说话人 N)」表示声纹软匹配
  中等置信度；高置信度匹配会直接显示人名。
- 请根据对话内容推理各编号最可能对应谁（如「我」通常是第一人称最多的人），
  并在复盘里统一使用推理后的称呼。

## 三、照片与音乐上下文分析
- 照片按 `HH-MM-SS_序号.jpg`（拍摄时刻）命名于 photos/ 目录；请结合前后数分钟
  的对话与已识别音乐，推理场景氛围、拍照意图与隐含意义；偶发无意义废图可忽略。
- 音乐节点带精确起止时间；已识别的曲目一并给出；若你能联想到这些歌的背景或
  与当天情绪的呼应，请自由展开。

## 四、输出格式（半结构化，正文保持自由深度）
1. 正文开头请输出一个 YAML Frontmatter 代码块（```yaml 包裹），字段固定为：
   date: {date_value}            # 单日为当天；多日合并为 START_to_END
   tags: [...]                   # 核心关键词 3-8 个
   mood: ...                     # 状态/情绪概览（一两句）
   todos: [...]                  # 提取的待办事项（字符串数组）
   facts: [...]                  # 值得长期沉淀的事实（人物/约定/数字等）
   music: [...]                  # 今天听到的歌曲（歌名 - 歌手）
2. Frontmatter 之后是自由正文：事件时间线、深度复盘、情绪洞察、建议等，
   保持你自己的思考深度，不要僵化。
3. 如果当前环境支持生成下载文件，请将整理结果打包为
   `{zip_name}`（内含 summary.md，可选 meta.json）供我下载；
   若不支持生成文件，请直接输出完整 Markdown 代码块供我复制。
"""


def _fetch_day(db: Database, day: str) -> dict:
    segments = [dict(r) for r in db.query(
        # 按挂钟时间排序（start_offset 是文件内偏移，跨文件不可比）
        "SELECT * FROM audio_segments WHERE day=?"
        " ORDER BY start_ts, start_offset", (day,))]
    music = [dict(r) for r in db.query(
        "SELECT * FROM music_tracks WHERE day=? ORDER BY start_ts", (day,))]
    photos = [dict(r) for r in db.query(
        "SELECT * FROM photos WHERE captured_at=? AND include_in_export=1"
        " ORDER BY captured_ts", (day,))]
    summaries = [dict(r) for r in db.query(
        "SELECT * FROM daily_summaries WHERE day_range=?", (day,))]
    todos = db.query(
        "SELECT t.* FROM todos t JOIN daily_summaries s ON s.id=t.summary_id"
        " WHERE s.day_range=? ORDER BY t.id", (day,))
    return {"segments": segments, "music": music, "photos": photos,
            "summaries": summaries, "todos": [dict(t) for t in todos]}


def _hms(dt_or_str: str) -> str:
    return (dt_or_str or "")[-8:]


def build_markdown(db: Database, cfg: AppConfig, days: list[str]) -> tuple[str, list[str]]:
    """构建投喂AI.md 正文。返回 (md, 复制的照片相对路径列表)。"""
    days = sorted(days)
    start, end = days[0], days[-1]
    if len(days) == 1:
        days_label = start
        date_value = start
        zip_name = f"EchoCard_{start}_投喂包.zip"
    else:
        days_label = f"{start} 至 {end}"
        date_value = f"{start}_to_{end}"
        zip_name = f"EchoCard_{start}_to_{end}_投喂包.zip"

    prompt = (cfg.export_prompt or DEFAULT_PROMPT).format(
        days=days_label, date_value=date_value, zip_name=zip_name)

    lines = [prompt, "\n---\n", f"# EchoCard 采集记录 · {days_label}\n"]
    photo_refs: list[str] = []
    for day in days:
        data = _fetch_day(db, day)
        lines.append(f"\n## {day}\n")

        lines.append("### 语音时间轴（自动转写 · 反偏见保留）\n")
        if not data["segments"]:
            lines.append("（本日无语音段落，或尚未清洗）\n")
        for s in data["segments"]:
            tags = ", ".join(json.loads(s["audio_tags"] or "[]"))
            speaker = (s["speaker_name"] or f"说话人 {s['cluster_id']}") \
                if s["cluster_id"] else "未知说话人"
            line = f"- **[{_hms(s['start_ts'])} - {_hms(s['end_ts'])}] {speaker}**：{s['text']}"
            if tags:
                line += f"\n  - （客观音频标注：{tags}；置信度 {s['speaker_match_score'] or '-'}）"
            if s["starred"]:
                line += "  ⭐"
            lines.append(line)

        lines.append("\n### 音乐节点\n")
        if not data["music"]:
            lines.append("（本日未检测到音乐片段）\n")
        for m in data["music"]:
            if m["status"] == "recognized" and m["title"]:
                lines.append(
                    f"- **[{_hms(m['start_ts'])} - {_hms(m['end_ts'])}]** "
                    f"🎵 已识别：《{m['title']}》 - {m['artist'] or '未知歌手'}"
                    f"（专辑：{m['album'] or '未知'}）")
            else:
                lines.append(f"- **[{_hms(m['start_ts'])} - {_hms(m['end_ts'])}]** 🎧 检测到音乐片段（未识别）")

        lines.append("\n### 照片（按拍摄时刻）\n")
        if not data["photos"]:
            lines.append("（本日无照片）\n")
        for p in data["photos"]:
            ref = f"photos/{Path(p['compressed_path']).name}"
            photo_refs.append(ref)
            ts = _hms(p["captured_ts"] or "")[:5]
            lines.append(f"- **{ts}** `{ref}`" + (" ⭐" if p["bookmark"] else ""))

        if data["summaries"]:
            lines.append("\n### 已归档的 AI 总结\n")
            for su in data["summaries"]:
                lines.append(f"> {su['content_md'][:2000]}\n")
    return "\n".join(lines), photo_refs


def export_days(db: Database, cfg: AppConfig, days: list[str]) -> dict:
    """导出单日/多日合并投喂包：zip + 解压目录。返回路径信息。"""
    days = sorted(set(days))
    if not days:
        raise ValueError("未选择任何日期")
    start, end = days[0], days[-1]
    if len(days) == 1:
        name = f"EchoCard_{start}_投喂包"
    else:
        name = f"EchoCard_{start}_to_{end}_投喂包"

    export_dir = cfg.subdir("exports")
    folder = export_dir / name
    if folder.exists():
        shutil.rmtree(folder)  # 覆盖式导出（导出目录属于可再生内容）
    (folder / "photos").mkdir(parents=True)

    md, photo_refs = build_markdown(db, cfg, days)
    md_path = folder / "投喂AI.md"
    md_path.write_text(md, encoding="utf-8")

    copied = []
    for day in days:
        for row in db.query(
            "SELECT compressed_path, captured_ts FROM photos WHERE captured_at=?"
            " AND include_in_export=1", (day,)
        ):
            src = Path(row["compressed_path"])
            if src.exists():
                dst = folder / "photos" / src.name
                shutil.copy2(src, dst)
                copied.append(src.name)

    zip_path = export_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, f"{name}/投喂AI.md")
        for rel in photo_refs:
            p = folder / rel
            if p.exists():
                zf.write(p, f"{name}/{rel}")

    log.info("投喂包导出完成：%s（照片 %d 张）", name, len(copied))
    return {"folder": str(folder), "zip": str(zip_path),
            "markdown": str(md_path), "photos": len(copied),
            "exported_at": now_iso()}
