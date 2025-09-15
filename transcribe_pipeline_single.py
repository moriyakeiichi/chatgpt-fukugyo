#!/usr/bin/env python3
"""
Single-file transcription pipeline (Mac/Windows/Linux)
-----------------------------------------------------------------
統合版スクリプト（修正版：Part.from_bytes の互換呼び出し対応、ffmpeg -y 追加）
使い方は変わらず：

python transcribe_pipeline_single.py input.mp3 --work-dir work

依存：
- Python 3.9+
- ffmpeg (推奨。なければ pydub にフォールバック)
- pip install google-genai
- 環境変数 GEMINI_API_KEY
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import dataclasses
import json
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# --- ロギング設定 -----------------------------------------------------------
LOG = logging.getLogger("pipeline")

# --- ユーティリティ ---------------------------------------------------------
TS_RE = re.compile(r"^\s*(\d+):(\d{2})(?::(\d{2}))?\b")


def hhmmss_to_ms(s: str) -> int:
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h = 0
        m, sec = parts
    else:
        raise ValueError(f"Bad time '{s}' (use m:ss or h:mm:ss)")
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000


def ms_to_hhmmss(ms: int) -> str:
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def ms_to_fname(ms: int) -> str:
    # Windows安全な名前（: を - に）
    hms = ms_to_hhmmss(ms)
    return hms.replace(":", "-")


def find_on_path(cmd: str) -> Optional[str]:
    path = shutil.which(cmd)
    return path


def run(cmd: List[str]) -> None:
    LOG.debug("$ %s", " ".join(cmd))
    subprocess.check_call(cmd)


def get_duration_ms_ffprobe(path: pathlib.Path) -> Optional[int]:
    ffprobe = find_on_path("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.check_output([
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path)
        ], text=True)
        sec = float(out.strip())
        return int(round(sec * 1000))
    except Exception:
        return None


def get_duration_ms_fallback(path: pathlib.Path) -> int:
    # pydub fallback（ffprobeが無い環境用）
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "音声長を取得できませんでした。ffprobe か pydub を用意してください"
        ) from e
    seg = AudioSegment.from_file(path)
    return len(seg)


def get_duration_ms(path: pathlib.Path) -> int:
    return get_duration_ms_ffprobe(path) or get_duration_ms_fallback(path)


def ffmpeg_trim_copy(src: pathlib.Path, dst: pathlib.Path, start_ms: int, end_ms: int) -> None:
    ffmpeg = find_on_path("ffmpeg")
    if not ffmpeg:
        # pydub 再エンコードにフォールバック
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(src)
        clip = seg[start_ms:end_ms]
        dst.parent.mkdir(parents=True, exist_ok=True)
        clip.export(dst, format=dst.suffix.lstrip("."))
        return
    dur = max(0, end_ms - start_ms)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -y を入れて既存ファイル上書きを自動化（プロンプト回避）
    run([
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_ms/1000:.3f}", "-t", f"{dur/1000:.3f}",
        "-i", str(src),
        "-c", "copy",
        str(dst)
    ])


# --- データ構造 -------------------------------------------------------------
@dataclasses.dataclass
class Chunk:
    index: int
    start_ms: int
    end_ms: int
    path: pathlib.Path


# --- 分割（人間指定の境界で） ---------------------------------------------
def compute_chunks_by_boundaries(audio_ms: int, boundaries_ms: List[int]) -> List[Tuple[int, int]]:
    """0 と audio_ms を含めて、[start, end) の区間リストを返す。"""
    uniq = sorted(set([0] + [b for b in boundaries_ms if 0 < b < audio_ms] + [audio_ms]))
    return list(zip(uniq[:-1], uniq[1:]))


def compute_chunks_from_csv(rows: List[dict], audio_ms: int) -> List[Tuple[int, int]]:
    spans = []
    for r in rows:
        s = hhmmss_to_ms(r["start"]) if r.get("start") else 0
        e = hhmmss_to_ms(r["end"]) if r.get("end") else audio_ms
        s = max(0, min(s, audio_ms))
        e = max(0, min(e, audio_ms))
        if e > s:
            spans.append((s, e))
    try:
        rows_sorted = sorted(zip(rows, spans), key=lambda x: int(x[0].get("index", 0)))
        spans = [sp for _, sp in rows_sorted]
    except Exception:
        spans = sorted(spans)
    return spans


def split_audio(input_mp3: pathlib.Path, out_dir: pathlib.Path, spans: List[Tuple[int, int]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, (s, e) in enumerate(spans, start=1):
        name = f"{input_mp3.stem}_{i:02d}_{ms_to_fname(s)}_{ms_to_fname(e)}{input_mp3.suffix}"
        dst = out_dir / name
        ffmpeg_trim_copy(input_mp3, dst, s, e)
        chunks.append(Chunk(index=i, start_ms=s, end_ms=e, path=dst))
        LOG.info("[split] #%02d %s -> %s", i, f"{ms_to_hhmmss(s)}-{ms_to_hhmmss(e)}", dst.name)
    (out_dir / "manifest.json").write_text(
        json.dumps([dataclasses.asdict(c) | {"path": str(c.path)} for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks


# --- Gemini 呼び出し（互換対応ラッパ） -----------------------------------------
class GeminiClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY が未設定です。環境変数に設定してください。")
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError("'google-genai' がインストールされていません。'pip install google-genai' を実行してください") from e
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def transcribe(self, mp3_path: pathlib.Path, system_prompt: str, timeout: int = 180) -> str:
        """
        音声ファイルを受け取って、タイムスタンプ＋話者ラベル付きテキストを返す（互換呼び出し）。
        google-genai のバージョン差に備えて、Part.from_bytes の呼び方をキーワードで試行します。
        """
        b = mp3_path.read_bytes()
        parts = []
        tried = []

        Part = getattr(self._genai.types, "Part", None)
        if Part is None:
            raise RuntimeError("google.genai.types.Part が見つかりません。ライブラリのバージョンを確認してください。")

        # audio part: data=... を使って呼び出す（mime_type あり／なしを試す）
        try:
            audio_part = Part.from_bytes(data=b, mime_type="audio/mpeg")
            parts.append(audio_part)
            tried.append("from_bytes(data=..., mime_type=...) ok")
        except TypeError as te:
            tried.append(f"from_bytes with mime_type TypeError: {te}")
            try:
                audio_part = Part.from_bytes(data=b)
                parts.append(audio_part)
                tried.append("from_bytes(data=...) ok")
            except Exception as e2:
                tried.append(f"from_bytes(data=b) failed: {type(e2).__name__}:{e2}")
                # ここはフォールバックを試すために続行
        except Exception as e:
            tried.append(f"from_bytes(data=..., mime_type=...) other err: {type(e).__name__}:{e}")

        # text part (system prompt)
        try:
            if hasattr(Part, "from_text"):
                text_part = Part.from_text(system_prompt)
                parts.append(text_part)
            else:
                # 古い実装などで無ければ plain text をそのまま渡す（generate_content の実装次第）
                parts.append(system_prompt)
        except Exception as e:
            tried.append(f"Part.from_text failed: {type(e).__name__}:{e}")
            parts.append(system_prompt)

        # ここまでで parts が空なら audio API のフォールバックを試す
        if parts:
            delay = 2.0
            for attempt in range(6):
                try:
                    resp = self.client.models.generate_content(model=self.model, contents=parts)
                    text = getattr(resp, "text", None)
                    if not text and hasattr(resp, "candidates") and resp.candidates:
                        try:
                            text = resp.candidates[0].content.parts[0].text
                        except Exception:
                            text = None
                    if text:
                        return text
                    # ループ最終回で取得できなければ例外で補足する
                    if attempt >= 5:
                        raise RuntimeError("Geminiの応答からテキストを取得できませんでした")
                except Exception as e:
                    tried.append(f"generate_content attempt{attempt}:{type(e).__name__}:{e}")
                    if attempt >= 5:
                        break
                    LOG.warning("[gemini] retry %d: %s (tried=%s)", attempt + 1, e, tried)
                    time.sleep(delay)
                    delay = min(delay * 2.0, 30.0)

        # フォールバック: client.audio.* のような API があれば試す
        try:
            audio_api = getattr(self.client, "audio", None)
            if audio_api is not None:
                if hasattr(audio_api, "transcribe"):
                    resp = audio_api.transcribe(file=str(mp3_path), model=self.model)
                    text = getattr(resp, "text", None) or str(resp)
                    return text
                if hasattr(audio_api, "recognize"):
                    resp = audio_api.recognize(file=str(mp3_path), model=self.model)
                    text = getattr(resp, "text", None) or str(resp)
                    return text
        except Exception as e:
            tried.append(f"audio api fallback failed: {type(e).__name__}:{e}")

        # ここまで到達したら失敗
        raise RuntimeError(
            "google-genai 呼び出しの互換性エラー。試行ログ: "
            + json.dumps(tried[:20], ensure_ascii=False)
            + "\n対処案: python -m pip install --upgrade google-genai を試すか、スクリプトを環境に合わせて調整してください。"
        )

    def clean_transcript(self, raw_text: str, system_prompt: str) -> str:
        parts = [self._genai.types.Part.from_text(system_prompt + "\n\n" + raw_text)] if hasattr(self._genai.types.Part, "from_text") else [system_prompt + "\n\n" + raw_text]
        resp = self.client.models.generate_content(model=self.model, contents=parts)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = None
        if not text:
            raise RuntimeError("Geminiの応答からテキストを取得できませんでした")
        return text


TRANSCRIBE_PROMPT = (
    "以下の音声を日本語で厳密に文字起こししてください。"
    "各行は先頭に mm:ss または h:mm:ss のタイムスタンプ、続いて **話者1:**, **話者2:** のように話者ラベルを付け、"
    "内容をそのまま書き起こしてください。語尾の伸ばしや相槌もできるだけ保持。改行は発話の自然な切れ目ごとに。"
)

CLEAN_PROMPT = (
    "以下のトランスクリプトを読みやすく整形してください。要求：\n"
    "- ケバ取り（『えー』『あのー』『はい』などの冗長な相槌・言い直しは削る）\n"
    "- 同一話者の連続発話は適度に結合\n"
    "- 行頭のタイムスタンプ(mm:ss / h:mm:ss)は維持\n"
    "- 固有名詞の表記ゆれを整える（不明はそのまま）\n"
    "- 出力はプレーンテキストのみ\n"
)


# --- 文字起こしステップ ---------------------------------------------------
def transcribe_chunks(chunks: List[Chunk], out_dir: pathlib.Path, client: GeminiClient, max_workers: int = 2) -> List[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txts: List[pathlib.Path] = []

    def _do(chunk: Chunk) -> pathlib.Path:
        txt_path = out_dir / (chunk.path.stem + ".txt")
        if txt_path.exists():
            LOG.info("[asr] skip exists: %s", txt_path.name)
            return txt_path
        text = client.transcribe(chunk.path, TRANSCRIBE_PROMPT)
        txt_path.write_text(text, encoding="utf-8")
        LOG.info("[asr] wrote %s", txt_path.name)
        return txt_path

    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p in ex.map(_do, chunks):
            out_txts.append(p)
    return sorted(out_txts)


# --- タイムスタンプシフト ---------------------------------------------------
def parse_ts_seconds(line: str) -> Optional[int]:
    m = TS_RE.match(line)
    if not m:
        return None
    a, b, c = m.groups()
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def shift_file_timestamps(txt_path: pathlib.Path, offset_sec: int) -> pathlib.Path:
    out = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TS_RE.match(ln)
        if not m:
            out.append(ln)
            continue
        total = parse_ts_seconds(ln)
        if total is None:
            out.append(ln)
            continue
        total += offset_sec
        h = total // 3600
        m2 = (total % 3600) // 60
        s2 = total % 60
        new_ts = f"{h:02d}:{m2:02d}:{s2:02d}" if h > 0 else f"{m2:02d}:{s2:02d}"
        ln2 = TS_RE.sub(new_ts, ln, count=1)
        out.append(ln2)
    shifted = txt_path.with_name(txt_path.stem + "_shifted.txt")
    shifted.write_text("\n".join(out), encoding="utf-8")
    return shifted


def shift_timestamps_all(chunks: List[Chunk], txt_files: List[pathlib.Path]) -> List[pathlib.Path]:
    """
    txt_files: transcribe_chunks が出したテキスト群（パスのリスト）
    chunks: チャンク情報（path, start_ms ...）
    -> 各チャンクに対応する txt を見つけてシフト版を作る。
    """
    # stem -> Path マップを作る（例: "#1_01_00-00-00_00-04-38" -> work/texts/#1_01_00-00-00_00-04-38.txt）
    by_stem = {p.stem: p for p in txt_files}
    shifted_paths: List[pathlib.Path] = []
    for c in chunks:
        base_stem = c.path.stem  # チャンクファイル名の stem
        p = by_stem.get(base_stem)
        # 互換性のため従来の場所もチェック
        if p is None:
            candidate = c.path.parent / (base_stem + ".txt")
            if candidate.exists():
                p = candidate
        if p is None or not p.exists():
            LOG.warning("[shift] missing txt for %s (expected stems: %s)", c.path.name, base_stem)
            continue
        off = c.start_ms // 1000
        shifted = shift_file_timestamps(p, off)
        shifted_paths.append(shifted)
        LOG.info("[shift] %s +%ds -> %s", p.name, off, shifted.name)
    return sorted(shifted_paths)


# --- 連結 -------------------------------------------------------------------
def concat_texts(txt_files: List[pathlib.Path], out_file: pathlib.Path) -> pathlib.Path:
    out_lines: List[str] = []
    for i, p in enumerate(sorted(txt_files)):
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if i > 0:
            out_lines.append("")
        out_lines.extend(lines)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(out_lines), encoding="utf-8")
    LOG.info("[concat] wrote %s (%d lines)", out_file, len(out_lines))
    return out_file


# --- 仕上げ整形 -------------------------------------------------------------
def clean_with_gemini(raw_path: pathlib.Path, out_path: pathlib.Path, client: GeminiClient) -> pathlib.Path:
    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    # 互換ラッパに差し替え：GeminiClient の clean_transcript を使うのではなくここで安全に呼ぶ
    try:
        # まず types.Part.from_text(text=...) を試す
        Part = getattr(client._genai.types, "Part", None)
        if Part is not None and hasattr(Part, "from_text"):
            try:
                parts = [Part.from_text(text=CLEAN_PROMPT + "\n\n" + raw)]
            except TypeError:
                # 旧インターフェースなら位置引数で
                parts = [Part.from_text(CLEAN_PROMPT + "\n\n" + raw)]
        else:
            # ライブラリ側に Part.from_text が無ければ plain text を渡す
            parts = [CLEAN_PROMPT + "\n\n" + raw]

        resp = client.client.models.generate_content(model=client.model, contents=parts)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = None
        if not text:
            raise RuntimeError("Geminiの応答からテキストを取得できませんでした")
        out_path.write_text(text, encoding="utf-8")
        LOG.info("[clean] wrote %s", out_path)
        return out_path
    except Exception as e:
        LOG.exception("clean_with_gemini failed")
        raise



# --- メイン -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Single-file transcription pipeline")
    ap.add_argument("input", type=pathlib.Path, help="入力MP3/音声ファイル")
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("work"))
    ap.add_argument("--splits", nargs="*", help="0:12:34 のような境界時刻（空ならCSV/自動を使用）")
    ap.add_argument("--splits-file", type=pathlib.Path, help="start,end を含むCSV（ヘッダ: index,start,end, ...）")
    ap.add_argument("--csv-has-end", action="store_true", help="CSVにend列がある（start/end区間で切る）")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--max-workers", type=int, default=2, help="同時文字起こし数（APIレートに合わせて調整）")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-shift", action="store_true")
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    input_mp3 = args.input
    work = args.work_dir
    chunks_dir = work / "chunks"
    txt_dir = work / "texts"
    combined_raw = work / "combined_raw.txt"
    combined_clean = work / "combined_clean.txt"

    work.mkdir(parents=True, exist_ok=True)

    # 1) 分割
    if not args.skip_split:
        audio_ms = get_duration_ms(input_mp3)
        LOG.info("[info] duration = %s", ms_to_hhmmss(audio_ms))

        spans: List[Tuple[int, int]]
        if args.splits_file:
            import csv
            rows = list(csv.DictReader(args.splits_file.open("r", encoding="utf-8")))
            spans = compute_chunks_from_csv(rows, audio_ms)
        else:
            boundaries_ms: List[int] = []
            for t in (args.splits or []):
                boundaries_ms.append(hhmmss_to_ms(t))
            spans = compute_chunks_by_boundaries(audio_ms, boundaries_ms)

        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_audio(input_mp3, chunks_dir, spans)
    else:
        manifest_p = chunks_dir / "manifest.json"
        if not manifest_p.exists():
            LOG.error("--skip-split ですが %s が見つかりません。", manifest_p)
            sys.exit(2)
        data = json.loads(manifest_p.read_text(encoding="utf-8"))
        chunks = [Chunk(index=d["index"], start_ms=d["start_ms"], end_ms=d["end_ms"], path=pathlib.Path(d["path"])) for d in data]

    # 2) 文字起こし
    if not args.skip_asr:
        client = GeminiClient(model=args.model)
        # デバッグのため最初は max_workers=1 を推奨
        txts = transcribe_chunks(chunks, txt_dir, client, max_workers=args.max_workers)
    else:
        txts = sorted(txt_dir.glob("*.txt"))

    # 3) タイムスタンプを絶対時間にシフト
    if not args.skip_shift:
        shifted = shift_timestamps_all(chunks, txts)
    else:
        shifted = sorted(txt_dir.glob("*_shifted.txt"))
        if not shifted:
            shifted = txts

    # 4) 連結
    concat_texts(shifted, combined_raw)

    # 5) 仕上げ整形
    if not args.skip_clean:
        client = client if 'client' in locals() else GeminiClient(model=args.model)
        clean_with_gemini(combined_raw, combined_clean, client if isinstance(client, GeminiClient) else GeminiClient(args.model))

    LOG.info("DONE. raw=%s clean=%s", combined_raw, combined_clean)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.error("Interrupted by user")
        sys.exit(130)
