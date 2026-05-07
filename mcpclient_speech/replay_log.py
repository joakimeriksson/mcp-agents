"""Render an interaction-log JSONL file as a human-readable transcript.

Usage:
    python replay_log.py path/to/session.jsonl
    cat session.jsonl | python replay_log.py -
    python replay_log.py session.jsonl --show-state --show-messages

By default state_change events and the full pre/post-truncation message lists
inside llm_request events are suppressed for readability. Toggle with the
--show-state and --show-messages flags.
"""

import argparse
import json
import sys
from datetime import datetime


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_elapsed(start: datetime | None, ts: datetime | None) -> str:
    if start is None or ts is None:
        return "        "
    secs = (ts - start).total_seconds()
    return f"{secs:7.2f}s"


def _truncate(text: str | None, limit: int) -> str:
    if text is None:
        return ""
    text = text.replace("\n", " \\n ")
    if limit > 0 and len(text) > limit:
        return text[:limit - 1] + "…"
    return text


def _render(event: dict, start: datetime | None, args) -> list[str]:
    """Return zero or more output lines for a single event."""
    et = event.get("type", "?")
    ts = _parse_ts(event.get("ts", ""))
    pre = _fmt_elapsed(start, ts)

    def line(body: str) -> str:
        return f"{pre}  {body}"

    limit = args.text_limit

    if et == "session_start":
        a = event.get("args", {})
        return [
            line(f"=== SESSION START  pid={event.get('pid')} commit={event.get('git_commit')}"),
            line(f"    model={a.get('llm_model')}  lang={a.get('lang_default')}  server={a.get('server')}"),
        ]

    if et == "session_end":
        reason = event.get("reason")
        dur = event.get("duration_seconds")
        out = [line(f"=== SESSION END    reason={reason}  duration={dur}s")]
        if event.get("error_class"):
            out.append(line(f"    ERROR {event['error_class']}: {event.get('error_message','')}"))
        if event.get("traceback") and args.show_traceback:
            for tline in event["traceback"].rstrip().split("\n"):
                out.append(line(f"    {tline}"))
        return out

    if et == "face_change":
        return [line(f"[face]   id={event.get('face_id')}  status={event.get('status')}  "
                     f"name={event.get('name')}  lang={event.get('lang')}")]

    if et == "person_resumed":
        return [line(f"[person] resumed id={event.get('face_id')} name={event.get('name')} "
                     f"lang={event.get('lang')} restored_msgs={event.get('restored_message_count')}")]

    if et == "person_distillation":
        return [line(f"[distill] id={event.get('face_id')} "
                     f"name={event.get('parsed_name')} lang={event.get('parsed_lang')} "
                     f"prefs={_truncate(event.get('parsed_preferences'), 60)}")]

    if et == "transcription":
        return [line(f"[STT]    ({event.get('language')} {event.get('language_probability', 0):.2f}) "
                     f"{_truncate(event.get('text'), limit)}")]

    if et == "stdin_command":
        return [line(f"[stdin]  {event.get('cmd')}  {_truncate(event.get('arg'), limit)}")]

    if et == "user_turn":
        return [line(f"USER ({event.get('kind')}/{event.get('lang')}): "
                     f"{_truncate(event.get('content'), limit)}")]

    if et == "mcp_prompt":
        return [line(f"[mcp]    {event.get('kind')} prompt (lang={event.get('lang')}): "
                     f"{_truncate(event.get('content'), limit)}")]

    if et == "llm_request":
        out = [line(f"[llm]    -> request iter={event.get('iteration')} "
                    f"tools={event.get('tool_count')} model={event.get('model')}")]
        if args.show_messages:
            for m in event.get("messages_sent", []):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "?")
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                out.append(line(f"           {role}: {_truncate(content, limit)}"))
        return out

    if et == "llm_response":
        out = [line(f"[llm]    <- response iter={event.get('iteration')} "
                    f"content={_truncate(event.get('content'), limit)}")]
        for tc in event.get("tool_calls") or []:
            out.append(line(f"           tool_call {tc.get('name')}({_truncate(tc.get('arguments'), limit)})"))
        return out

    if et == "llm_error":
        return [line(f"[llm]    ERROR iter={event.get('iteration')} "
                     f"{event.get('error_class')}: {event.get('error_message')}")]

    if et == "mcp_tool_call":
        if event.get("success"):
            return [line(f"[tool]   {event.get('name')}({_truncate(event.get('arguments'), limit)}) "
                         f"-> {_truncate(event.get('result_text'), limit)}")]
        return [line(f"[tool]   {event.get('name')}({_truncate(event.get('arguments'), limit)}) "
                     f"-> ERROR {event.get('error_message')}")]

    if et == "tts_speak":
        return [line(f"BOT  ({event.get('status')}/{event.get('lang')}): "
                     f"{_truncate(event.get('text'), limit)}")]

    if et == "mute":
        return [line(f"[mute]   muted={event.get('muted')} interrupted={event.get('interrupted_speech')}")]

    if et == "state_change":
        if not args.show_state:
            return []
        return [line(f"[state]  {event.get('from')} -> {event.get('to')}")]

    if et == "recording_saved":
        return [line(f"[rec]    {event.get('wav_path')}  remaining={event.get('remaining')}")]

    if et == "clear_history":
        return [line("[clear]  history cleared")]

    return [line(f"[{et}]  {json.dumps({k: v for k, v in event.items() if k not in ('ts', 'type')}, ensure_ascii=False)}")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an interaction-log JSONL file as a transcript.")
    parser.add_argument("path", help="Path to .jsonl log file, or '-' for stdin.")
    parser.add_argument("--types", default=None,
                        help="Comma-separated list of event types to include (default: all).")
    parser.add_argument("--show-state", action="store_true",
                        help="Include state_change events (suppressed by default).")
    parser.add_argument("--show-messages", action="store_true",
                        help="Dump the messages_sent list inside each llm_request event.")
    parser.add_argument("--show-traceback", action="store_true",
                        help="Print the full traceback string from session_end on error.")
    parser.add_argument("--text-limit", type=int, default=300,
                        help="Truncate text fields longer than this (0 = no truncation; default: 300).")
    args = parser.parse_args()

    type_filter = set(args.types.split(",")) if args.types else None
    src = sys.stdin if args.path == "-" else open(args.path, "r", encoding="utf-8")
    start_ts: datetime | None = None
    try:
        for raw in src:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"!! skipping malformed line: {e}", file=sys.stderr)
                continue
            if start_ts is None and event.get("type") == "session_start":
                start_ts = _parse_ts(event.get("ts", ""))
            elif start_ts is None:
                start_ts = _parse_ts(event.get("ts", ""))
            if type_filter and event.get("type") not in type_filter:
                continue
            for output_line in _render(event, start_ts, args):
                print(output_line)
    finally:
        if src is not sys.stdin:
            src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
