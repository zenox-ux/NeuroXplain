import builtins
import linecache
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class TraceConfig:
    enabled: bool = False
    include_root: Optional[str] = None
    exclude_substrings: Optional[tuple[str, ...]] = None
    max_steps: int = 0  # 0 == unlimited
    print_locals: bool = True
    max_value_len: int = 300


def _safe_repr(value: Any, max_len: int) -> str:
    try:
        s = repr(value)
    except Exception as e:  # pragma: no cover
        s = f"<unreprable {type(value).__name__}: {e}>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_locals_diff(prev: Dict[str, str], curr: Dict[str, str]) -> str:
    added = {k: v for k, v in curr.items() if k not in prev}
    changed = {k: v for k, v in curr.items() if k in prev and prev[k] != v}
    removed = sorted([k for k in prev.keys() if k not in curr])

    parts: list[str] = []
    if added:
        parts.append("added=" + ", ".join(f"{k}={v}" for k, v in sorted(added.items())))
    if changed:
        parts.append(
            "changed=" + ", ".join(f"{k}={v}" for k, v in sorted(changed.items()))
        )
    if removed:
        parts.append("removed=" + ", ".join(removed))
    return " | ".join(parts)


def enable_line_tracing(config: TraceConfig) -> Callable[[], None]:
    """Enable a per-line tracer for Python code.

    Prints each executed line (file:line + source) and a locals diff.

    This is meant for debugging / dry-run understanding, not for normal training/eval.
    """

    if not config.enabled:
        return lambda: None

    include_root = None
    if config.include_root:
        include_root = os.path.abspath(config.include_root)

    exclude = config.exclude_substrings or ()

    state = {
        "steps": 0,
        # frame_id -> last seen info for that frame
        # {
        #   "last_lineno": int,
        #   "last_filename": str,
        #   "last_locals": {name: repr},
        # }
        "frames": {},
    }

    def should_trace(filename: str) -> bool:
        if not filename:
            return False
        abs_path = os.path.abspath(filename)
        if include_root and not abs_path.startswith(include_root):
            return False
        for needle in exclude:
            if needle and needle in abs_path:
                return False
        return True

    def _locals_snapshot(frame) -> Dict[str, str]:
        snap: Dict[str, str] = {}
        for k, v in frame.f_locals.items():
            snap[k] = _safe_repr(v, config.max_value_len)
        return snap

    def _emit_executed_line(frame, frame_state: Dict[str, Any]):
        """Emit the last executed line for this frame + locals diff."""
        last_filename = frame_state.get("last_filename")
        last_lineno = frame_state.get("last_lineno")
        last_locals = frame_state.get("last_locals", {})

        if not last_filename or not last_lineno:
            return

        # We may have entered a new file that isn't traced; honor filtering.
        if not should_trace(last_filename):
            return

        src = (linecache.getline(last_filename, last_lineno) or "").rstrip("\n")
        src_print = src.strip() if src.strip() else "<blank>"

        builtins.print(f"[TRACE] {last_filename}:{last_lineno} | {src_print}")

        if config.print_locals:
            curr = _locals_snapshot(frame)
            diff = _format_locals_diff(last_locals, curr)
            if diff:
                builtins.print(f"[TRACE] locals: {diff}")
            frame_state["last_locals"] = curr

    def trace_func(frame, event, arg):
        filename = frame.f_code.co_filename

        # Always return the tracer so we can catch nested calls,
        # but only emit for files we care about.
        if event == "call":
            if should_trace(filename):
                frame_id = id(frame)
                state["frames"][frame_id] = {
                    "last_filename": filename,
                    "last_lineno": frame.f_lineno,
                    "last_locals": _locals_snapshot(frame) if config.print_locals else {},
                }
            return trace_func

        if event in ("line", "return"):
            if not should_trace(filename):
                return trace_func

            state["steps"] += 1
            if config.max_steps and state["steps"] > config.max_steps:
                sys.settrace(None)
                builtins.print(
                    f"[TRACE] stopped after {config.max_steps} steps (max_steps)"
                )
                return None

            frame_id = id(frame)
            frame_state = state["frames"].get(frame_id)
            if frame_state is None:
                frame_state = {
                    "last_filename": filename,
                    "last_lineno": frame.f_lineno,
                    "last_locals": _locals_snapshot(frame) if config.print_locals else {},
                }
                state["frames"][frame_id] = frame_state

            # At a new 'line' event we have just finished executing the previous line.
            _emit_executed_line(frame, frame_state)

            # Record the next line that will run.
            if event == "line":
                frame_state["last_filename"] = filename
                frame_state["last_lineno"] = frame.f_lineno
                if config.print_locals:
                    frame_state["last_locals"] = _locals_snapshot(frame)
            elif event == "return":
                # Emit return value too (truncated) for clarity.
                builtins.print(
                    f"[TRACE] return from {filename}:{frame.f_lineno} -> {_safe_repr(arg, config.max_value_len)}"
                )
                state["frames"].pop(frame_id, None)

        return trace_func

    sys.settrace(trace_func)

    def disable():
        sys.settrace(None)

    return disable
