#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import copy
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

DEFAULT_WHAM_URL = "https://chatgpt.com/backend-api/wham/usage"
DEFAULT_TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_USER_AGENT = "codex_cli_rs/0.101.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464"
DEFAULT_VERSION = "0.101.0"
LOW_THRESHOLD_PERCENT = 20.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def b64url_decode(value: str) -> bytes:
    value = value.strip()
    padding = (-len(value)) % 4
    value += "=" * padding
    return base64.urlsafe_b64decode(value.encode("utf-8"))


def parse_jwt_payload(token: str | None) -> dict[str, Any]:
    if not token or not isinstance(token, str):
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        return json.loads(b64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return {}


def first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        elif value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def clean_plan(value: Any) -> str:
    plan = first_non_empty(value).lower().replace("_", "-")
    return plan



def infer_provider_from_base_url(value: Any) -> str:
    text = first_non_empty(value).lower()
    if "codex" in text:
        return "codex"
    return ""



def payload_email(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    email = first_non_empty(payload.get("email"))
    if email:
        return email
    profile_claim = payload.get("https://api.openai.com/profile")
    if isinstance(profile_claim, dict):
        return first_non_empty(profile_claim.get("email"))
    return ""


def nested_get(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def provider_hint_to_provider(hint: Any) -> str:
    text = clean_plan(hint)
    if text in {"codex", "openai-codex", "openaicodex"}:
        return "codex"
    if "codex" in text:
        return "codex"
    if text == "":
        return ""
    return text



def discover_entries(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def remember(node: dict[str, Any], provider_hint: str = "") -> None:
        candidate = dict(node)
        if provider_hint and "__provider_hint" not in candidate:
            candidate["__provider_hint"] = provider_hint
        found.append(candidate)

    def walk(node: Any, provider_hint: str = "") -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, provider_hint)
            return
        if not isinstance(node, dict):
            return

        if "credential_pool" in node and isinstance(node["credential_pool"], dict):
            for pool_provider, pool_entries in node["credential_pool"].items():
                walk(pool_entries, provider_hint_to_provider(pool_provider))

        candidate_provider = provider_hint_to_provider(node.get("provider") or node.get("type") or provider_hint)
        candidate = candidate_provider == "codex"
        if not candidate:
            tokenish = any(k in node for k in ("access_token", "refresh_token", "id_token", "auth_index"))
            if tokenish and any(k in node for k in ("email", "account_id", "name", "id", "label")):
                candidate = candidate_provider in {"", "codex"}
        if not candidate and "base_url" in node:
            candidate = "codex" in first_non_empty(node.get("base_url")).lower()
        if candidate:
            remember(node, candidate_provider or provider_hint)

        for key in ("files", "auths", "entries", "accounts", "items", "data"):
            child = node.get(key)
            if isinstance(child, (list, dict)):
                walk(child, provider_hint)

        for value in node.values():
            if isinstance(value, (list, dict)):
                walk(value, provider_hint)

    walk(obj)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in found:
        provider = provider_hint_to_provider(entry.get("provider") or entry.get("type") or entry.get("__provider_hint"))
        if provider and provider != "codex":
            continue
        key = (
            first_non_empty(entry.get("id")),
            first_non_empty(entry.get("label")),
            first_non_empty(entry.get("access_token"))[:24],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def load_auth_json(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        entries = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict):
        direct_provider = clean_plan(raw.get("provider") or raw.get("type"))
        if direct_provider == "codex" or any(k in raw for k in ("access_token", "refresh_token", "id_token")):
            entries = [raw]
        else:
            entries = discover_entries(raw)
    else:
        raise ValueError("Unsupported auth.json format")
    return raw, entries


def extract_auth_info(entry: dict[str, Any], index: int) -> dict[str, Any]:
    id_token = first_non_empty(
        entry.get("id_token"),
        nested_get(entry, "metadata", "id_token"),
        nested_get(entry, "attributes", "id_token"),
    )
    access_token = first_non_empty(
        entry.get("access_token"),
        nested_get(entry, "metadata", "access_token"),
        nested_get(entry, "attributes", "access_token"),
    )
    refresh_token = first_non_empty(
        entry.get("refresh_token"),
        nested_get(entry, "metadata", "refresh_token"),
        nested_get(entry, "attributes", "refresh_token"),
    )
    id_payload = parse_jwt_payload(id_token)
    access_payload = parse_jwt_payload(access_token)

    def auth_claim(payload: dict[str, Any]) -> dict[str, Any]:
        value = payload.get("https://api.openai.com/auth") if isinstance(payload, dict) else {}
        return value if isinstance(value, dict) else {}

    id_auth = auth_claim(id_payload)
    access_auth = auth_claim(access_payload)

    account_id = first_non_empty(
        entry.get("account_id"),
        nested_get(entry, "metadata", "account_id"),
        id_payload.get("chatgpt_account_id") if isinstance(id_payload, dict) else None,
        id_auth.get("chatgpt_account_id"),
        access_payload.get("chatgpt_account_id") if isinstance(access_payload, dict) else None,
        access_auth.get("chatgpt_account_id"),
    )
    plan_type = clean_plan(
        entry.get("plan_type")
        or nested_get(entry, "metadata", "plan_type")
        or id_payload.get("chatgpt_plan_type")
        or id_auth.get("chatgpt_plan_type")
        or access_payload.get("chatgpt_plan_type")
        or access_auth.get("chatgpt_plan_type")
    )
    profile_claim = id_payload.get("https://api.openai.com/profile") if isinstance(id_payload, dict) else {}
    if not isinstance(profile_claim, dict):
        profile_claim = {}
    access_profile_claim = access_payload.get("https://api.openai.com/profile") if isinstance(access_payload, dict) else {}
    if not isinstance(access_profile_claim, dict):
        access_profile_claim = {}
    email = first_non_empty(
        entry.get("email"),
        nested_get(entry, "metadata", "email"),
        payload_email(id_payload),
        profile_claim.get("email"),
        payload_email(access_payload),
        access_profile_claim.get("email"),
        entry.get("label"),
    )
    name = first_non_empty(entry.get("name"), entry.get("id"), entry.get("label"), email, f"codex-{index}")
    provider = provider_hint_to_provider(entry.get("provider") or entry.get("type") or entry.get("__provider_hint") or infer_provider_from_base_url(entry.get("base_url")))
    auth_index = first_non_empty(entry.get("auth_index"), entry.get("authIndex"), str(index))
    return {
        "entry": entry,
        "provider": provider,
        "name": name,
        "email": email,
        "account_id": account_id,
        "plan_type": plan_type or "unknown",
        "auth_index": auth_index,
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "jwt_payload": id_payload or access_payload,
    }


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def token_expiry(info: dict[str, Any]) -> datetime | None:
    explicit = parse_datetime(info["entry"].get("expired") or info["entry"].get("expires_at") or nested_get(info["entry"], "metadata", "expired"))
    if explicit:
        return explicit
    payload = parse_jwt_payload(info.get("access_token"))
    exp = payload.get("exp") if isinstance(payload, dict) else None
    if isinstance(exp, (int, float)):
        try:
            return datetime.fromtimestamp(float(exp), tz=timezone.utc)
        except Exception:
            return None
    return None


def should_refresh(info: dict[str, Any], force_refresh: bool) -> bool:
    if force_refresh:
        return True
    if not info.get("refresh_token"):
        return False
    if not info.get("access_token"):
        return True
    expires_at = token_expiry(info)
    if expires_at is None:
        return False
    return expires_at <= utc_now() + timedelta(minutes=5)


def http_json(url: str, method: str = "GET", headers: dict[str, str] | None = None, body: bytes | None = None, timeout: float = 30.0) -> tuple[int, dict[str, Any], dict[str, str]]:
    req = request.Request(url=url, method=method, data=body, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw.strip() else {}
            return resp.status, payload, dict(resp.headers.items())
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except Exception:
            payload = {"raw": raw}
        return exc.code, payload, dict(exc.headers.items())


def refresh_tokens(info: dict[str, Any], token_url: str, timeout: float, client_id: str) -> tuple[bool, str | None]:
    refresh_token = info.get("refresh_token")
    if not refresh_token:
        return False, "missing refresh_token"
    form = parse.urlencode(
        {
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "openid profile email",
        }
    ).encode("utf-8")
    status, payload, _ = http_json(
        token_url,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        body=form,
        timeout=timeout,
    )
    if status < 200 or status >= 300:
        return False, f"refresh failed HTTP {status}: {payload}"
    info["access_token"] = first_non_empty(payload.get("access_token"), info.get("access_token"))
    info["refresh_token"] = first_non_empty(payload.get("refresh_token"), info.get("refresh_token"))
    info["id_token"] = first_non_empty(payload.get("id_token"), info.get("id_token"))
    id_payload = parse_jwt_payload(info.get("id_token"))
    access_payload = parse_jwt_payload(info.get("access_token"))

    def auth_claim(data: dict[str, Any]) -> dict[str, Any]:
        value = data.get("https://api.openai.com/auth") if isinstance(data, dict) else {}
        return value if isinstance(value, dict) else {}

    id_auth = auth_claim(id_payload)
    access_auth = auth_claim(access_payload)
    info["jwt_payload"] = id_payload or access_payload
    info["account_id"] = first_non_empty(
        info.get("account_id"),
        id_auth.get("chatgpt_account_id"),
        id_payload.get("chatgpt_account_id") if isinstance(id_payload, dict) else None,
        access_auth.get("chatgpt_account_id"),
        access_payload.get("chatgpt_account_id") if isinstance(access_payload, dict) else None,
    )
    if info.get("plan_type") in ("", "unknown"):
        info["plan_type"] = clean_plan(
            id_auth.get("chatgpt_plan_type")
            or id_payload.get("chatgpt_plan_type")
            or access_auth.get("chatgpt_plan_type")
            or access_payload.get("chatgpt_plan_type")
        ) or "unknown"
    if not info.get("email"):
        info["email"] = first_non_empty(payload_email(id_payload), payload_email(access_payload), info["entry"].get("label"))
    return True, None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def reset_label(window: dict[str, Any]) -> str:
    reset_at = number(window.get("reset_at") or window.get("resetAt"))
    if reset_at and reset_at > 0:
        return datetime.fromtimestamp(reset_at, tz=timezone.utc).astimezone().strftime("%m-%d %H:%M")
    after = number(window.get("reset_after_seconds") or window.get("resetAfterSeconds"))
    if after and after > 0:
        return (datetime.now().astimezone() + timedelta(seconds=after)).strftime("%m-%d %H:%M")
    return "-"


def deduce_used_percent(window: dict[str, Any], limit_reached: Any, allowed: Any) -> float | None:
    used = number(window.get("used_percent") or window.get("usedPercent"))
    if used is not None:
        return clamp(used, 0.0, 100.0)
    if boolish(limit_reached) or (allowed is not None and not boolish(allowed)):
        if reset_label(window) != "-":
            return 100.0
    return None


def build_window(window_id: str, label: str, window: dict[str, Any] | None, limit_reached: Any, allowed: Any) -> dict[str, Any] | None:
    if not isinstance(window, dict):
        return None
    used = deduce_used_percent(window, limit_reached, allowed)
    remaining = None if used is None else clamp(100.0 - used, 0.0, 100.0)
    return {
        "id": window_id,
        "label": label,
        "used_percent": used,
        "remaining_percent": remaining,
        "reset_label": reset_label(window),
        "exhausted": used is not None and used >= 100.0,
    }


def find_codex_windows(rate_limit: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    primary = rate_limit.get("primary_window") or rate_limit.get("primaryWindow")
    secondary = rate_limit.get("secondary_window") or rate_limit.get("secondaryWindow")
    five_hour = None
    weekly = None
    for candidate in (primary, secondary):
        if not isinstance(candidate, dict):
            continue
        duration = number(candidate.get("limit_window_seconds") or candidate.get("limitWindowSeconds"))
        if duration == 5 * 60 * 60 and five_hour is None:
            five_hour = candidate
        if duration == 7 * 24 * 60 * 60 and weekly is None:
            weekly = candidate
    if five_hour is None and isinstance(primary, dict):
        five_hour = primary
    if weekly is None and isinstance(secondary, dict):
        weekly = secondary
    return five_hour, weekly


def parse_wham_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rate_limit = payload.get("rate_limit") or payload.get("rateLimit") or {}
    if not isinstance(rate_limit, dict):
        rate_limit = {}
    limit_reached = rate_limit.get("limit_reached") or rate_limit.get("limitReached")
    allowed = rate_limit.get("allowed")
    windows: list[dict[str, Any]] = []
    additional: list[dict[str, Any]] = []
    five_hour, weekly = find_codex_windows(rate_limit)
    for item in (
        build_window("code-5h", "5h", five_hour, limit_reached, allowed),
        build_window("code-7d", "7d", weekly, limit_reached, allowed),
    ):
        if item:
            windows.append(item)

    extra_limits = payload.get("additional_rate_limits") or payload.get("additionalRateLimits") or []
    if isinstance(extra_limits, list):
        for idx, entry in enumerate(extra_limits, start=1):
            if not isinstance(entry, dict):
                continue
            name = first_non_empty(
                entry.get("limit_name"),
                entry.get("limitName"),
                entry.get("metered_feature"),
                entry.get("meteredFeature"),
                f"additional-{idx}",
            )
            rl = entry.get("rate_limit") or entry.get("rateLimit") or {}
            if not isinstance(rl, dict):
                continue
            primary = rl.get("primary_window") or rl.get("primaryWindow")
            secondary = rl.get("secondary_window") or rl.get("secondaryWindow")
            for item in (
                build_window(f"{name}-primary", f"{name} 5h", primary, rl.get("limit_reached") or rl.get("limitReached"), rl.get("allowed")),
                build_window(f"{name}-secondary", f"{name} 7d", secondary, rl.get("limit_reached") or rl.get("limitReached"), rl.get("allowed")),
            ):
                if item:
                    additional.append(item)
    return windows, additional


def classify_status(windows: list[dict[str, Any]], error_text: str = "") -> str:
    if error_text:
        return "error"
    exhausted = any(window.get("exhausted") for window in windows)
    if exhausted:
        return "exhausted"
    remaining_values = [window.get("remaining_percent") for window in windows if window.get("remaining_percent") is not None]
    if remaining_values and min(remaining_values) < LOW_THRESHOLD_PERCENT:
        return "low"
    return "ok" if windows else "unknown"


def query_quota(info: dict[str, Any], wham_url: str, timeout: float, user_agent: str, version: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, int | None]:
    headers = {
        "Authorization": f"Bearer {info['access_token']}",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
        "Chatgpt-Account-Id": info["account_id"],
        "Version": version,
        "Session_id": str(uuid.uuid4()),
        "Originator": "codex_cli_rs",
        "Accept": "application/json",
    }
    status, payload, _ = http_json(wham_url, method="GET", headers=headers, timeout=timeout)
    if 200 <= status < 300:
        windows, additional = parse_wham_payload(payload)
        return windows, additional, None, status
    return [], [], f"quota request failed HTTP {status}: {payload}", status


def maybe_write_back(path: Path, raw_root: Any, infos: list[dict[str, Any]]) -> None:
    for info in infos:
        entry = info["entry"]
        if info.get("access_token"):
            entry["access_token"] = info["access_token"]
        if info.get("refresh_token"):
            entry["refresh_token"] = info["refresh_token"]
        if info.get("id_token"):
            entry["id_token"] = info["id_token"]
        if info.get("account_id"):
            entry["account_id"] = info["account_id"]
        if info.get("email") and not entry.get("email"):
            entry["email"] = info["email"]
        if info.get("plan_type") and info.get("plan_type") != "unknown":
            entry["plan_type"] = info["plan_type"]
    path.write_text(json.dumps(raw_root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_one(info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "provider": "codex",
        "name": info["name"],
        "email": info.get("email", ""),
        "auth_index": info.get("auth_index", ""),
        "account_id": info.get("account_id", ""),
        "plan_type": info.get("plan_type") or "unknown",
        "status": "unknown",
        "windows": [],
        "additional_windows": [],
        "error": "",
        "refreshed": False,
    }

    if not info.get("account_id"):
        report["error"] = "missing chatgpt_account_id"
        report["status"] = "error"
        return report
    if not info.get("access_token") and not info.get("refresh_token"):
        report["error"] = "missing access_token and refresh_token"
        report["status"] = "error"
        return report

    if should_refresh(info, args.force_refresh):
        ok, err_text = refresh_tokens(info, args.token_url, args.timeout, args.client_id)
        report["refreshed"] = ok
        if not ok:
            report["error"] = err_text or "refresh failed"
            report["status"] = "error"
            return report

    windows, additional, err_text, status_code = query_quota(info, args.wham_url, args.timeout, args.user_agent, args.version)
    if err_text and status_code == 401 and info.get("refresh_token"):
        ok, refresh_err = refresh_tokens(info, args.token_url, args.timeout, args.client_id)
        report["refreshed"] = report["refreshed"] or ok
        if ok:
            windows, additional, err_text, status_code = query_quota(info, args.wham_url, args.timeout, args.user_agent, args.version)
        else:
            err_text = refresh_err or err_text

    report["windows"] = windows
    report["additional_windows"] = additional
    report["error"] = err_text or ""
    report["status"] = classify_status(windows + additional, report["error"])
    return report


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "accounts": len(reports),
        "status_counts": {},
        "plan_counts": {},
        "exhausted_accounts": 0,
        "low_accounts": 0,
        "error_accounts": 0,
    }
    for report in reports:
        status = report.get("status") or "unknown"
        plan = report.get("plan_type") or "unknown"
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        summary["plan_counts"][plan] = summary["plan_counts"].get(plan, 0) + 1
        if status == "exhausted":
            summary["exhausted_accounts"] += 1
        elif status == "low":
            summary["low_accounts"] += 1
        elif status == "error":
            summary["error_accounts"] += 1
    return summary


def window_by_id(report: dict[str, Any], window_id: str) -> dict[str, Any] | None:
    for window in report.get("windows", []):
        if window.get("id") == window_id:
            return window
    return None


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"



def ansi(text: str, *codes: str) -> str:
    active = [code for code in codes if code]
    if not active:
        return text
    return "".join(active) + text + ANSI_RESET



def visible_len(text: str) -> int:
    length = 0
    in_escape = False
    for ch in text:
        if ch == "\033":
            in_escape = True
            continue
        if in_escape:
            if ch == "m":
                in_escape = False
            continue
        length += 1
    return length



def pad_ansi(text: str, width: int) -> str:
    if width <= 0:
        return text
    missing = width - visible_len(text)
    if missing > 0:
        return text + (" " * missing)
    return text



def truncate_text(text: str, width: int) -> str:
    if width <= 0:
        return text
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"



def quota_color(value: float | None) -> str:
    if value is None:
        return ANSI_DIM
    if value > 50:
        return ANSI_GREEN
    if value >= 20:
        return ANSI_YELLOW
    return ANSI_RED



def status_color(status: str) -> str:
    normalized = (status or "").lower()
    if normalized == "ok":
        return ANSI_GREEN
    if normalized == "low":
        return ANSI_YELLOW
    if normalized in {"exhausted", "error"}:
        return ANSI_RED
    return ANSI_DIM



def render_bar(value: float | None, width: int = 10, ascii_mode: bool = False) -> str:
    if value is None:
        return ansi("[??????????]", ANSI_DIM)
    value = clamp(value, 0.0, 100.0)
    filled = int(round((value / 100.0) * width))
    filled = max(0, min(width, filled))
    if ascii_mode:
        bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    else:
        bar = "[" + ("█" * filled) + ("░" * (width - filled)) + "]"
    return ansi(bar, quota_color(value))



def fmt_remaining(window: dict[str, Any] | None) -> str:
    if not window:
        return "-"
    value = window.get("remaining_percent")
    if value is None:
        return "-"
    return f"{value:.0f}%"



def fmt_bar(window: dict[str, Any] | None, ascii_mode: bool = False) -> str:
    if not window:
        return "[??????????]"
    return render_bar(window.get("remaining_percent"), width=10, ascii_mode=ascii_mode)



def fmt_reset(window: dict[str, Any] | None) -> str:
    if not window:
        return "-"
    return first_non_empty(window.get("reset_label"), "-")



def status_icon(status: str) -> str:
    normalized = (status or "").lower()
    if normalized == "ok":
        return "[OK]"
    if normalized == "low":
        return "[WARN]"
    if normalized == "exhausted":
        return "[BAD]"
    if normalized == "error":
        return "[ERR]"
    return "[?]"



def print_pretty_report(reports: list[dict[str, Any]], summary: dict[str, Any], ascii_mode: bool = False) -> None:
    print("Codex quota inspector")
    print()

    total = summary.get("accounts", 0)
    ok = summary.get("status_counts", {}).get("ok", 0)
    low = summary.get("low_accounts", 0)
    exhausted = summary.get("exhausted_accounts", 0)
    errors = summary.get("error_accounts", 0)

    if ok:
        print(f"{status_icon('ok')} {ok} account(s) OK")
    if low:
        print(f"{status_icon('low')} {low} account(s) LOW")
    if exhausted:
        print(f"{status_icon('exhausted')} {exhausted} account(s) EXHAUSTED")
    if errors:
        print(f"{status_icon('error')} {errors} account(s) ERROR")
    if total == 0:
        print("No accounts found")
        return

    print(f"Plan summary: {', '.join(f'{k}={v}' for k, v in sorted(summary.get('plan_counts', {}).items())) or '-'}")
    print()

    for idx, report in enumerate(reports, start=1):
        email_or_name = first_non_empty(report.get('email'), report.get('name'), f'account-{idx}')
        print(f"{idx}) {status_icon(report.get('status', 'unknown'))} {email_or_name}")
        print(f"   plan: {report.get('plan_type', 'unknown')}")
        w5 = window_by_id(report, 'code-5h')
        w7 = window_by_id(report, 'code-7d')
        print(f"   5h  {fmt_bar(w5, ascii_mode=ascii_mode)} {fmt_remaining(w5)}")
        print(f"   7d  {fmt_bar(w7, ascii_mode=ascii_mode)} {fmt_remaining(w7)}")
        print(f"   reset: {fmt_reset(w5)} / {fmt_reset(w7)}")
        for extra in report.get('additional_windows', []):
            print(f"   extra {extra.get('label', 'window')}: {fmt_bar(extra, ascii_mode=ascii_mode)} {fmt_remaining(extra)} reset {fmt_reset(extra)}")
        if report.get('error'):
            print(f"   error: {report['error']}")
        if idx != len(reports):
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect remaining Codex quotas from auth.json")
    parser.add_argument("auth_path", nargs="?", default="auth.json", help="Path to auth.json (default: ./auth.json)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Print JSON output")
    parser.add_argument("--ascii", action="store_true", help="ASCII fallback for borders and bars")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel workers")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh token before querying quota")
    parser.add_argument("--write-back", action="store_true", help="Persist refreshed tokens back into auth.json")
    parser.add_argument("--wham-url", default=DEFAULT_WHAM_URL, help="Quota endpoint override")
    parser.add_argument("--token-url", default=DEFAULT_TOKEN_URL, help="Refresh endpoint override")
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID, help="OAuth client_id for token refresh")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent for wham/usage request")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Version header for wham/usage request")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = Path(args.auth_path).expanduser().resolve()
    if not auth_path.exists():
        print(f"auth file not found: {auth_path}", file=sys.stderr)
        return 2

    try:
        raw_root, raw_entries = load_auth_json(auth_path)
    except Exception as exc:
        print(f"failed to read auth file: {exc}", file=sys.stderr)
        return 2

    infos = [extract_auth_info(entry, idx + 1) for idx, entry in enumerate(raw_entries)]
    infos = [info for info in infos if info.get("provider") == "codex"]
    if not infos:
        result = {"auth_path": str(auth_path), "reports": [], "summary": summarize([])}
        if args.json_output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("No codex entries found.")
        return 0

    indexed_infos = list(infos)
    reports: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(inspect_one, info, args) for info in indexed_infos]
        for future in concurrent.futures.as_completed(futures):
            reports.append(future.result())

    reports.sort(key=lambda item: (item.get("status") != "error", item.get("name", "")))
    summary = summarize(reports)

    if args.write_back:
        maybe_write_back(auth_path, raw_root, indexed_infos)

    result = {
        "auth_path": str(auth_path),
        "generated_at": utc_now().isoformat(),
        "reports": reports,
        "summary": summary,
    }
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_pretty_report(reports, summary, ascii_mode=args.ascii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
