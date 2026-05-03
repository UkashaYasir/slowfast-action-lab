"""Local chat history and Gemini-backed response helpers."""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from google import genai


logger = logging.getLogger("ChatService")

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
CHAT_HISTORY_FILE = Path(os.getenv("CHAT_HISTORY_FILE", "data/chat_history.json"))


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    """Small local JSON store for chat sessions."""

    def __init__(self, path: Path | str = CHAT_HISTORY_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"sessions": []})

    def _read(self) -> dict:
        if not self.path.exists():
            return {"sessions": []}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, payload: dict) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)

    def list_sessions(self) -> list[dict]:
        payload = self._read()
        sessions = payload.get("sessions", [])
        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return [
            {
                "id": session["id"],
                "title": session.get("title", "New chat"),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "message_count": len(session.get("messages", [])),
                "last_preview": next(
                    (
                        message.get("content", "")
                        for message in reversed(session.get("messages", []))
                        if message.get("role") == "assistant"
                    ),
                    "",
                )[:120],
            }
            for session in sessions
        ]

    def get_session(self, session_id: str) -> Optional[dict]:
        payload = self._read()
        for session in payload.get("sessions", []):
            if session["id"] == session_id:
                return session
        return None

    def delete_session(self, session_id: str) -> bool:
        payload = self._read()
        sessions = payload.get("sessions", [])
        remaining = [session for session in sessions if session["id"] != session_id]
        if len(remaining) == len(sessions):
            return False
        payload["sessions"] = remaining
        self._write(payload)
        return True

    def rename_session(self, session_id: str, title: str) -> Optional[dict]:
        payload = self._read()
        cleaned_title = (title or "").strip()[:80]
        if not cleaned_title:
            return None
        for session in payload.get("sessions", []):
            if session["id"] == session_id:
                session["title"] = cleaned_title
                session["updated_at"] = utc_now()
                self._write(payload)
                return session
        return None

    def ensure_session(self, session_id: Optional[str], title_hint: str = "") -> dict:
        payload = self._read()
        sessions = payload.get("sessions", [])
        if session_id:
            for session in sessions:
                if session["id"] == session_id:
                    return session

        created_at = utc_now()
        title = (title_hint or "New conversation").strip()[:60] or "New conversation"
        session = {
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": created_at,
            "updated_at": created_at,
            "messages": [],
        }
        sessions.append(session)
        self._write(payload)
        return session

    def append_messages(self, session_id: str, messages: list[dict]) -> dict:
        payload = self._read()
        for session in payload.get("sessions", []):
            if session["id"] == session_id:
                session_messages = session.setdefault("messages", [])
                session_messages.extend(messages)
                session["updated_at"] = utc_now()
                if session.get("title", "").startswith("New conversation"):
                    first_user = next(
                        (
                            item.get("content", "").strip()
                            for item in session_messages
                            if item.get("role") == "user" and item.get("content", "").strip()
                        ),
                        "",
                    )
                    if first_user:
                        session["title"] = first_user[:60]
                self._write(payload)
                return session
        raise KeyError(f"Session not found: {session_id}")


def gemini_chat_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def generate_chat_response(
    user_message: str,
    history: Optional[list[dict]] = None,
    video_context: Optional[dict] = None,
    recent_analyses: Optional[list[dict]] = None,
) -> str:
    """Generate a local assistant reply using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return (
            "Gemini is not configured yet. Open Settings in the app and save a backend "
            "API key to enable chat replies."
        )

    history = history or []
    cleaned_message = (user_message or "").strip()
    if not cleaned_message and not video_context:
        return "Please send a message or attach a video."

    prompt_parts = [
        "You are NeuralVision AI, a concise assistant inside a local video intelligence app.",
        "Answer clearly and helpfully. Keep replies practical and grounded.",
    ]

    if history:
        prompt_parts.append("Recent conversation:")
        for message in history[-8:]:
            role = message.get("role", "user").upper()
            content = (message.get("content", "") or "").strip()
            if content:
                prompt_parts.append(f"{role}: {content}")

    if video_context:
        prediction = video_context.get("top_prediction", "unknown")
        confidence = float(video_context.get("top_confidence") or 0)
        prompt_parts.append(
            "Attached video context: "
            f"predicted action={prediction}, confidence={confidence:.1f}%, "
            f"raw prediction={video_context.get('raw_top_prediction', prediction)}, "
            f"uncertain={video_context.get('is_uncertain', False)}."
        )
        top_predictions = video_context.get("predictions") or []
        if top_predictions:
            top_lines = ", ".join(
                f"{item.get('label', 'unknown')} ({float(item.get('confidence') or 0):.1f}%)"
                for item in top_predictions[:5]
            )
            prompt_parts.append(f"Top predictions: {top_lines}.")

    if recent_analyses:
        prompt_parts.append("Recent analyses from the workspace:")
        for item in recent_analyses[:5]:
            prompt_parts.append(
                "- "
                f"{item.get('filename', 'clip')}: prediction={item.get('top_prediction')}, "
                f"raw={item.get('raw_top_prediction')}, "
                f"confidence={float(item.get('top_confidence') or 0):.1f}%, "
                f"uncertain={item.get('is_uncertain', False)}"
            )

    if cleaned_message:
        prompt_parts.append(f"User request: {cleaned_message}")
    else:
        prompt_parts.append("User request: Explain the attached video analysis.")

    prompt_parts.append(
        "If the video prediction is uncertain, say so plainly and avoid overclaiming."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents="\n".join(prompt_parts),
        )
        text = (response.text or "").strip()
        return text or "Gemini returned an empty response."
    except Exception as exc:
        logger.warning("Gemini chat failed: %s", exc)
        if video_context and video_context.get("is_uncertain"):
            return (
                "The last video result is uncertain. The model saw readable frames, "
                "but confidence or class separation was too weak to trust the prediction."
            )
        if recent_analyses and cleaned_message:
            latest = recent_analyses[0]
            return (
                "Gemini is unavailable right now. The most recent analysis was "
                f"{latest.get('top_prediction', 'unknown')} at "
                f"{float(latest.get('top_confidence') or 0):.1f}% confidence."
            )
        return "Gemini could not generate a response right now."


def export_session_markdown(session: dict) -> str:
    lines = [f"# {session.get('title', 'Chat Session')}", ""]
    for message in session.get("messages", []):
        role = "NeuralVision" if message.get("role") == "assistant" else "You"
        lines.append(f"## {role}")
        lines.append(message.get("content", ""))
        attachment = message.get("attachment")
        if attachment:
            lines.append(
                f"- Attachment: {attachment.get('filename')} ({attachment.get('size_mb')} MB)"
            )
        video_context = message.get("video_context")
        if video_context:
            lines.append(
                f"- Video result: {video_context.get('top_prediction')} "
                f"({float(video_context.get('top_confidence') or 0):.1f}%)"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
