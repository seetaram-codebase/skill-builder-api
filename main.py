from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import uuid, os, json

from google.genai import types

import storage, agent as ag
from skills import save_skill_from_output, build_zip, SkillParseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    yield

app = FastAPI(title="Skill Builder API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None   # omit to start a new conversation
    skill_name: Optional[str] = None   # optional name override for generated skill


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    draft_id: Optional[str] = None     # set when agent produces a skill this turn


class DraftResponse(BaseModel):
    draft_id: str
    skill_name: str
    skill_dir: str
    files: dict[str, str]
    created_at: str


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(event: str, data: dict, *, role: str = "assistant", is_last: bool = False) -> str:
    """Format a server-sent event with standard envelope fields.

    Every event includes:
      event      — event type (status, token, draft, done, error)
      role       — who produced it: "assistant" | "system"
      is_last    — true on the final event of a turn (done or error)
      data       — event-specific payload
    """
    payload = {"role": role, "is_last": is_last, **data}
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def extract_reply_text(raw: str) -> str:
    """Strip fenced skill blocks from agent output, leaving only the conversational reply.

    The agent mixes prose ('Here is your skill:') with fenced file blocks
    (```skill.md ... ```). UI should only show the prose as streaming tokens;
    file contents are delivered separately via the draft event.
    """
    return re.sub(r"```\S+\n.*?```", "", raw, flags=re.DOTALL).strip()


# ── Session helper ────────────────────────────────────────────────────────────

async def get_or_create_session(conversation_id: str) -> tuple[str, bool]:
    """Return (adk_session_id, is_new). Creates ADK session if needed."""
    adk_id = storage.get_adk_session_id(conversation_id)
    if adk_id:
        return adk_id, False
    adk_session = await ag.session_service.create_session(
        app_name=ag.APP_NAME,
        user_id=conversation_id,
    )
    storage.save_session(conversation_id, adk_session.id)
    return adk_session.id, True


# ── Primary endpoint: multi-turn chat with SSE streaming ─────────────────────

@app.post("/skills/chat/stream")
async def chat_stream(req: ChatRequest):
    """Multi-turn skill creation with real-time SSE streaming.

    Start a conversation by omitting conversation_id.
    Pass the returned conversation_id to continue the same conversation.

    Events emitted:
      status  {message}                          — phase progress
      token   {text}                             — partial agent reply (live)
      draft   {draft_id, skill_name, files_created} — skill saved this turn
      done    {conversation_id, draft_id}             — turn complete
      error   {detail}                           — failure, stream closes
    """
    async def stream() -> AsyncGenerator[str, None]:
        try:
            sid = req.conversation_id or str(uuid.uuid4())
            adk_conversation_id, is_new = await get_or_create_session(sid)

            yield sse("status", {
                "message": "New conversation started" if is_new else "Continuing conversation",
                "conversation_id": sid,
            }, role="system")

            message = types.Content(
                role="user",
                parts=[types.Part(text=req.message)]
            )

            raw_reply = ""
            async for event in ag.runner.run_async(
                user_id=sid,
                session_id=adk_conversation_id,
                new_message=message,
            ):
                if event.is_final_response() and event.content:
                    for part in event.content.parts:
                        if part.text:
                            raw_reply += part.text

            # Stream only the conversational prose — no fenced file blocks
            reply_text = extract_reply_text(raw_reply)
            if reply_text:
                yield sse("token", {"text": reply_text})

            # Save draft and emit file contents if agent produced a skill this turn
            draft_id = None
            blocks = ag.parse_skill_blocks(raw_reply)
            if blocks and any(f == "skill.md" for f, _ in blocks):
                try:
                    result = save_skill_from_output(raw_reply, req.skill_name)
                    draft_id = result["draft_id"]
                    yield sse("draft", {
                        "draft_id": draft_id,
                        "skill_name": result["skill_name"],
                        "files_created": result["files_created"],
                        "files": result["files"],
                    })
                except SkillParseError as e:
                    yield sse("error", {"detail": str(e)}, role="system", is_last=True)
                    return

            yield sse("done", {"conversation_id": sid, "draft_id": draft_id}, role="system", is_last=True)

        except Exception as e:
            yield sse("error", {"detail": str(e)}, role="system", is_last=True)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Non-streaming chat (simple clients) ──────────────────────────────────────

@app.post("/skills/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Multi-turn skill creation — non-streaming version.

    Use /skills/chat/stream for real-time UI; use this for CLI or simple clients.
    """
    sid = req.conversation_id or str(uuid.uuid4())
    adk_conversation_id, _ = await get_or_create_session(sid)

    message = types.Content(
        role="user",
        parts=[types.Part(text=req.message)]
    )

    reply = ""
    async for event in ag.runner.run_async(
        user_id=sid,
        conversation_id=adk_conversation_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    reply += part.text

    draft_id = None
    blocks = ag.parse_skill_blocks(reply)
    if blocks and any(f == "skill.md" for f, _ in blocks):
        try:
            result = save_skill_from_output(reply, req.skill_name)
            draft_id = result["draft_id"]
        except SkillParseError:
            pass

    return ChatResponse(conversation_id=sid, reply=reply, draft_id=draft_id)


# ── Draft endpoints ───────────────────────────────────────────────────────────

@app.get("/skills", summary="List all skill drafts")
async def list_drafts():
    return {"drafts": storage.list_drafts(), "count": len(storage.list_drafts())}


@app.get("/skills/{draft_id}", response_model=DraftResponse)
async def get_draft(draft_id: str):
    """Get a skill draft with all file contents."""
    draft = storage.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return DraftResponse(
        draft_id=draft["draft_id"],
        skill_name=draft["skill_name"],
        skill_dir=draft["skill_dir"],
        files=draft["files"],
        created_at=draft["created_at"],
    )


@app.get("/skills/{draft_id}/files/{file_path:path}", summary="Get a single file from a draft")
async def get_draft_file(draft_id: str, file_path: str):
    """Get raw content of a single file (e.g. SKILL.md, references/REFERENCE.md)."""
    draft = storage.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    if file_path not in draft["files"]:
        raise HTTPException(status_code=404, detail=f"File '{file_path}' not found in draft")
    return {"file_path": file_path, "content": draft["files"][file_path]}


@app.get("/skills/{draft_id}/download", summary="Download skill as zip")
async def download_draft(draft_id: str):
    """Download the entire skill directory as a zip file."""
    draft = storage.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    if not os.path.isdir(draft["skill_dir"]):
        raise HTTPException(status_code=404, detail="Skill files not found on disk")

    buffer = build_zip(draft["skill_name"], draft["skill_dir"])
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={draft['skill_name']}.zip"},
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}
