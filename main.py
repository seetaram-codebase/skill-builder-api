from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid, os, re

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types

app = FastAPI(title="ADK Skill Creator", version="1.0.0")

# SkillToolset loads the skill-creator SKILL.md with progressive disclosure:
# L1 (~100 tokens): name + description loaded at startup
# L2 (<5000 tokens): full instructions loaded when skill is activated
# L3 (as needed): reference files loaded on demand
skill_toolset = SkillToolset(skills_dir="./skills/skill-creator")

agent = LlmAgent(
    name="skill_creator_agent",
    model="gemini-2.5-pro",
    instruction="""You create complete, production-ready skills from user intent.

Use the skill-creator skill to generate all necessary files.

Output only the files that are genuinely needed, using named fences:

```skill.md              → SKILL.md with valid frontmatter (name + description required)
```references/REFERENCE.md  → detailed reference docs (if the skill needs lookup material)
```scripts/run.py           → helper script (if the skill needs executable code)
```assets/template.md       → templates or static resources (if the skill needs them)

Rules:
- skill name must be lowercase, hyphens only, no spaces
- folder name must match the name field in frontmatter
- only include files that genuinely add value""",
    tools=[skill_toolset],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="skill-creator-api",
    session_service=session_service,
)

SKILLS_OUTPUT_DIR = "./generated-skills"
os.makedirs(SKILLS_OUTPUT_DIR, exist_ok=True)


class CreateSkillRequest(BaseModel):
    intent: str          # e.g. "review pull requests for security vulnerabilities"
    name: Optional[str] = None  # optional slug override, e.g. "pr-security-review"


class CreateSkillResponse(BaseModel):
    draft_id: str
    skill_name: str
    skill_path: str
    files_created: list[str]


class GetSkillResponse(BaseModel):
    draft_id: str
    skill_name: str
    skill_path: str
    files: dict[str, str]  # relative path -> content


# In-memory draft registry: draft_id -> {skill_name, skill_path, files}
draft_registry: dict[str, dict] = {}


@app.post("/skills", response_model=CreateSkillResponse)
async def create_skill(req: CreateSkillRequest):
    """Generate a complete skill from natural language intent."""
    session = await session_service.create_session(
        app_name="skill-creator-api",
        user_id="skill-creator",
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"Create a complete skill for: {req.intent}")]
    )

    raw_output = ""
    async for event in runner.run_async(
        user_id="skill-creator",
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    raw_output += part.text

    # Parse all named fenced blocks: ```filename\ncontent```
    blocks = re.findall(r"```(\S+)\n(.*?)```", raw_output, re.DOTALL)
    if not blocks:
        raise HTTPException(status_code=500, detail="Agent did not produce any skill files")

    skill_md_content = next((c for f, c in blocks if f == "skill.md"), None)
    if not skill_md_content:
        raise HTTPException(status_code=500, detail="Agent did not produce SKILL.md")

    # Extract skill name from frontmatter
    name_match = re.search(r"^name:\s*(.+)$", skill_md_content, re.MULTILINE)
    skill_name = req.name or (name_match.group(1).strip() if name_match else str(uuid.uuid4())[:8])

    # Write all files to disk
    skill_dir = os.path.join(SKILLS_OUTPUT_DIR, skill_name)
    files_created = []

    for fname, content in blocks:
        # Remap skill.md -> SKILL.md
        target = "SKILL.md" if fname == "skill.md" else fname
        file_path = os.path.join(skill_dir, target)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content.strip())
        files_created.append(target)

    draft_id = str(uuid.uuid4())
    draft_registry[draft_id] = {
        "skill_name": skill_name,
        "skill_path": skill_dir,
        "files": {f: open(os.path.join(skill_dir, f)).read() for f in files_created},
    }

    return CreateSkillResponse(
        draft_id=draft_id,
        skill_name=skill_name,
        skill_path=skill_dir,
        files_created=files_created,
    )


@app.get("/skills/{draft_id}", response_model=GetSkillResponse)
async def get_skill(draft_id: str):
    """Retrieve a generated skill by draft ID."""
    if draft_id not in draft_registry:
        raise HTTPException(status_code=404, detail="Draft not found")
    entry = draft_registry[draft_id]
    return GetSkillResponse(
        draft_id=draft_id,
        skill_name=entry["skill_name"],
        skill_path=entry["skill_path"],
        files=entry["files"],
    )


@app.get("/skills/{draft_id}/files/{file_path:path}")
async def get_skill_file(draft_id: str, file_path: str):
    """Get the raw content of a single file within a skill draft.

    Examples:
      GET /skills/{draft_id}/files/SKILL.md
      GET /skills/{draft_id}/files/references/REFERENCE.md
      GET /skills/{draft_id}/files/scripts/run.py
      GET /skills/{draft_id}/files/assets/template.md
    """
    if draft_id not in draft_registry:
        raise HTTPException(status_code=404, detail="Draft not found")

    skill_dir = draft_registry[draft_id]["skill_path"]
    full_path = os.path.normpath(os.path.join(skill_dir, file_path))

    # Prevent path traversal
    if not full_path.startswith(os.path.abspath(skill_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"File '{file_path}' not found in skill")

    with open(full_path) as f:
        content = f.read()

    return {"file_path": file_path, "content": content}


@app.get("/skills")
async def list_skills():
    """List all generated skill drafts."""
    return {
        "drafts": [
            {"draft_id": did, "skill_name": v["skill_name"], "skill_path": v["skill_path"]}
            for did, v in draft_registry.items()
        ],
        "count": len(draft_registry),
    }
