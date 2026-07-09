"""
Centralized prompt templates.

Keeping prompts in one file makes prompt-engineering iteration easy to
review and keeps the service layer focused on orchestration rather than
string-building. Each prompt explicitly restates the JSON schema expected,
since Gemini's JSON mode guarantees *valid JSON* but not necessarily
adherence to *our specific* schema - being explicit reduces retries.
"""

ROADMAP_SYSTEM_INSTRUCTION = """\
You are an expert technical career mentor and curriculum designer. You design
realistic, well-sequenced learning roadmaps for software engineering goals,
tailored precisely to what an individual learner already knows.
Output rules: respond with a single raw JSON object and nothing else. No
markdown code fences, no leading/trailing commentary, no trailing commas.
"""


def build_roadmap_prompt(
    goal_title: str,
    experience: str,
    known_skills: list[str],
    learning_style: str,
    weekly_hours: int,
) -> str:
    skills_str = ", ".join(known_skills) if known_skills else "None specified"
    return f"""\
Create a personalized learning roadmap for the following learner.

Goal: {goal_title}
Experience level: {experience}
Known skills: {skills_str}
Preferred learning style: {learning_style}
Hours available per week: {weekly_hours}

Requirements:
- Do NOT include any of the learner's known skills ({skills_str}) as a
  top-level "skills" entry or as a task title - assume those are mastered.
  You may still reference a known skill inside a subtask if it's a genuine
  prerequisite refresher for a new topic.
- Every entry in "skills" must be distinct in substance - no synonyms or
  near-duplicates (e.g. do not list both "REST APIs" and "RESTful API
  design").
- Sequence tasks from foundational to advanced; each task should build on
  the previous one rather than covering isolated, unrelated ground.
- Each task must have a realistic estimated_hours given the weekly_hours
  budget and total scope, and 2-5 concrete, actionable subtasks (verbs, not
  vague nouns - e.g. "Build a JWT middleware" not "Authentication").
- Include 4-10 tasks total, and 4-8 top-level skills to be learned.
- estimated_hours (top level) must equal the sum of task estimated_hours
  (roughly; small rounding is fine).

Return a single JSON object with EXACTLY this shape:
{{
  "estimated_hours": <int>,
  "skills": [<string>, ...],
  "tasks": [
    {{
      "title": <string>,
      "estimated_hours": <int>,
      "subtasks": [ {{ "title": <string> }}, ... ]
    }}
  ]
}}
"""


PROJECT_SYSTEM_INSTRUCTION = """\
You are an expert software engineering mentor who recommends portfolio
projects that reinforce a learner's current skills while stretching them
into 1-2 new areas. You favor concrete, buildable-in-days projects over
vague or overly ambitious ideas.
Output rules: respond with a single raw JSON object and nothing else. No
markdown code fences, no leading/trailing commentary, no trailing commas.
"""


def build_project_prompt(goal_title: str, skills: list[str], roadmap_context: str | None) -> str:
    skills_str = ", ".join(skills) if skills else "Not specified"
    context_block = f"\nRoadmap context:\n{roadmap_context}\n" if roadmap_context else ""
    return f"""\
Recommend ONE project idea for a learner targeting: {goal_title}
Relevant skills: {skills_str}
{context_block}
Requirements:
- The project must be practically buildable using the listed skills plus at
  most 1-2 modest stretch technologies - do not require skills the learner
  hasn't been given.
- difficulty must be exactly one of: "Beginner", "Intermediate", "Advanced",
  calibrated to the listed skills, not to the goal title alone.
- Include 3-6 tech_stack items (only ones actually needed) and 3-6 concrete,
  demo-able features (not generic buzzwords).
- why_this_project should be 1-2 sentences tying the project directly to a
  specific skill gap or reinforcement need, not a generic platitude.

Return a single JSON object with EXACTLY this shape:
{{
  "title": <string>,
  "difficulty": <"Beginner" | "Intermediate" | "Advanced">,
  "estimated_hours": <int>,
  "tech_stack": [<string>, ...],
  "features": [<string>, ...],
  "why_this_project": <string>
}}
"""


CHAT_SYSTEM_INSTRUCTION = """\
You are an AI learning assistant helping a learner navigate a personalized
roadmap you previously generated for them. Ground your answer primarily in
the provided roadmap context and conversation history - if the question
goes slightly beyond what the context covers, answer using general,
reasonable engineering knowledge while staying consistent with the
roadmap's plan and sequencing (never contradict it).
Be concise, encouraging, and concrete.
Output rules: respond with a single raw JSON object and nothing else. No
markdown code fences, no leading/trailing commentary, no trailing commas.
"""


def build_chat_prompt(
    roadmap_context: str,
    conversation_history: str,
    user_message: str,
) -> str:
    return f"""\
Roadmap context (retrieved, most relevant labeled chunks of the learner's
roadmap - each is tagged with its source so you can refer to it by name,
e.g. "as covered in Task 2"):
---
{roadmap_context}
---

Recent conversation history:
---
{conversation_history if conversation_history else "(no prior messages)"}
---

Learner's new message: "{user_message}"

Requirements:
- Directly answer the learner's message, referencing specific tasks from the
  roadmap context by name where relevant, rather than speaking generically.
- If the retrieved context doesn't contain enough to answer, say so briefly
  before falling back to general knowledge - don't fabricate roadmap details.
- Provide 1-3 follow_up_questions that anticipate the learner's likely next
  question, grounded in the SPECIFIC tasks/topics just discussed (not
  generic questions like "anything else?").

Return a single JSON object with EXACTLY this shape:
{{
  "response": <string>,
  "follow_up_questions": [<string>, ...]
}}
"""
