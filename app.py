import os
import json
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env
load_dotenv()
client = OpenAI()

SCENE_DIR = "scenes"

app = Flask(__name__, static_folder='.', static_url_path='')

from pathlib import Path

def load_scene_text(act: str, scene: str) -> str:
    """
    Load full text for King Lear, Act {act}, Scene {scene}
    from scenes/lear_act{act}_scene{scene}.txt
    """
    filename = f"lear_act{act}_scene{scene}.txt"
    path = Path(SCENE_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"No scene file: {path}")
    return path.read_text(encoding="utf-8")


def analyze_with_gpt(
    text: str,
    act: str | None = None,
    scene: str | None = None,
) -> dict:
    """
    GPT-based madness scoring.

    Key idea: we are always in Shakespeare's *King Lear*.
    - Lear's mind should be treated as troubled / distorted, even when the language
      is controlled and formal (e.g. Act 1 division of the kingdom).
    - Storm / heath / hut scenes in Acts 3–4 should be read as extremely mad.
    GPT can infer who is speaking from the style/content, but we give it Act/Scene.
    """

    context_bits = []
    if act:
        context_bits.append(f"Act {act}")
    if scene:
        context_bits.append(f"Scene {scene}")
    context_str = ""
    if context_bits:
        context_str = "Context: this passage is from King Lear, " + ", ".join(context_bits) + ".\n"

    prompt = f"""
You are a literary critic and linguistics assistant analyzing **psychological madness** in Shakespeare's *King Lear*.

{context_str}
Your job is to rate the **mental / emotional disturbance of the speaker**, not just how broken the syntax looks.

Key idea:
- In *King Lear*, a speaker can sound controlled, formal, and rhetorically polished **and still be very mad** if their judgment is warped, cruel, impulsive, or detached from reality.
- If you recognize Lear speaking (e.g. dividing the kingdom, cursing his daughters, raging in the storm, or philosophizing on the heath), you should usually assign a **high madness_score**, often 70+.
- Storm / heath / hut scenes in Acts 3–4 where Lear's mind openly unravels can deserve **very high** scores (80–100).
- Other characters (Cordelia, Kent, etc.) can have low scores if they sound clear, grounded, and compassionate.

You will output:
1. An overall **madness_score** (0–100), where:
   - 0 = mentally stable, balanced, clear judgment
   - 100 = extremely disturbed, unstable, or unmoored

2. Three sub-scores in [0,1] that *explain* the flavor of madness.
   They are interpretive dimensions, not strict mathematical measures:

- semantic_disorganization:
  * Higher = bigger jumps in thought, conflicting images, weak logical progression.
  * For controlled but mad Lear, this can still be moderate–high (e.g. 0.5–0.8) if the content is emotionally or morally deranged.

- graph_randomness:
  * Higher = more tangled flow of ideas, looping back, abrupt transitions, or piling up of images.

- lexical_weirdness:
  * Higher = strange or theatrical pronoun use, vocatives, curses, heavy questions/exclamations, and emotionally charged diction.

Use your knowledge of the play and of Lear's arc. The numbers are just a way to encode your critical judgment.

Now analyze this passage:

\"\"\"{text}\"\"\"

Return ONLY a JSON object with these keys:
- "madness_score" (number 0–100)
- "semantic_disorganization" (number 0–1)
- "graph_randomness" (number 0–1)
- "lexical_weirdness" (number 0–1)
- "comment" (short 1–3 sentence explanation)
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-4.1 / gpt-5-mini / etc.
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    content = completion.choices[0].message.content
    data = json.loads(content)
    return data


@app.route("/")
def index():
    # serve index.html from this folder
    return send_from_directory(".", "index.html")


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    act = payload.get("act")
    scene = payload.get("scene")

    try:
        result = analyze_with_gpt(text, act=act, scene=scene)
        return jsonify(result)
    except Exception as e:
        print("Error calling GPT:", e)
        return jsonify({"error": "GPT analysis failed."}), 500


@app.get("/api/scene")
def api_scene():
    """
    GET /api/scene?act=3&scene=4
    → { "text": "full scene text..." }
    """
    act = request.args.get("act")
    scene = request.args.get("scene")
    if not act or not scene:
        return jsonify({"error": "act and scene are required"}), 400

    try:
        text = load_scene_text(act, scene)
        return jsonify({"text": text})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print("Error loading scene:", e)
        return jsonify({"error": "Failed to load scene"}), 500


if __name__ == "__main__":
    app.run(debug=True)
