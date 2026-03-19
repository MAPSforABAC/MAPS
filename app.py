#!/usr/bin/env python3
import os
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file, redirect, url_for, flash

from app10 import run_pipeline_to_zip

app = Flask(__name__)
app.secret_key = "change-me" 

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
WORK_DIR = os.path.abspath("webwork")
os.makedirs(WORK_DIR, exist_ok=True)

ALLOWED_EXT = {".zip"}

PROVIDERS = [
    ("openai", "OpenAI"),
    ("gemini", "Gemini"),
    ("tng", "TNG (OpenRouter)"),
    ("openai_four", "Openai4"),
    ("perplexity", "Perplexity"),
]

DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-2.5-flash",
    "tng": "tngtech/deepseek-r1t2-chimera:free",
    "openai_four": "gpt-4o",
    "perplexity": "sonar",
}

MODES = [("auto", "Auto (prefer attributes.txt)"), ("both", "Use attributes.txt"), ("single", "Dynamic attributes"), ("lmn0", "LMN0 (10 rules)")]


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        providers=PROVIDERS,
        default_models=DEFAULT_MODELS,
        modes=MODES,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "zipfile" not in request.files:
        flash("Please choose a .zip file.")
        return redirect(url_for("index"))

    f = request.files["zipfile"]
    if not f or f.filename.strip() == "":
        flash("Please choose a .zip file.")
        return redirect(url_for("index"))

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        flash("Only .zip files are accepted.")
        return redirect(url_for("index"))

    provider = request.form.get("provider", "openai")
    mode = request.form.get("mode", "auto")

    model_stage = request.form.get("model_stage", "").strip() or DEFAULT_MODELS.get(provider, "gpt-3.5-turbo")
    model_rules = request.form.get("model_rules", "").strip() or DEFAULT_MODELS.get(provider, "gpt-3.5-turbo")
    model_table = request.form.get("model_table", "").strip() or DEFAULT_MODELS.get(provider, "gpt-3.5-turbo")

    job_id = uuid.uuid4().hex[:10]
    job_dir = os.path.join(WORK_DIR, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)

    safe_name = secure_filename(f.filename) or "input.zip"
    in_path = os.path.join(job_dir, safe_name)
    f.save(in_path)

    base_name = os.path.splitext(secure_filename(f.filename))[0]
    out_path = os.path.join(job_dir, f"{base_name}_out.zip")


    try:
        zip_path = run_pipeline_to_zip(
            input_path=in_path,
            outzip_path=out_path,
            provider=provider,
            model_stage=model_stage,
            model_rules=model_rules,
            model_table=model_table,
            mode=mode,
        )
    except Exception as e:
        flash(f"Processing failed: {e}")
        return redirect(url_for("index"))

    return send_file(
        zip_path,
        as_attachment=True,
        download_name=os.path.basename(zip_path),
        mimetype="application/zip",
        conditional=False,
        max_age=0,
    )


if __name__ == "__main__":
    app.run(debug=True)
