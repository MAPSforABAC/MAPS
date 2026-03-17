#!/usr/bin/env python3

import os
import re
import argparse
import zipfile
import tempfile
import pdfplumber
import json
from typing import List, Tuple
import time 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap
import math
import openai
from google.ai.generativelanguage_v1 import types
import google.generativeai as genai
import requests
import argparse
import re
import zipfile
import os
import tempfile
import pdfplumber
import json
from typing import List, Tuple
import anthropic

OPENAI_35_KEY = "" #Please add openai gpt-3.5-turbo key here
GEMINI_25_KEY = "" #Please add gemini-2.5-flash key here
OPENROUTER_API_KEY =  "" #Please add your api key here
OPENAI_GPT_4O = "" #Please add openai gpt-4o key here
MINIMAX_API_KEY ="" #Please add your api key here
MINIMAX_MODEL_ID = "MiniMax-M2"
PERPLEXITY_API_KEY = "" #please add your api key here
PERPLEXITY_DEFAULT_MODEL = "sonar"  # you can also use 'sonar', 'sonar-medium-chat', etc.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


SYSTEM_STAGE1 = (
    "You filter meeting text. Keep ONLY access-control-relevant content: "
    "roles/titles, departments, systems/resources, allow/deny/grant, purpose, day/time windows, locations. "
    "No chit-chat, no meta-discussion, no greetings. Output exactly one concise paragraph."
)

USER_STAGE1_TEMPLATE = (
    "Input:\n{chunk}\n\n"
    "Task: Return a single compact paragraph with ONLY facts that could become access rules. "
    "Remove everything else."
)

SYSTEM_STAGE2 = (
    "You generate standardized NLACP-style sentences. "
    "Follow a strict pattern. No explanations. One policy per line."
)

USER_STAGE2 = (
    "From the input paragraph, write access rules as single sentences. "
    "Pattern: (Subject/Role [+ Department]) are (allowed/denied/granted) access to "
    "(System or Resource) [for <purpose>] [during <days>] [from/before/after <time-range>]. "
    "If a system/resource name is missing, use a neutral placeholder like SystemX. "
    "Keep wording formal and concise. One rule per line only.\n\n"
    "Input:\n{filtered_paragraph}"
)

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text.append(p.extract_text() or "")
    raw = "\n".join(text)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return raw

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def extract_text_from_docx(path: str) -> str:
    """
    Single-extractor version using docx2txt only.
    Returns normalized plain text.
    """
    try:
        import docx2txt
    except Exception as e:
        raise RuntimeError("Please install docx2txt: pip install docx2txt") from e

    text = docx2txt.process(path, None) or ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    print(text)
    return text


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        return ""  # ignore unsupported types

def extract_text_from_dir(dir_path: str) -> Tuple[str, str]:
    """Return (merged_text, attributes_content_if_found)."""
    merged = []
    attributes_content = ""
    for root, _, files in os.walk(dir_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            fp = os.path.join(root, name)
            if name.lower() == "attributes.txt":
                try:
                    attributes_content = extract_text_from_txt(fp)
                except Exception:
                    pass
                continue
            if ext in {".pdf", ".txt", ".docx"}:
                try:
                    merged.append(extract_text_from_file(fp))
                except Exception:
                    pass
    return "\n\n".join([m for m in merged if m.strip()]), attributes_content.strip()

def extract_text_from_zip(zip_path: str) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        merged_text, attributes = extract_text_from_dir(td)
    return merged_text, attributes

def load_input_text_and_attributes(input_path: str) -> Tuple[str, str]:
    """Returns (merged_text, attributes_content_if_any)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if os.path.isdir(input_path):
        return extract_text_from_dir(input_path)

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".zip":
        return extract_text_from_zip(input_path)
    elif ext in {".pdf", ".txt", ".docx"}:
        return extract_text_from_file(input_path), ""
    else:
        raise ValueError("Input must be a .zip, .pdf, .txt, .docx, or a directory containing these files.")

def chunk_text(s: str, max_chars: int = 4000) -> List[str]:
    s = s.strip()
    if not s:
        return []
    chunks, start = [], 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        cut = s.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + 1200:
            cut = end
        chunk = s[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut
    return chunks


def llm_openai(system: str, user: str, model: str = "gpt-3.5-turbo") -> str:
    openai.api_key =  OPENAI_35_KEY 
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

def llm_gemini(system: str, user: str, model: str = "gemini-2.5-flash") -> str:
    api_key = GEMINI_25_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    full_prompt = f"System: {system}\nUser: {user}"
    data = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ]
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        result = resp.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Gemini response format unexpected. Full response:\n" + str(result)
    else:
        return f"Gemini API Error {resp.status_code}: {resp.text}"


TNG_MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"
def llm_tng(system: str, user: str, model: str = TNG_MODEL_ID) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "NLACP-to-MESP-System"
    }
    payload = {
        "model": model or TNG_MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"TNG OpenRouter Error: {str(e)}"

QWEN_MODEL_ID = "qwen/qwen3-235b-a22b:free"
def llm_qwen(system: str, user: str, model: str = QWEN_MODEL_ID) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "NLACP-to-MESP-System"
    }
    payload = {
        "model": model or QWEN_MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f" Qwen OpenRouter Error: {str(e)}"

def llm_openai_four(system: str, user: str, model: str = "gpt-4o") -> str:
    openai.api_key = OPENAI_GPT_4O
    
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    
    return response["choices"][0]["message"]["content"].strip()


def llm_minimax(system: str, user: str, model: str = MINIMAX_MODEL_ID) -> str:
    try:
        client = anthropic.Anthropic(
            base_url="https://api.minimax.io/anthropic",
            api_key=MINIMAX_API_KEY
        )
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        if hasattr(message, "content") and message.content:
            first = message.content[0]
            if hasattr(first, "text"):
                return first.text.strip()
            if isinstance(first, dict) and "text" in first:
                return first["text"].strip()
        return str(message)
    except Exception as e:
        return f"MiniMax Error: {e}"


def llm_perplexity(system: str, user: str, model: str = PERPLEXITY_DEFAULT_MODEL) -> str:
    """
    Perplexity chat.completions (OpenAI-compatible)
    Endpoint: https://api.perplexity.ai/chat/completions
    """
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model or PERPLEXITY_DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Perplexity Error: {e}"

def stage1_filter(raw_text: str, llm_fn) -> str:
    chunks = chunk_text(raw_text)
    if not chunks:
        return ""
    filtered_bits = []
    for c in chunks:
        user = USER_STAGE1_TEMPLATE.format(chunk=c)
        filtered_bits.append(llm_fn(SYSTEM_STAGE1, user))
    merged = " ".join(filtered_bits)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged

def stage2_to_nlacp(filtered_paragraph: str, llm_fn) -> str:
    if not filtered_paragraph.strip():
        return ""
    return llm_fn(SYSTEM_STAGE2, USER_STAGE2.format(filtered_paragraph=filtered_paragraph))

TIME_NORMALIZATION = [
    (r"\boffice hours\b", "weekdays from 9 AM to 5 PM"),
    (r"\bworking hours\b", "weekdays from 9 AM to 5 PM"),
]

def normalize_time_phrases(s: str) -> str:
    out = s
    for pat, repl in TIME_NORMALIZATION:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

RULE_OK_RE = re.compile(r"\b(allowed|denied|granted)\b.*\baccess to\b", re.IGNORECASE)

def validate_and_fix_rules(nlacp_text: str) -> str:
    lines = [ln.strip() for ln in nlacp_text.split("\n") if ln.strip()]
    fixed = []
    for ln in lines:
        ln = normalize_time_phrases(ln)
        if not ln.endswith("."):
            ln += "."
        if RULE_OK_RE.search(ln) and len(ln.split()) >= 6:
            fixed.append(ln)
        else:
            if "access to" not in ln.lower():
                ln = re.sub(r"(allowed|denied|granted)\b", r"\1 access to SystemX", ln, flags=re.IGNORECASE)
            if not RULE_OK_RE.search(ln):
                continue
            fixed.append(ln)
    uniq, seen = [], set()
    for ln in fixed:
        key = ln.lower()
        if key not in seen:
            uniq.append(ln)
            seen.add(key)
    return "\n".join(uniq)

def extract_attributes_used(gpt_output: str):
    pattern = re.compile(r"\(([^:]+):")
    attributes = set(re.findall(pattern, gpt_output))
    return list(attributes)

def _call_llm_provider(system_prompt: str, prompt: str,
                       provider: str,
                       model_openai: str,
                       model_gemini: str,
                       model_tng: str,
                       model_qwen: str,
                       model_openai_four: str,
                       model_minimax: str,
                       model_perplexity: str):
    if provider == "gemini":
        return llm_gemini(system_prompt, prompt, model=model_gemini or "gemini-2.5-flash")
    elif provider == "tng":
        return llm_tng(system_prompt, prompt, model=model_tng or "tngtech/deepseek-r1t2-chimera:free")
    elif provider == "qwen":
        return llm_qwen(system_prompt, prompt, model=model_qwen or "qwen/qwen3-235b-a22b:free")
    elif provider == "openai_four":
        return llm_openai_four(system_prompt, prompt, model=model_openai_four or "gpt-4o")
    elif provider == "minimax":
        return llm_minimax(system_prompt, prompt, model=model_minimax or "MiniMax-M2")
    elif provider == "perplexity":
        return llm_perplexity(system_prompt, prompt, model=model_perplexity or PERPLEXITY_DEFAULT_MODEL)
    else:
        return llm_openai(system_prompt, prompt, model=model_openai or "gpt-3.5-turbo")

def generate_abac_rules_from_content(nlacp_content: str,
                                     model_selection: str,
                                     attributes_content: str = "",
                                     provider: str = "openai",
                                     model_openai: str = "gpt-3.5-turbo",
                                     model_gemini: str = "gemini-2.5-flash",
                                     model_tng: str = "tngtech/deepseek-r1t2-chimera:free",
                                     model_qwen: str = "qwen/qwen3-235b-a22b:free",
                                     model_openai_four: str = "gpt-4o",
                                     model_minimax: str = "MiniMax-M2",
                                     model_perplexity: str = PERPLEXITY_DEFAULT_MODEL):
    print("I am here..")
    if model_selection == "both":
        prompt = (
            "Convert the following natural language access control policies into structured ABAC rules using "
            f"the specified attributes: {attributes_content}.\n\n"
            "Example Format:\n"
            "1: (Label: Allow), (Role: User), (Resource: System)\n"
            f"Input:\n{nlacp_content}"
        )
    elif model_selection == "single":
        prompt = (
            "Dynamically extract attributes and generate structured ABAC rules from the following "
            "natural language descriptions of access control policies.\n"
            "Example Format:\n"
            "1: (Label: Allow), (Role: User), (Resource: System)\n"
            f"Input:\n{nlacp_content}"
        )
    elif model_selection == "lmn0":
        prompt = (
            "I am giving you input as a statement of my new model which needs to generate ABAC rules and then modify "
            "them in the given format. You must list rules as per format and not like a statement:\n"
            "1: (Attribute1: Value1), (Attribute2: Value2),(Attribute3: Value3) ... (min 5, max 10 attributes)\n"
            "Take the following statement as title and generate 10 rules in the above format.\n"
            f"Statement:\n{nlacp_content}"
        )
    else:
        raise ValueError("Invalid model_selection (use 'both', 'single', or 'lmn0').")

    system_prompt = "Please format the rules and list the attributes as specified."
    output = _call_llm_provider(system_prompt, prompt,
                                provider, model_openai, model_gemini, model_tng, model_qwen,
                                model_openai_four, model_minimax, model_perplexity)
    attributes_used = extract_attributes_used(output)
    return output, attributes_used

def generate_xacml_snippet(rule: str,
                           provider: str = "openai",
                           model_openai: str = "gpt-3.5-turbo",
                           model_gemini: str = "gemini-2.5-flash",
                           model_tng: str = "tngtech/deepseek-r1t2-chimera:free",
                           model_qwen: str = "qwen/qwen3-235b-a22b:free",
                           model_openai_four: str = "gpt-4o",
                           model_minimax: str = "MiniMax-M2",
                           model_perplexity: str = PERPLEXITY_DEFAULT_MODEL) -> str:
    prompt = f"Convert the following ABAC rule into an XACML policy snippet:\n{rule}"
    system = "Please generate a well-structured XACML policy snippet for the given rule."
    return _call_llm_provider(system, prompt, provider,
                              model_openai, model_gemini, model_tng, model_qwen,
                              model_openai_four, model_minimax, model_perplexity)

def generate_complete_xacml(gpt_output: str,
                            provider: str = "openai",
                            model_openai: str = "gpt-3.5-turbo",
                            model_gemini: str = "gemini-2.5-flash",
                            model_tng: str = "tngtech/deepseek-r1t2-chimera:free",
                            model_qwen: str = "qwen/qwen3-235b-a22b:free",
                            model_openai_four: str = "gpt-4o",
                            model_minimax: str = "MiniMax-M2",
                            model_perplexity: str = PERPLEXITY_DEFAULT_MODEL) -> str:
    rules = [r.strip() for r in gpt_output.split("\n") if r.strip()]
    xacml_policies = []
    for rule in rules:
        xacml_snippet = generate_xacml_snippet(
            rule,
            provider=provider,
            model_openai=model_openai,
            model_gemini=model_gemini,
            model_tng=model_tng,
            model_qwen=model_qwen,
            model_openai_four=model_openai_four,
            model_minimax=model_minimax,
            model_perplexity=model_perplexity
        )
        xacml_policies.append(xacml_snippet)
    full_xacml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PolicySet xmlns="urn:oasis:names:tc:xacml:3.0" PolicySetId="ExamplePolicySet" Version="1.0" '
        'PolicyCombiningAlgId="urn:oasis:names:tc:xacml:3.0:policy-combining-algorithm:deny-overrides">\n'
        + "\n".join(xacml_policies) +
        "\n</PolicySet>"
    )
    return full_xacml

def request_table_json_from_gpt(mesp_text: str,
                                model: str = "gpt-3.5-turbo",
                                provider: str = "openai",
                                model_gemini: str = "gemini-2.5-flash",
                                model_tng: str = "tngtech/deepseek-r1t2-chimera:free",
                                model_qwen: str = "qwen/qwen3-235b-a22b:free",
                                model_openai_four: str = "gpt-4o",
                                model_minimax: str = "MiniMax-M2",
                                model_perplexity: str = PERPLEXITY_DEFAULT_MODEL) -> str:
    system = "You convert ABAC-like rule lines into a normalized table."
    user = (
        "You are given a list of policy lines in the form: "
        "1: (AttrA: ValA), (AttrB: ValB), ... one line per policy.\n\n"
        "Task: Build a table that includes ALL unique attribute names found across ALL policies as columns. "
        "Return strict JSON ONLY in the form:\n"
        "{\n"
        '  "columns": ["AttrA","AttrB",...],\n'
        '  "rows": [\n'
        '     {"AttrA": "value or null", "AttrB": "value or null", ...},\n'
        "     ... one object per input policy line ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Include a row for each policy in input order.\n"
        "- For missing values in a row, use null.\n"
        "- If a value contains commas, keep it as a single string.\n"
        "- Do not add any extra text, no markdown fences, just JSON.\n\n"
        f"Input:\n{mesp_text}"
    )
    if provider == "gemini":
        return llm_gemini(system, user, model=model_gemini)
    elif provider == "tng":
        return llm_tng(system, user, model=model_tng)
    elif provider == "qwen":
        return llm_qwen(system, user, model=model_qwen)
    elif provider == "openai_four":
        return llm_openai_four(system, user, model=model_openai_four)
    elif provider == "minimax":
        return llm_minimax(system, user, model=model_minimax)
    elif provider == "perplexity":
        return llm_perplexity(system, user, model=model_perplexity or PERPLEXITY_DEFAULT_MODEL)
    else:
        return llm_openai(system, user, model=model)


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(json)?", "", s.strip(), flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s.strip()).strip()
    return s


def json_to_dataframe(json_text: str):
    import pandas as pd
    data = json.loads(json_text)
    cols = data.get("columns", [])
    rows = data.get("rows", [])
    norm_rows = []
    for r in rows:
        norm = {c: (r.get(c, None)) for c in cols}
        norm_rows.append(norm)
    return pd.DataFrame(norm_rows, columns=cols)

matplotlib.use("Agg")  

def _split_multi_tokens(x: str) -> list:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "null":
        return []
    s = s.replace(" and ", ", ")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or [s]

def collect_attribute_values(table_df: pd.DataFrame) -> dict:
    """Build {attribute: sorted unique values} from the MESP table DataFrame."""
    attr_vals = {}
    for col in table_df.columns:
        values = []
        for raw in table_df[col].dropna().astype(str).tolist():
            values.extend(_split_multi_tokens(raw))
        seen, uniq = set(), []
        for v in values:
            if v and v.lower() != "null" and v not in seen:
                seen.add(v)
                uniq.append(v)
        attr_vals[col] = uniq
    return attr_vals

def plot_attribute_suns(attr_values: dict, out_path: str = "attributes_graph.png",
                        cols: int = 3, label_wrap: int = 18) -> str:
    n = len(attr_values)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             subplot_kw={'projection': 'polar'},
                             figsize=(5.2 * cols, 5.2 * rows),
                             constrained_layout=True)

    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    items = list(attr_values.items())

    for ax, (attr, values) in zip(axes, items):
        ax.set_axis_off()
        ax.set_ylim(0, 1.1)

        ax.scatter([0], [0.02], s=260, color="#1f77b4", zorder=3)
        ax.text(0, 0.02, "\n".join(wrap(str(attr), width=14)),
                ha="center", va="center", fontsize=10, color="white")

        k = max(1, len(values))
        for i, val in enumerate(values):
            theta = 2 * np.pi * i / k
            ax.plot([theta, theta], [0.08, 0.86], lw=0.8, color="0.45", zorder=1)
            ax.scatter([theta], [0.9], s=25, color="0.25", zorder=2)

            txt = "\n".join(wrap(str(val), width=label_wrap))
            rotation = np.degrees(theta)
            if 90 < rotation < 270:
                rotation += 180
                ha = "right"
            else:
                ha = "left"
            ax.text(theta, 0.98, txt, fontsize=8, ha=ha, va="center",
                    rotation=rotation, rotation_mode="anchor")

        circle_t = np.linspace(0, 2 * np.pi, 360)
        ax.plot(circle_t, np.full_like(circle_t, 0.88), color="0.75", lw=0.6, zorder=0)

    for j in range(len(items), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attribute → Values Mapping", fontsize=14, y=1.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_pipeline_to_zip(
    input_path: str,
    outzip_path: str = None,
    provider: str = "openai",
    model_stage: str = "gpt-3.5-turbo",
    model_rules: str = "gpt-3.5-turbo",
    model_table: str = "gpt-3.5-turbo",
    mode: str = "auto"  # choices: auto|both|single|lmn0
) -> str:
    
    start_time = time.time()

    if outzip_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        outzip_path = f"{base_name}_out.zip"

    raw, attributes_content = load_input_text_and_attributes(input_path)

    if provider == "gemini":
        def llm(sys, usr, mdl=model_stage):
            gem_model = "gemini-2.5-flash" if mdl == "gpt-3.5-turbo" else mdl
            return llm_gemini(sys, usr, model=gem_model)
    elif provider == "tng":
        def llm(sys, usr, mdl=model_stage):
            tng_model = "tngtech/deepseek-r1t2-chimera:free" if mdl == "gpt-3.5-turbo" else mdl
            return llm_tng(sys, usr, model=tng_model)
    elif provider == "qwen":
        def llm(sys, usr, mdl=model_stage):
            qwen_model = "qwen/qwen3-235b-a22b:free" if mdl == "gpt-3.5-turbo" else mdl
            return llm_qwen(sys, usr, model=qwen_model)
    elif provider == "openai_four":
        def llm(sys, usr, mdl=model_stage):
            gem_or_model = "gpt-4o" if mdl == "gpt-3.5-turbo" else mdl
            return llm_openai_four(sys, usr, model=gem_or_model)
    elif provider == "minimax":
        def llm(sys, usr, mdl=model_stage):
            minimax_model = "MiniMax-M2" if mdl == "gpt-3.5-turbo" else mdl
            return llm_minimax(sys, usr, model=minimax_model)
    elif provider == "perplexity":
        def llm(sys, usr, mdl=model_stage):
            pp_model = PERPLEXITY_DEFAULT_MODEL if mdl == "gpt-3.5-turbo" else mdl
            return llm_perplexity(sys, usr, model=pp_model)
    else:
        def llm(sys, usr, mdl=model_stage):
            return llm_openai(sys, usr, model=mdl)

    filtered = stage1_filter(raw, llm)
    nlacp_raw = stage2_to_nlacp(filtered, llm)
    nlacp_raw = re.sub(r"\.\s+", ".\n", nlacp_raw)
    nlacp_clean = validate_and_fix_rules(nlacp_raw)

    model_selection = "both" if (mode == "auto" and attributes_content) else ("single" if mode == "auto" else mode)

    gpt_output, attributes_used = generate_abac_rules_from_content(
        nlacp_clean,
        model_selection=model_selection,
        attributes_content=attributes_content,
        provider=provider,
        model_openai=model_rules,
        model_gemini=("gemini-2.5-flash" if model_rules == "gpt-3.5-turbo" else model_rules),
        model_tng=("tngtech/deepseek-r1t2-chimera:free" if model_rules == "gpt-3.5-turbo" else model_rules),
        model_qwen=("qwen/qwen3-235b-a22b:free" if model_rules == "gpt-3.5-turbo" else model_rules),
        model_openai_four=("gpt-4o" if model_rules == "gpt-3.5-turbo" else model_rules),
        model_minimax=("MiniMax-M2" if model_rules == "gpt-3.5-turbo" else model_rules),
        model_perplexity=(PERPLEXITY_DEFAULT_MODEL if model_rules == "gpt-3.5-turbo" else model_rules)
    )

    table_json_str = request_table_json_from_gpt(
        gpt_output,
        model=model_table,
        provider=provider,
        model_gemini=("gemini-2.5-flash" if model_table == "gpt-3.5-turbo" else model_table),
        model_tng=("tngtech/deepseek-r1t2-chimera:free" if model_table == "gpt-3.5-turbo" else model_table),
        model_qwen=("qwen/qwen3-235b-a22b:free" if model_table == "gpt-3.5-turbo" else model_table),
        model_openai_four=("gpt-4o" if model_table == "gpt-3.5-turbo" else model_table),
        model_minimax=("MiniMax-M2" if model_table == "gpt-3.5-turbo" else model_table),
        model_perplexity=(PERPLEXITY_DEFAULT_MODEL if model_table == "gpt-3.5-turbo" else model_table)
    )
    table_json_str = strip_code_fences(table_json_str)

    df = json_to_dataframe(table_json_str)
    table_csv_path = "attributes_table.csv"
    df.to_csv(table_csv_path, index=False)
    attr_values = collect_attribute_values(df)
    graph_img_path = plot_attribute_suns(attr_values, out_path="attributes_graph.png", cols=3, label_wrap=18)

    stage1_path = "stage1.txt"
    stage2_path = "stage2.txt"
    nlacp_path  = "NLACP.txt"
    mesp_path   = "MESP.txt"
    gpta_path   = "gpt_attribute.txt"

    with open(stage1_path, "w", encoding="utf-8") as f:
        f.write(raw)
    with open(stage2_path, "w", encoding="utf-8") as f:
        f.write(filtered)
    with open(nlacp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(nlacp_clean.splitlines()))
    with open(mesp_path, "w", encoding="utf-8") as f:
        f.write(gpt_output)
    with open(gpta_path, "w", encoding="utf-8") as f:
        f.write(", ".join(attributes_used))

    elapsed_time = time.time() - start_time

    with open("time_taken.txt", "w") as tfile:
        tfile.write(f"Processing completed in {elapsed_time:.2f} seconds.\n")

    with zipfile.ZipFile(outzip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(stage1_path, arcname="stage1.txt")
        zf.write(stage2_path, arcname="stage2.txt")
        zf.write(nlacp_path,  arcname="NLACP.txt")
        zf.write(mesp_path,   arcname="MESP.txt")
        zf.write(gpta_path,   arcname="gpt_attribute.txt")
        zf.write(table_csv_path, arcname="attributes_table.csv")
        zf.write(graph_img_path, arcname="attributes_graph.png")
        zf.write("time_taken.txt", arcname="time_taken.txt")

    

    print(f"[INFO] Total processing time: {elapsed_time:.2f} seconds")    

    return outzip_path

def main():
    ap = argparse.ArgumentParser(description="ZIP/pdf/txt/docx ➜ filtered paragraph ➜ NLACP ➜ MESP + XACML (policy.zip)")
    ap.add_argument("input", help="Path to .zip | .pdf | .txt | .docx | directory containing these files")

    ap.add_argument("--model_stage", default="gpt-3.5-turbo",
                    help="Model for Stage 1 & 2 (OpenAI) or gemini-2.5-flash / deepseek-r1t2-chimera / qwen3-235b-a22b / gpt-4o / MiniMax-M2 / sonar if other provider")
    ap.add_argument("--model_rules", default="gpt-3.5-turbo",
                    help="Model for ABAC/XACML (OpenAI) or gemini-2.5-flash / deepseek-r1t2-chimera / qwen3-235b-a22b / gpt-4o / MiniMax-M2 / sonar for others")
    ap.add_argument("--model_table", default="gpt-3.5-turbo",
                    help="Model used for table extraction from MESP.txt")

    ap.add_argument("--mode", choices=["auto", "both", "single", "lmn0"], default="auto",
                    help="'auto' uses 'both' if attributes.txt is present, else 'single'")
    ap.add_argument("--outzip", default="policy.zip", help="Output ZIP file name")

    ap.add_argument("--provider",
                    choices=["openai", "gemini", "tng", "qwen", "openai_four", "minimax", "perplexity"],
                    default="openai",
                    help="Select LLM provider: openai | gemini | tng | qwen | openai_four | minimax | perplexity")
    args = ap.parse_args()

    raw, attributes_content = load_input_text_and_attributes(args.input)

    if args.provider == "gemini":
        def llm(sys, usr, mdl=args.model_stage):
            gem_model = "gemini-2.5-flash" if mdl == "gpt-3.5-turbo" else mdl
            return llm_gemini(sys, usr, model=gem_model)
    elif args.provider == "tng":
        def llm(sys, usr, mdl=args.model_stage):
            tng_model = "tngtech/deepseek-r1t2-chimera:free" if mdl == "gpt-3.5-turbo" else mdl
            return llm_tng(sys, usr, model=tng_model)
    elif args.provider == "qwen":
        def llm(sys, usr, mdl=args.model_stage):
            qwen_model = "qwen/qwen3-235b-a22b:free" if mdl == "gpt-3.5-turbo" else mdl
            return llm_qwen(sys, usr, model=qwen_model)
    elif args.provider == "openai_four":
        def llm(sys, usr, mdl=args.model_stage):
            gem_or_model = "gpt-4o" if mdl == "gpt-3.5-turbo" else mdl
            return llm_openai_four(sys, usr, model=gem_or_model)
    elif args.provider == "minimax":
        def llm(sys, usr, mdl=args.model_stage):
            minimax_model = "MiniMax-M2" if mdl == "gpt-3.5-turbo" else mdl
            return llm_minimax(sys, usr, model=minimax_model)
    elif args.provider == "perplexity":
        def llm(sys, usr, mdl=args.model_stage):
            pp_model = PERPLEXITY_DEFAULT_MODEL if mdl == "gpt-3.5-turbo" else mdl
            return llm_perplexity(sys, usr, model=pp_model)
    else:
        def llm(sys, usr, mdl=args.model_stage):
            return llm_openai(sys, usr, model=mdl)

    filtered = stage1_filter(raw, llm)

    nlacp_raw = stage2_to_nlacp(filtered, llm)
    nlacp_raw = re.sub(r"\.\s+", ".\n", nlacp_raw)  
    nlacp_clean = validate_and_fix_rules(nlacp_raw)

    model_selection = "both" if (args.mode == "auto" and attributes_content) else ("single" if args.mode == "auto" else args.mode)

    gpt_output, attributes_used = generate_abac_rules_from_content(
        nlacp_clean,
        model_selection=model_selection,
        attributes_content=attributes_content,
        provider=args.provider,
        model_openai=args.model_rules,
        model_gemini=("gemini-2.5-flash" if args.model_rules == "gpt-3.5-turbo" else args.model_rules),
        model_tng=("tngtech/deepseek-r1t2-chimera:free" if args.model_rules == "gpt-3.5-turbo" else args.model_rules),
        model_qwen=("qwen/qwen3-235b-a22b:free" if args.model_rules == "gpt-3.5-turbo" else args.model_rules),
        model_openai_four=("gpt-4o" if args.model_rules == "gpt-3.5-turbo" else args.model_rules),
        model_minimax=("MiniMax-M2" if args.model_rules == "gpt-3.5-turbo" else args.model_rules),
        model_perplexity=(PERPLEXITY_DEFAULT_MODEL if args.model_rules == "gpt-3.5-turbo" else args.model_rules)
    )
    print("I have generated output")
    table_json_str = request_table_json_from_gpt(
        gpt_output,
        model=args.model_table,
        provider=args.provider,
        model_gemini=("gemini-2.5-flash" if args.model_table == "gpt-3.5-turbo" else args.model_table),
        model_tng=("tngtech/deepseek-r1t2-chimera:free" if args.model_table == "gpt-3.5-turbo" else args.model_table),
        model_qwen=("qwen/qwen3-235b-a22b:free" if args.model_table == "gpt-3.5-turbo" else args.model_table),
        model_openai_four=("gpt-4o" if args.model_table == "gpt-3.5-turbo" else args.model_table),
        model_minimax=("MiniMax-M2" if args.model_table == "gpt-3.5-turbo" else args.model_table),
        model_perplexity=(PERPLEXITY_DEFAULT_MODEL if args.model_table == "gpt-3.5-turbo" else args.model_table)
    )
    table_json_str = strip_code_fences(table_json_str)
    print("I am at table")
    print(table_json_str)
    # ---- Dataframe + CSV + Graph
    df = json_to_dataframe(table_json_str)
    table_csv_path = "attributes_table.csv"
    df.to_csv(table_csv_path, index=False)

    attr_values = collect_attribute_values(df)
    graph_img_path = plot_attribute_suns(
        attr_values,
        out_path="attributes_graph.png",
        cols=3,
        label_wrap=18
    )

    stage1_path = "stage1.txt"  # merged content
    stage2_path = "stage2.txt"  # filtered paragraph
    nlacp_path = "NLACP.txt"    # final NLACP sentences
    mesp_path = "MESP.txt"      # ABAC rules
    gpta_path = "gpt_attribute.txt"
    # xacml_path = "xacml_policies.xml"  # optional if you enable generate_complete_xacml()

    with open(stage1_path, "w", encoding="utf-8") as f:
        f.write(raw)
    with open(stage2_path, "w", encoding="utf-8") as f:
        f.write(filtered)
    with open(nlacp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(nlacp_clean.splitlines()))
    with open(mesp_path, "w", encoding="utf-8") as f:
        f.write(gpt_output)
    with open(gpta_path, "w", encoding="utf-8") as f:
        f.write(", ".join(attributes_used))
    # If you want full XACML, uncomment below:
    # xacml_full = generate_complete_xacml(
    #     gpt_output,
    #     provider=args.provider,
    #     model_openai=args.model_rules,
    #     model_gemini="gemini-2.5-flash",
    #     model_tng="tngtech/deepseek-r1t2-chimera:free",
    #     model_qwen="qwen/qwen3-235b-a22b:free",
    #     model_openai_four="gpt-4o:free",
    #     model_minimax="MiniMax-M2",
    #     model_perplexity=PERPLEXITY_DEFAULT_MODEL
    # )
    # with open(xacml_path, "w", encoding="utf-8") as f:
    #     f.write(xacml_full)

    # ---- Pack ZIP
    with zipfile.ZipFile(args.outzip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(stage1_path, arcname="stage1.txt")
        zf.write(stage2_path, arcname="stage2.txt")
        zf.write(nlacp_path, arcname="NLACP.txt")
        zf.write(mesp_path, arcname="MESP.txt")
        zf.write(gpta_path, arcname="gpt_attribute.txt")
        # zf.write(xacml_path, arcname="xacml_policies.xml")
        zf.write(table_csv_path, arcname="attributes_table.csv")
        zf.write(graph_img_path, arcname="attributes_graph.png")

    print(f"[OK] Created {args.outzip} with stage1.txt, stage2.txt, NLACP.txt, MESP.txt, gpt_attribute.txt, attributes_table.csv, attributes_graph.png")

if __name__ == "__main__":
    main()

