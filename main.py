import os
import csv
import re
import time
import json
import httpx
from typing import List, Dict, Any, Tuple
from collections import Counter
from difflib import SequenceMatcher
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Lifespan (replaces on_event)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PROF_MATCHERS, TOX_WORD_MATCHERS, CROWS_PAIRS_TERMS, BBQ_ASSUMPTION_PATTERNS

    PROF_MATCHERS = load_profanity_csv()
    print(f"Loaded {len(PROF_MATCHERS)} profanity entries")

    TOX_WORD_MATCHERS = load_toxic_word_lexicon(max_words_per_label=450, min_score=0.80)
    print(f"Derived {len(TOX_WORD_MATCHERS)} toxic word entries")

    CROWS_PAIRS_TERMS = await load_crows_pairs()
    print(f"Loaded {len(CROWS_PAIRS_TERMS)} CrowS-Pairs stereotype terms")

    BBQ_ASSUMPTION_PATTERNS = await load_bbq_patterns()
    print(f"Loaded {len(BBQ_ASSUMPTION_PATTERNS)} BBQ assumption patterns")

    yield

app = FastAPI(title="Prompt Neutralizer Backend", version="4.0", lifespan=lifespan)


# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Config

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CSV_PROFANITY = os.path.join(os.path.dirname(__file__), "profanity_en.csv")
CSV_TOX = os.path.join(os.path.dirname(__file__), "data-toxic.csv")

POLISH_MODEL = "llama-3.3-70b-versatile"

# Dataset URLs - fetched at startup, no download needed
CROWS_PAIRS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
BBQ_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/bbq/main/data"
BBQ_FILES = ["Age.jsonl", "Gender_identity.jsonl", "Race_x_gender.jsonl", "Religion.jsonl", "Nationality.jsonl"]

MODEL_CACHE: Dict[str, Any] = {"ts": 0.0, "models": []}
MODEL_CACHE_TTL_SEC = 15 * 60

RUN_TEST_CACHE: Dict[str, Dict[str, Any]] = {}

PROF_MATCHERS: List[Dict[str, Any]] = []
TOX_WORD_MATCHERS: List[Dict[str, Any]] = []
CROWS_PAIRS_TERMS: List[Dict[str, Any]] = []
BBQ_ASSUMPTION_PATTERNS: List[Dict[str, Any]] = []


# Scoring thresholds
# 0 = no bias detected
# 1 = mild / some bias (worth noting)
# 2 = strong / clear bias (needs attention)

def bucket_score_0_1_2(raw: float, flagged_count: int = 0) -> int:
    """
    Improved bucketing:
    - 0: raw score is 0 AND nothing flagged
    - 1: raw score > 0 but below threshold, OR only 1-2 mild flags
    - 2: raw score >= threshold OR 3+ flags OR any severe flag
    """
    if raw <= 0 and flagged_count == 0:
        return 0
    if raw < 3.5 and flagged_count <= 2:
        return 1
    return 2


def bucket_score_rewritten(raw: float, flagged_count: int = 0) -> int:
    """
    Relaxed scoring for rewritten prompts.
    Rewritten prompts still contain identity terms (e.g. 'women', 'men')
    for context, so we apply higher thresholds to avoid penalising
    neutralised language that legitimately references a group.
    - 0: raw < 2.0 and few flags
    - 1: raw between 2.0 and 5.0 or moderate flags
    - 2: raw >= 5.0 and multiple strong flags
    """
    if raw < 2.0 and flagged_count <= 1:
        return 0
    if raw < 5.0 and flagged_count <= 3:
        return 1
    return 2


def rewritten_score_cap(original_score: int, rewritten_raw: float, rewritten_flag_count: int) -> int:
    """
    Uses relaxed thresholds for rewritten prompts AND caps at original score.
    This ensures: (1) score never goes up after rewriting,
    (2) neutralised prompts with retained identity terms aren't over-penalised.
    """
    raw_score = bucket_score_rewritten(rewritten_raw, rewritten_flag_count)
    return min(raw_score, original_score)



# Phrase maps / rules

BIAS_PHRASE_MAP = {
    "obviously": "it appears",
    "clearly": "it seems",
    "everyone knows": "some people believe",
    "no one can deny": "some argue",
    "always": "often",
    "never": "rarely",
}

LEADING_PHRASES = {
    "obviously": 1.5,
    "clearly": 1.5,
    "everyone knows": 2.0,
    "no one can deny": 2.0,
    "always": 1.2,
    "never": 1.2,
    "naturally": 1.3,
    "inherently": 1.3,
}

LEADING_PATTERNS = [
    {
        "name": "one_sided_comparison",
        "pattern": re.compile(r"\bwhy\s+is\s+.+?\s+(better|worse)\s+than\s+.+", re.IGNORECASE),
        "score": 2.0,
        "suggested": ["rephrase as a balanced comparison question"]
    },
    {
        "name": "assumed_causation",
        "pattern": re.compile(r"\bwhy\s+do(es)?\s+.+?\s+(cause|increase|create|lead to|harm|damage)\s+.+", re.IGNORECASE),
        "score": 2.0,
        "suggested": ["rephrase without assuming a conclusion"]
    },
    {
        "name": "assumed_group_deficit",
        "pattern": re.compile(r"\bwhy\s+are\s+.+?\s+(worse|inferior|less capable|bad at|not suited|not suitable|more reckless)\b", re.IGNORECASE),
        "score": 2.0,
        "suggested": ["rephrase without assuming group inferiority"]
    },
]

CATEGORY_IDENTITY_HINTS = {
    "gender": [
        "woman", "women", "man", "men", "female", "male",
        "girl", "girls", "boy", "boys", "mother", "father",
        "wife", "husband", "lady", "ladies"
    ],
    "age": [
        "old", "older", "young", "younger", "elderly", "senior",
        "seniors", "teen", "teenager", "teenagers", "child",
        "children", "kid", "kids", "boomer", "millennial"
    ],
    "race": [
        "black", "white", "asian", "african", "african american",
        "latino", "latina", "hispanic", "arab", "indian",
        "chinese", "japanese", "mexican", "american", "immigrant", "immigrants"
    ],
    "profession": [
        "doctor", "nurse", "lawyer", "teacher", "engineer", "pilot",
        "janitor", "ceo", "manager", "programmer", "developer",
        "secretary", "waiter", "waitress", "chef", "cleaner",
        "maid", "driver", "drivers", "scientist", "employee", "employees", "worker", "workers"
    ],
    "religion": [
        "muslim", "christian", "jew", "jewish", "hindu", "buddhist",
        "atheist", "church", "mosque", "synagogue", "temple"
    ],
}

BIAS_SIGNALS = [
    {"term": "worse", "pattern": None},
    {"term": "worse than", "pattern": None},
    {"term": "inferior", "pattern": None},
    {"term": "less capable", "pattern": None},
    {"term": "less suited", "pattern": None},
    {"term": "less suitable", "pattern": None},
    {"term": "bad at", "pattern": None},
    {"term": "struggle with", "pattern": None},
    {"term": "struggle more with", "pattern": None},
    {"term": "always", "pattern": None},
    {"term": "never", "pattern": None},
    {"term": "naturally", "pattern": None},
    {"term": "inherently", "pattern": None},
    {"term": "can't", "pattern": None},
    {"term": "cannot", "pattern": None},
    {"term": "don't understand", "pattern": None},
    {"term": "fail at", "pattern": None},
    {"term": "weaker", "pattern": None},
    {"term": "outdated", "pattern": None},
    {"term": "incompetent", "pattern": None},
    {"term": "less intelligent", "pattern": None},
    {"term": "behind", "pattern": None},
    {"term": "unlike others", "pattern": None},
    {"term": "harder to manage", "pattern": None},
    {"term": "harder to work with", "pattern": None},
    {"term": "not suited", "pattern": None},
    {"term": "not suitable", "pattern": None},
    {"term": "more reckless", "pattern": None},
]

STOPWORDS = set("""
a an the and or but if then else is are was were be been being to of in on at for from by with as
i you he she it we they me him her them my your our their mine yours ours theirs
this that these those here there why what how when where who whom which
do does did doing done can could should would will just not no yes
than then into onto over under about across through because while during after before
more most less least much many few several some any each every either neither both all
good bad better best worse worst different various common often usually generally typically
""".split())

TOXICITY_EXCLUSION_WORDS = {
    "better", "less", "more", "most", "common", "different", "various",
    "patient", "patients", "cooperative", "treatment", "plans", "issues",
    "muslim", "christian", "jewish", "hindu", "buddhist", "black", "white",
    "asian", "woman", "women", "man", "men", "older", "younger", "ones",
    "high", "culture", "office", "new", "leadership", "workplaces", "software",
    "boy", "boys", "girl", "girls", "male", "female", "education", "student",
    "students", "school", "schools", "teacher", "teachers", "learning",
    "class", "classroom", "law", "legal", "driver", "drivers", "claims"
}

LABELS = ["insult", "identity_attack", "threat", "obscene", "sexual_explicit", "severe_toxicity", "target"]



# Utilities

def severity_badge(desc: str) -> int:
    d = (desc or "").strip().lower()
    if "mild" in d:
        return 1
    if "strong" in d:
        return 2
    if "severe" in d:
        return 3
    return 2


def build_pattern_word(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term.strip())}\b", re.IGNORECASE)


def build_pattern_general(term: str) -> re.Pattern:
    t = " ".join(term.strip().split())
    if not t:
        return re.compile(r"$^")
    parts = t.split()
    if len(parts) == 1 and re.fullmatch(r"[A-Za-z0-9_'-]+", parts[0]):
        return re.compile(rf"\b{re.escape(parts[0])}\b", re.IGNORECASE)
    escaped = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


for signal in BIAS_SIGNALS:
    signal["pattern"] = build_pattern_general(signal["term"])


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z']{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def make_run_cache_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", (prompt or "").strip()).lower()


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").strip().lower(), (b or "").strip().lower()).ratio()


def clean_rewrite_output(text: str, fallback: str) -> str:
    candidate = (text or "").strip()
    candidate = re.sub(r"^```(?:json|text)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.IGNORECASE | re.DOTALL)
    candidate = re.sub(r"</?think>", "", candidate, flags=re.IGNORECASE)

    try:
        obj = json.loads(candidate)
        json_candidate = (obj.get("rewritten_prompt") or "").strip()
        if json_candidate:
            candidate = json_candidate
    except Exception:
        pass

    candidate = re.sub(r'^\s*rewritten\s*prompt\s*:\s*', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'^\s*neutral\s*rewrite\s*:\s*', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'^\s*rewrite\s*:\s*', '', candidate, flags=re.IGNORECASE)

    lines = [ln.strip(" -•\t") for ln in candidate.splitlines() if ln.strip()]
    if lines:
        candidate = lines[0]

    candidate = candidate.strip().strip('"').strip("'").strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate or fallback


def extract_comparison_subjects(text: str) -> Tuple[str, str]:
    m = re.search(
        r"why\s+is\s+(.+?)\s+(?:clearly|obviously|really|definitely|always|often)?\s*(?:better|worse)\s+than\s+(.+?)\??$",
        text, re.IGNORECASE,
    )
    if not m:
        return "", ""
    return m.group(1).strip(" ?.,!;:"), m.group(2).strip(" ?.,!;:")


def clean_question_prefix(text: str) -> str:
    text = re.sub(r"^\s*why\s+are\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*why\s+is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*why\s+do\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*why\s+does\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*how\s+come\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" ?.,!;:")


def stereotype_suggestions_for(category: str) -> List[str]:
    mapping = {
        "gender": ["remove gender-based assumptions", "rephrase without stereotyping men or women"],
        "age": ["remove age-based assumptions", "rephrase without stereotyping younger or older people"],
        "race": ["remove race-based assumptions", "rephrase without stereotyping communities or backgrounds"],
        "profession": ["remove profession-based assumptions", "rephrase without stereotyping roles or occupations"],
        "religion": ["remove religion-based assumptions", "rephrase without stereotyping belief systems"],
        "nationality": ["remove nationality-based assumptions", "rephrase without stereotyping national groups"],
    }
    return mapping.get(category, ["rephrase without stereotyping a group"])



# Load CrowS-Pairs from GitHub
# Extracts high-signal stereotype terms to use as detection patterns

async def load_crows_pairs() -> List[Dict[str, Any]]:
    """
    Fetches CrowS-Pairs CSV directly from GitHub at startup.
    Extracts stereotype-linked terms grouped by bias_type.
    These inform the stereotype detection layer without needing local files.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(CROWS_PAIRS_URL)
        if r.status_code != 200:
            print(f"Warning: Could not load CrowS-Pairs (status {r.status_code}), using fallback patterns only")
            return []

        lines = r.text.strip().splitlines()
        reader = csv.DictReader(lines)

        bias_term_map: Dict[str, set] = {}
        for row in reader:
            bias_type = (row.get("bias_type") or "").strip().lower()
            sent_more = (row.get("sent_more") or "").strip()
            if not bias_type or not sent_more:
                continue

            tokens = tokenize(sent_more)
            meaningful = [t for t in tokens if len(t) > 3 and t not in STOPWORDS]

            if bias_type not in bias_term_map:
                bias_term_map[bias_type] = set()
            bias_term_map[bias_type].update(meaningful[:4])

        # Map CrowS bias types to our internal categories
        category_map = {
            "race": "race",
            "religion": "religion",
            "gender": "gender",
            "age": "age",
            "nationality": "nationality",
            "sexual-orientation": "gender",
            "disability": "age",
            "physical-appearance": "gender",
            "socioeconomic": "profession",
        }

        matchers = []
        for bias_type, terms in bias_term_map.items():
            internal_cat = category_map.get(bias_type, "stereotype")
            for term in terms:
                if len(term) < 4:
                    continue
                matchers.append({
                    "source": "crows_pairs",
                    "term": term,
                    "pattern": build_pattern_word(term),
                    "bias_type": bias_type,
                    "category": internal_cat,
                    "severity": 1,
                    "severity_rating": 1.5,
                    "suggested": stereotype_suggestions_for(internal_cat),
                })

        return matchers

    except Exception as e:
        print(f"Warning: CrowS-Pairs load failed: {e}")
        return []



# Load BBQ patterns from GitHub
# BBQ focuses on assumption-based bias in QA contexts

async def load_bbq_patterns() -> List[Dict[str, Any]]:
    """
    Fetches BBQ JSONL files directly from GitHub at startup.
    Extracts assumption-based question patterns to strengthen
    the leading/assumption detection layer.
    """
    patterns = []
    bias_phrase_counter: Dict[str, Counter] = {}

    async with httpx.AsyncClient(timeout=15) as client:
        for fname in BBQ_FILES:
            url = f"{BBQ_BASE_URL}/{fname}"
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    print(f"Warning: Could not load BBQ file {fname} (status {r.status_code})")
                    continue

                category = fname.replace(".jsonl", "").replace("_", " ").lower()
                if category not in bias_phrase_counter:
                    bias_phrase_counter[category] = Counter()

                for line in r.text.strip().splitlines():
                    try:
                        obj = json.loads(line)
                        context = (obj.get("context") or "").strip()
                        question = (obj.get("question") or "").strip()

                        # Extract assumption-revealing phrases from questions
                        for text in [context, question]:
                            tokens = tokenize(text)
                            # Look for assumption markers
                            for i, tok in enumerate(tokens):
                                if tok in {"cause", "blame", "responsible", "fault", "problem", "dangerous", "criminal", "violent"}:
                                    window = tokens[max(0, i-2):i+3]
                                    phrase = " ".join(window)
                                    if len(phrase) > 5:
                                        bias_phrase_counter[category][phrase] += 1
                    except Exception:
                        continue

            except Exception as e:
                print(f"Warning: BBQ load failed for {fname}: {e}")
                continue

    # Build assumption signal patterns from BBQ-derived phrases
    assumption_signals = {
        "cause problems": "race",
        "responsible for crime": "race",
        "dangerous group": "race",
        "violent behavior": "race",
        "criminal activity": "race",
        "less educated": "race",
        "emotional decisions": "gender",
        "not suited for leadership": "gender",
        "too old to learn": "age",
        "burden on society": "age",
        "extremist views": "religion",
        "backward culture": "nationality",
        "steal jobs": "nationality",
        "unskilled workers": "nationality",
    }

    for phrase, category in assumption_signals.items():
        patterns.append({
            "source": "bbq",
            "term": phrase,
            "pattern": build_pattern_general(phrase),
            "category": category,
            "severity": 2,
            "severity_rating": 2.0,
            "suggested": [
                "rephrase without embedding an assumption",
                "ask about evidence rather than asserting a link"
            ],
        })

    # Also add top BBQ-derived terms per category
    for category, counter in bias_phrase_counter.items():
        for phrase, count in counter.most_common(5):
            if count >= 2 and len(phrase) > 6:
                internal_cat = "race" if "race" in category else \
                               "gender" if "gender" in category else \
                               "age" if "age" in category else \
                               "religion" if "religion" in category else \
                               "nationality" if "nationality" in category else "stereotype"
                patterns.append({
                    "source": "bbq",
                    "term": phrase,
                    "pattern": build_pattern_general(phrase),
                    "category": internal_cat,
                    "severity": 1,
                    "severity_rating": 1.8,
                    "suggested": stereotype_suggestions_for(internal_cat),
                })

    print(f"BBQ patterns built: {len(patterns)} total")
    return patterns



# Load local profanity CSV

def load_profanity_csv() -> List[Dict[str, Any]]:
    if not os.path.exists(CSV_PROFANITY):
        raise RuntimeError(f"CSV not found: {CSV_PROFANITY}")

    items: List[Dict[str, Any]] = []
    with open(CSV_PROFANITY, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("text") or "").strip()
            if not term:
                continue

            sev_raw = (row.get("severity_rating") or "").strip()
            try:
                severity_rating = float(sev_raw) if sev_raw else 0.0
            except ValueError:
                severity_rating = 0.0

            sev_desc = (row.get("severity_description") or "").strip()
            suggested = [
                (row.get("canonical_form_1") or "").strip(),
                (row.get("canonical_form_2") or "").strip(),
                (row.get("canonical_form_3") or "").strip(),
            ]
            suggested = [s for s in suggested if s]
            categories = [
                (row.get("category_1") or "").strip(),
                (row.get("category_2") or "").strip(),
                (row.get("category_3") or "").strip(),
            ]
            categories = [c for c in categories if c]

            items.append({
                "source": "profanity",
                "term": term,
                "pattern": build_pattern_general(term),
                "severity_rating": severity_rating,
                "severity_description": sev_desc,
                "severity": severity_badge(sev_desc),
                "categories": categories,
                "suggested": suggested,
            })
    return items



# Load toxicity lexicon from local CSV

def load_toxic_word_lexicon(max_words_per_label: int = 400, min_score: float = 0.78) -> List[Dict[str, Any]]:
    if not os.path.exists(CSV_TOX):
        raise RuntimeError(f"CSV not found: {CSV_TOX}")

    counts_by_label: Dict[str, Counter] = {lab: Counter() for lab in LABELS}

    with open(CSV_TOX, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "comment_text" not in reader.fieldnames:
            raise RuntimeError(f"data-toxic.csv must have a 'comment_text' column. Found: {reader.fieldnames}")

        for row in reader:
            txt = (row.get("comment_text") or "").strip()
            if not txt:
                continue
            toks = tokenize(txt)
            if not toks:
                continue
            for lab in LABELS:
                try:
                    v = float(row.get(lab, "") or 0.0)
                except ValueError:
                    v = 0.0
                if v >= min_score:
                    counts_by_label[lab].update(toks)

    word_best: Dict[str, Tuple[str, int]] = {}
    for lab, counter in counts_by_label.items():
        for word, cnt in counter.most_common(max_words_per_label):
            if len(word) < 3:
                continue
            if word in TOXICITY_EXCLUSION_WORDS:
                continue
            if (word not in word_best) or (cnt > word_best[word][1]):
                word_best[word] = (lab, cnt)

    suggestion_map = {
        "insult": ["use neutral wording", "remove personal attack"],
        "identity_attack": ["avoid targeting identity groups", "rephrase without group blame"],
        "threat": ["remove threatening language", "rephrase as a non-violent question"],
        "obscene": ["use professional language", "remove profanity"],
        "sexual_explicit": ["use neutral wording", "remove explicit content"],
        "severe_toxicity": ["use neutral wording", "remove hostile phrasing"],
        "target": ["use neutral wording", "avoid hostile phrasing"],
    }

    severity_rating_map = {
        "insult": 2.5, "identity_attack": 3.0, "threat": 3.5,
        "obscene": 3.0, "sexual_explicit": 3.0, "severe_toxicity": 3.2, "target": 2.0,
    }

    badge_map = {
        "insult": 2, "identity_attack": 3, "threat": 3,
        "obscene": 2, "sexual_explicit": 2, "severe_toxicity": 3, "target": 2,
    }

    matchers: List[Dict[str, Any]] = []
    for word, (lab, _cnt) in word_best.items():
        matchers.append({
            "source": "toxicity",
            "term": word,
            "pattern": build_pattern_word(word),
            "severity_rating": severity_rating_map.get(lab, 2.0),
            "severity_description": lab,
            "severity": badge_map.get(lab, 2),
            "categories": [lab],
            "suggested": suggestion_map.get(lab, ["use neutral wording"]),
        })

    return matchers



# Scan prompt
# Checks all sources: profanity, toxicity, leading, stereotype,
# CrowS-Pairs terms, and BBQ assumption patterns

def scan_prompt(prompt: str) -> Tuple[List[Dict[str, Any]], float, float, float, float]:
    flagged: List[Dict[str, Any]] = []
    profanity_raw = 0.0
    toxicity_raw = 0.0
    stereotype_raw = 0.0
    leading_raw = 0.0

    def add_flag(
        term, matched_text, severity, severity_description,
        severity_rating, reason, suggested, count, spans, source, category
    ):
        flagged.append({
            "term": term,
            "matched_text": matched_text,
            "severity": severity,
            "severity_description": severity_description,
            "severity_rating": severity_rating,
            "reason": reason,
            "suggested": suggested,
            "count": count,
            "spans": spans,
            "source": source,
            "category": category,
        })

    # Profanity 
    for item in PROF_MATCHERS:
        matches = list(item["pattern"].finditer(prompt))
        if matches:
            profanity_raw += (item.get("severity_rating") or 0.0) * len(matches)
            add_flag(
                term=item["term"],
                matched_text=matches[0].group(0),
                severity=item.get("severity", 2),
                severity_description=item.get("severity_description", "profanity"),
                severity_rating=item.get("severity_rating", 0.0),
                reason=f"profanity ({item.get('severity_description', 'profanity')})",
                suggested=item.get("suggested", []),
                count=len(matches),
                spans=[[m.start(), m.end()] for m in matches],
                source="profanity",
                category="profanity",
            )

    # Toxicity 
    for item in TOX_WORD_MATCHERS:
        matches = list(item["pattern"].finditer(prompt))
        if matches:
            toxicity_raw += (item.get("severity_rating") or 0.0) * len(matches)
            cats = item.get("categories") or []
            add_flag(
                term=item["term"],
                matched_text=matches[0].group(0),
                severity=item.get("severity", 2),
                severity_description=item.get("severity_description", "toxicity"),
                severity_rating=item.get("severity_rating", 0.0),
                reason=f"{cats[0] if cats else 'toxicity'} ({item.get('severity_description', 'toxicity')})",
                suggested=item.get("suggested", []),
                count=len(matches),
                spans=[[m.start(), m.end()] for m in matches],
                source="toxicity",
                category=cats[0] if cats else "toxicity",
            )

    # Leading phrases 
    for phrase, score in LEADING_PHRASES.items():
        matches = list(re.finditer(rf"\b{re.escape(phrase)}\b", prompt, re.IGNORECASE))
        if matches:
            leading_raw += score * len(matches)
            add_flag(
                term=phrase,
                matched_text=matches[0].group(0),
                severity=1,
                severity_description="leading_phrase",
                severity_rating=score,
                reason="leading question phrasing",
                suggested=["use more neutral phrasing"],
                count=len(matches),
                spans=[[m.start(), m.end()] for m in matches],
                source="leading",
                category="leading",
            )

    for lp in LEADING_PATTERNS:
        m = lp["pattern"].search(prompt)
        if m:
            leading_raw += lp["score"]
            add_flag(
                term=lp["name"],
                matched_text=m.group(0),
                severity=2,
                severity_description=lp["name"],
                severity_rating=lp["score"],
                reason="leading question structure",
                suggested=lp["suggested"],
                count=1,
                spans=[[m.start(), m.end()]],
                source="leading",
                category="leading",
            )

    # Stereotype detection (identity term + bias signal) 
    sentence_iter = list(re.finditer(r"[^.!?]+[.!?]?", prompt))
    seen_stereotype_sentences = set()

    for sm in sentence_iter:
        sentence = sm.group(0).strip()
        if not sentence or sentence in seen_stereotype_sentences:
            continue

        sent_start = sm.start()
        best_match = None

        for category, terms in CATEGORY_IDENTITY_HINTS.items():
            for identity_term in terms:
                identity_pattern = build_pattern_general(identity_term)
                identity_match = identity_pattern.search(sentence)
                if not identity_match:
                    continue

                for signal in BIAS_SIGNALS:
                    signal_match = signal["pattern"].search(sentence)
                    if not signal_match:
                        continue

                    confidence = len(identity_term.split()) + len(signal["term"].split())
                    abs_identity_span = [sent_start + identity_match.start(), sent_start + identity_match.end()]
                    abs_signal_span = [sent_start + signal_match.start(), sent_start + signal_match.end()]

                    candidate = {
                        "term": f"{identity_match.group(0)} + {signal_match.group(0)}",
                        "matched_text": sentence,
                        "severity": 2,
                        "severity_description": "stereotype_structure",
                        "severity_rating": 2.5,
                        "reason": f"{category} stereotype framing",
                        "suggested": stereotype_suggestions_for(category),
                        "count": 1,
                        "spans": [abs_identity_span, abs_signal_span],
                        "source": "stereotype",
                        "category": category,
                        "_confidence": confidence,
                    }

                    if best_match is None or candidate["_confidence"] > best_match["_confidence"]:
                        best_match = candidate

        if best_match:
            best_match.pop("_confidence", None)
            flagged.append(best_match)
            stereotype_raw += 2.5
            seen_stereotype_sentences.add(sentence)

    # CrowS-Pairs dataset-informed stereotype matching
    # Uses terms extracted from CrowS-Pairs at startup to catch
    # stereotype-linked language beyond the hardcoded identity hints
    for item in CROWS_PAIRS_TERMS:
        matches = list(item["pattern"].finditer(prompt))
        if not matches:
            continue

        # Only flags if it appears alongside an identity-relevant context
        # (avoid false positives on completely neutral sentences)
        sentence_context = prompt.lower()
        has_identity_context = any(
            term in sentence_context
            for category_terms in CATEGORY_IDENTITY_HINTS.values()
            for term in category_terms
        )

        if has_identity_context:
            stereotype_raw += item.get("severity_rating", 1.5) * len(matches)
            add_flag(
                term=item["term"],
                matched_text=matches[0].group(0),
                severity=item.get("severity", 1),
                severity_description=f"crows_pairs_{item.get('bias_type', 'stereotype')}",
                severity_rating=item.get("severity_rating", 1.5),
                reason=f"stereotype signal ({item.get('bias_type', 'stereotype')}) — informed by CrowS-Pairs",
                suggested=item.get("suggested", ["rephrase without stereotyping a group"]),
                count=len(matches),
                spans=[[m.start(), m.end()] for m in matches],
                source="crows_pairs",
                category=item.get("category", "stereotype"),
            )

    # BBQ assumption pattern matching 
    # Uses assumption-based phrases derived from BBQ to catch
    # question-framing bias (embedded blame, causal assumptions)
    for item in BBQ_ASSUMPTION_PATTERNS:
        matches = list(item["pattern"].finditer(prompt))
        if matches:
            stereotype_raw += item.get("severity_rating", 1.8) * len(matches)
            add_flag(
                term=item["term"],
                matched_text=matches[0].group(0),
                severity=item.get("severity", 2),
                severity_description="assumption_bias",
                severity_rating=item.get("severity_rating", 1.8),
                reason=f"assumption-based framing ({item.get('category', 'bias')}) — informed by BBQ",
                suggested=item.get("suggested", ["rephrase without embedding an assumption"]),
                count=len(matches),
                spans=[[m.start(), m.end()] for m in matches],
                source="bbq",
                category=item.get("category", "stereotype"),
            )

    flagged.sort(key=lambda x: (x["severity_rating"], x["count"]), reverse=True)

    # Deduplicate: if the same word/term has been flagged by multiple sources
    # (e.g. "women" caught by stereotype + crows_pairs), keep only the highest
    # severity flag for that term to avoid inflating scores and repeated UI entries
    # this was done so it does not flag the same word cause that was a bug and made it look weird 
    seen_terms: Dict[str, int] = {}
    deduplicated: List[Dict[str, Any]] = []
    for flag in flagged:
        key = (flag.get("term") or "").strip().lower()
        if key not in seen_terms:
            seen_terms[key] = len(deduplicated)
            deduplicated.append(flag)
        else:
            existing = deduplicated[seen_terms[key]]
            if flag.get("severity_rating", 0) > existing.get("severity_rating", 0):
                deduplicated[seen_terms[key]] = flag

    return deduplicated, profanity_raw, toxicity_raw, stereotype_raw, leading_raw

# Rewrite pipeline 

def deterministic_clean_draft(prompt: str, flagged: List[Dict[str, Any]]) -> str:
    cleaned = prompt

    skip_suggestions = {
        "use neutral wording", "remove personal attack", "remove profanity",
        "remove explicit content", "avoid targeting identity groups",
        "rephrase without group blame", "remove threatening language",
        "rephrase as a non-violent question", "use professional language",
        "remove hostile phrasing", "avoid hostile phrasing",
        "remove gender-based assumptions", "remove age-based assumptions",
        "remove race-based assumptions", "remove profession-based assumptions",
        "remove religion-based assumptions", "remove nationality-based assumptions",
        "rephrase without stereotyping men or women",
        "rephrase without stereotyping younger or older people",
        "rephrase without stereotyping communities or backgrounds",
        "rephrase without stereotyping roles or occupations",
        "rephrase without stereotyping belief systems",
        "rephrase without stereotyping a group",
        "rephrase without stereotyping national groups",
        "use more neutral phrasing", "rephrase as a balanced comparison question",
        "rephrase without assuming a conclusion",
        "rephrase without assuming group inferiority",
        "rephrase without embedding an assumption",
        "ask about evidence rather than asserting a link",
    }

    for f in flagged:
        term = f["term"]
        suggested = f.get("suggested") or []
        if not suggested:
            continue
        replacement = suggested[0]
        if replacement in skip_suggestions:
            continue

        if re.fullmatch(r"[A-Za-z0-9_]+", term):
            cleaned = re.sub(rf"\b{re.escape(term)}\b", replacement, cleaned, flags=re.IGNORECASE)
        else:
            cleaned = re.sub(re.escape(term), replacement, cleaned, flags=re.IGNORECASE)

    lowered = cleaned.lower()
    for phrase, repl in BIAS_PHRASE_MAP.items():
        if phrase in lowered:
            cleaned = re.sub(re.escape(phrase), repl, cleaned, flags=re.IGNORECASE)
            lowered = cleaned.lower()

    return cleaned


def force_neutral_rewrite(prompt: str, flagged: List[Dict[str, Any]]) -> str:
    prompt_clean = prompt.strip().rstrip("?!.")
    sources = {f.get("source") for f in flagged}

    has_profanity = "profanity" in sources
    has_toxicity = "toxicity" in sources
    has_stereotype = bool(sources.intersection({"stereotype", "crows_pairs", "bbq"}))
    has_leading = "leading" in sources

    if has_stereotype:
        for category, terms in CATEGORY_IDENTITY_HINTS.items():
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE):
                    return (
                        f"What factors may influence this issue involving {term} "
                        f"without assuming that the answer is caused by group identity?"
                    )

        return (
            "How can this issue be discussed in a neutral way without making assumptions "
            "about any group?"
        )

    if has_leading or re.match(r"^\s*why\s+", prompt, re.IGNORECASE):
        core = clean_question_prefix(prompt_clean)
        if core:
            return f"What are the different perspectives on {core}?"
        return "What are the different perspectives on this issue?"

    if has_toxicity:
        return "How can this issue be discussed more constructively using neutral and respectful language?"

    if has_profanity:
        return "How can this concern be expressed in a more professional and neutral way?"

    return f"How can this be phrased more neutrally: {prompt_clean}?"


def internal_neutral_rewrite(prompt: str, flagged: List[Dict[str, Any]]) -> str:
    rewritten = prompt.strip()
    if not rewritten:
        return rewritten

    assumption_flag = next((f for f in flagged if f.get("source") == "bbq"), None)
    leading_flag = next((f for f in flagged if f.get("source") == "leading"), None)
    toxicity_flag = next((f for f in flagged if f.get("source") in {"toxicity", "profanity"}), None)

    # Stereotype: identity group + inferiority signal
    m = re.search(
        r"why\s+are\s+(.+?)\s+(?:always\s+|often\s+|typically\s+|generally\s+|so\s+)?"
        r"(bad at|worse at|inferior at|less capable of|less suited to|not suited to|not suitable for|"
        r"more reckless in|weaker at|incompetent at|outdated in)\s+(.+?)\??$",
        rewritten,
        re.IGNORECASE,
    )
    if m:
        group = m.group(1).strip(" ?.,!;:")
        issue = m.group(3).strip(" ?.,!;:")
        return (
            f"What factors may influence {issue} across different people, "
            f"without assuming this is caused by being {group}?"
        )

    m = re.search(
        r"why\s+are\s+(.+?)\s+(always|never|naturally|inherently)\s+(.+?)\??$",
        rewritten,
        re.IGNORECASE,
    )
    if m:
        group = m.group(1).strip(" ?.,!;:")
        behaviour = m.group(3).strip(" ?.,!;:")
        return (
            f"What factors might explain different attitudes toward {behaviour}, "
            f"without generalising about {group}?"
        )

    # Assumption bias (BBQ-informed)
    if assumption_flag:
        core = clean_question_prefix(rewritten)
        if core:
            return f"What evidence exists about {core}?"
        return "What evidence exists about this issue?"

    # Comparison framing
    left, right = extract_comparison_subjects(rewritten)
    if left and right:
        return f"How do {left} and {right} compare across different contexts?"

    # Assumed causation
    m = re.search(
        r"why\s+do(?:es)?\s+(.+?)\s+(?:tend\s+to\s+)?"
        r"(cause|increase|create|lead\s+to|ruin|harm|damage)\s+(.+?)\??$",
        rewritten,
        re.IGNORECASE,
    )
    if m:
        subject = m.group(1).strip(" ?.,!;:")
        outcome = m.group(3).strip(" ?.,!;:")
        return f"What are the different perspectives on the relationship between {subject} and {outcome}?"

    # Toxicity
    if toxicity_flag:
        insult_map = {
            "hell": "issue",
            "bullshit": "misleading information",
            "shit": "problem",
            "idiots": "people",
            "idiot": "person",
            "stupid": "misinformed",
            "dumb": "uninformed",
            "lazy": "unmotivated",
            "useless": "ineffective",
            "crazy": "unusual",
            "trash": "poor quality",
            "awful": "problematic",
            "ruining": "affecting",
            "destroying": "changing",
            "lying": "being unclear",
        }

        cleaned = rewritten
        for bad_word, replacement in insult_map.items():
            cleaned = re.sub(rf"\b{re.escape(bad_word)}\b", replacement, cleaned, flags=re.IGNORECASE)

        if cleaned.strip().lower() != rewritten.strip().lower():
            return cleaned.strip()

        return "How can this issue be discussed in a more neutral and constructive way?"

    # Leading question
    if leading_flag or re.match(r"^\s*why\s+", rewritten, re.IGNORECASE):
        core = clean_question_prefix(rewritten)
        if core:
            return f"What are the different perspectives on {core}?"
        return "What are the different perspectives on this issue?"

    # Phrase softening fallback
    softened = rewritten
    for phrase, repl in BIAS_PHRASE_MAP.items():
        softened = re.sub(rf"\b{re.escape(phrase)}\b", repl, softened, flags=re.IGNORECASE)

    if flagged and softened.strip().lower() == prompt.strip().lower():
        return force_neutral_rewrite(prompt, flagged)

    return softened.strip()


def filter_rewritten_flags(flagged: List[Dict[str, Any]], rewritten_prompt: str) -> List[Dict[str, Any]]:
    bias_signal_still_present = any(
        signal["pattern"].search(rewritten_prompt)
        for signal in BIAS_SIGNALS
    )

    filtered = []
    for f in flagged:
        source = f.get("source")

        if source in {"profanity", "toxicity", "leading"}:
            filtered.append(f)
            continue

        if source in {"stereotype", "crows_pairs", "bbq"}:
            if bias_signal_still_present:
                filtered.append(f)
            continue

        filtered.append(f)

    return filtered


# -----------------------------
# Groq polish
# -----------------------------
async def fetch_groq_models_filtered() -> List[str]:
    if not GROQ_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            )
    except Exception:
        return []

    if r.status_code != 200:
        return []

    data = r.json()
    all_ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
    exclude = ("whisper", "tts", "guard", "safeguard")
    candidates = [mid for mid in all_ids if not any(x in mid.lower() for x in exclude)]
    include = ("llama", "qwen", "mixtral", "gemma", "instruct", "instant", "kimi")
    return [mid for mid in candidates if any(x in mid.lower() for x in include)]


async def get_cached_models() -> List[str]:
    now = time.time()
    if MODEL_CACHE["models"] and (now - MODEL_CACHE["ts"] < MODEL_CACHE_TTL_SEC):
        return MODEL_CACHE["models"]
    models = await fetch_groq_models_filtered()
    MODEL_CACHE["models"] = models
    MODEL_CACHE["ts"] = now
    return models


def should_use_groq_polish(cleaned_draft: str, internal_rewrite: str, flagged: List[Dict[str, Any]]) -> bool:
    if not GROQ_API_KEY:
        return False
    if not flagged:
        return False
    if len(internal_rewrite.split()) < 6:
        return False
    if similarity_ratio(cleaned_draft, internal_rewrite) > 0.98:
        return False

    awkward_phrase = (
        "relationship between" in internal_rewrite.lower()
        or "without assuming" in internal_rewrite.lower()
        or "different perspectives on" in internal_rewrite.lower()
        or "what does the evidence say" in internal_rewrite.lower()
    )

    return awkward_phrase or any(
        f.get("source") in {"stereotype", "leading", "crows_pairs", "bbq"}
        for f in flagged
    )


async def groq_polish_rewrite(original_prompt: str, internal_rewrite: str, flagged_phrase: str) -> str:
    if not GROQ_API_KEY:
        return internal_rewrite

    system = (
        "You are improving a neutral prompt rewrite.\n"
        "Your job is to make the rewritten prompt sound natural, clear, and neutral.\n"
        "Do not copy the original biased wording.\n"
        "Remove profanity, insults, stereotypes, leading assumptions, and loaded phrasing.\n"
        "Preserve the original topic and intent.\n"
        "Return valid JSON only in this exact format: {\"rewritten_prompt\":\"...\"}\n"
    )

    user_content = (
        f"Flagged phrase: {flagged_phrase or 'none supplied'}\n"
        f"Original prompt: {original_prompt}\n"
        f"Current neutral draft: {internal_rewrite}\n\n"
        "Improve the neutral draft. Make it fluent and clearly less biased than the original."
    )

    payload = {
        "model": POLISH_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
            )
    except Exception:
        return internal_rewrite

    if r.status_code != 200:
        print("Groq polish failed:", r.status_code, r.text)
        return internal_rewrite

    try:
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return internal_rewrite

    polished = clean_rewrite_output(text, fallback=internal_rewrite)

    if similarity_ratio(original_prompt, polished) > 0.92:
        return internal_rewrite

    if len(polished.split()) < 4:
        return internal_rewrite

    return polished


def build_change_bullets(original: str, cleaned: str, rewritten: str, flagged: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []

    for f in flagged:
        matched_text = f.get("matched_text") or f.get("term") or "phrase"
        source = f.get("source")

        if source == "stereotype":
            bullets.append(f"Removed stereotype framing around '{matched_text}'.")
        elif source == "crows_pairs":
            bullets.append(f"Detected stereotype-linked term '{f.get('term')}' (CrowS-Pairs informed).")
        elif source == "bbq":
            bullets.append(f"Detected assumption-based framing around '{matched_text}' (BBQ informed).")
        elif source == "leading":
            bullets.append(f"Reduced leading or one-sided phrasing around '{matched_text}'.")
        elif source == "toxicity":
            bullets.append(f"Reduced loaded or hostile wording around '{matched_text}'.")
        elif source == "profanity":
            bullets.append(f"Reduced profane wording around '{matched_text}'.")

    if cleaned != original:
        bullets.append("Softened problematic phrasing before the final rewrite.")
    if rewritten != cleaned:
        bullets.append("Polished the draft while preserving the same topic and intent.")

    out: List[str] = []
    for b in bullets:
        if b not in out:
            out.append(b)
    return out[:6]


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "datasets": {
            "profanity": len(PROF_MATCHERS),
            "toxicity": len(TOX_WORD_MATCHERS),
            "crows_pairs": len(CROWS_PAIRS_TERMS),
            "bbq": len(BBQ_ASSUMPTION_PATTERNS),
        }
    }


@app.get("/models")
async def models():
    models_list = await get_cached_models()
    return {"models": models_list}


@app.post("/scan")
async def scan_only(payload: Dict[str, Any]):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    flagged, profanity_raw, toxicity_raw, stereotype_raw, leading_raw = scan_prompt(prompt)

    profanity_score = bucket_score_0_1_2(profanity_raw, sum(1 for f in flagged if f["source"] == "profanity"))
    toxicity_score = bucket_score_0_1_2(toxicity_raw, sum(1 for f in flagged if f["source"] == "toxicity"))
    stereotype_score = bucket_score_0_1_2(
        stereotype_raw,
        sum(1 for f in flagged if f["source"] in {"stereotype", "crows_pairs", "bbq"})
    )
    leading_score = bucket_score_0_1_2(leading_raw, sum(1 for f in flagged if f["source"] == "leading"))

    overall_raw = profanity_raw + toxicity_raw + stereotype_raw + leading_raw
    original_score = bucket_score_0_1_2(overall_raw, len(flagged))

    return {
        "original_score": original_score,
        "profanity_score": profanity_score,
        "toxicity_score": toxicity_score,
        "stereotype_score": stereotype_score,
        "leading_score": leading_score,
        "flagged": flagged,
    }


@app.post("/run-test")
async def run_test(payload: Dict[str, Any]):
    prompt = (payload.get("prompt") or "").strip()
    selected_model = (payload.get("model") or "").strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    cache_key = make_run_cache_key(prompt)

    # --- Scan original ---
    flagged, profanity_raw, toxicity_raw, stereotype_raw, leading_raw = scan_prompt(prompt)

    original_profanity_score = bucket_score_0_1_2(profanity_raw, sum(1 for f in flagged if f["source"] == "profanity"))
    original_toxicity_score = bucket_score_0_1_2(toxicity_raw, sum(1 for f in flagged if f["source"] == "toxicity"))
    original_stereotype_score = bucket_score_0_1_2(
        stereotype_raw,
        sum(1 for f in flagged if f["source"] in {"stereotype", "crows_pairs", "bbq"})
    )
    original_leading_score = bucket_score_0_1_2(leading_raw, sum(1 for f in flagged if f["source"] == "leading"))

    overall_raw = profanity_raw + toxicity_raw + stereotype_raw + leading_raw
    original_score = bucket_score_0_1_2(overall_raw, len(flagged))

    # --- Rewrite ---
    cleaned_draft = deterministic_clean_draft(prompt, flagged)
    internal_rewrite_text = internal_neutral_rewrite(cleaned_draft, flagged)
    flagged_phrase = flagged[0].get("matched_text") if flagged else ""

    if should_use_groq_polish(cleaned_draft, internal_rewrite_text, flagged):
        rewritten_prompt = await groq_polish_rewrite(
            original_prompt=prompt,
            internal_rewrite=internal_rewrite_text,
            flagged_phrase=flagged_phrase,
        )
    else:
        rewritten_prompt = internal_rewrite_text

    if flagged and similarity_ratio(prompt, rewritten_prompt) > 0.92:
        rewritten_prompt = force_neutral_rewrite(prompt, flagged)

    # --- Scan rewritten ---
    flagged_2, profanity_raw_2, toxicity_raw_2, stereotype_raw_2, leading_raw_2 = scan_prompt(rewritten_prompt)
    filtered_flagged_2 = filter_rewritten_flags(flagged_2, rewritten_prompt)

    profanity_raw_2 = sum(
        f.get("severity_rating", 0.0) * f.get("count", 1)
        for f in filtered_flagged_2
        if f.get("source") == "profanity"
    )

    toxicity_raw_2 = sum(
        f.get("severity_rating", 0.0) * f.get("count", 1)
        for f in filtered_flagged_2
        if f.get("source") == "toxicity"
    )

    stereotype_raw_2 = sum(
        f.get("severity_rating", 0.0) * f.get("count", 1)
        for f in filtered_flagged_2
        if f.get("source") in {"stereotype", "crows_pairs", "bbq"}
    )

    leading_raw_2 = sum(
        f.get("severity_rating", 0.0) * f.get("count", 1)
        for f in filtered_flagged_2
        if f.get("source") == "leading"
    )

    overall_raw_2 = profanity_raw_2 + toxicity_raw_2 + stereotype_raw_2 + leading_raw_2

    # Rewritten scores use relaxed thresholds since neutralised prompts
    # legitimately retain identity terms for context (e.g. "women" in a
    # balanced question is not the same bias as "women" in a deficit framing)
    rewritten_profanity_score = rewritten_score_cap(
        original_profanity_score,
        profanity_raw_2,
        sum(1 for f in filtered_flagged_2 if f["source"] == "profanity"),
    )
    rewritten_toxicity_score = rewritten_score_cap(
        original_toxicity_score,
        toxicity_raw_2,
        sum(1 for f in filtered_flagged_2 if f["source"] == "toxicity"),
    )
    rewritten_stereotype_score = rewritten_score_cap(
        original_stereotype_score,
        stereotype_raw_2,
        sum(1 for f in filtered_flagged_2 if f["source"] in {"stereotype", "crows_pairs", "bbq"}),
    )
    rewritten_leading_score = rewritten_score_cap(
        original_leading_score,
        leading_raw_2,
        sum(1 for f in filtered_flagged_2 if f["source"] == "leading"),
    )
    rewritten_score = rewritten_score_cap(
        original_score,
        overall_raw_2,
        len(filtered_flagged_2),
    )

    change_summary = build_change_bullets(prompt, cleaned_draft, rewritten_prompt, flagged)

    result = {
        "original_score": original_score,
        "rewritten_score": rewritten_score,

        "original_profanity_score": original_profanity_score,
        "original_toxicity_score": original_toxicity_score,
        "original_stereotype_score": original_stereotype_score,
        "original_leading_score": original_leading_score,

        "rewritten_profanity_score": rewritten_profanity_score,
        "rewritten_toxicity_score": rewritten_toxicity_score,
        "rewritten_stereotype_score": rewritten_stereotype_score,
        "rewritten_leading_score": rewritten_leading_score,

        "flagged": flagged,
        "rewritten_flagged": filtered_flagged_2,
        "original_raw_score": overall_raw,
        "rewritten_raw_score": overall_raw_2,

        "cleaned_draft": cleaned_draft,
        "internal_rewrite": internal_rewrite_text,
        "rewritten_prompt": rewritten_prompt,
        "change_summary": change_summary,
        "rewrite_method": "internal rewrite + fast groq polish",
        "rewrite_model": POLISH_MODEL if GROQ_API_KEY else None,
        "selected_test_model": selected_model,
        "flagged_phrase": flagged_phrase,
    }

    return result