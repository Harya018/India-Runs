"""
Configuration and domain knowledge for the Redrob candidate ranker.

Every constant here traces back to either:
  (a) an explicit statement in job_description.md, or
  (b) a pattern empirically observed in candidates.jsonl during EDA
      (see notebooks/eda.ipynb for the exploration that produced these numbers).

Keeping this in one file means the scoring logic in ranker.py stays readable,
and a reviewer can audit "why does the model believe X" by reading this file
alone, without digging through scoring code.
"""

# ---------------------------------------------------------------------------
# Skill taxonomy
#
# The skill vocabulary in the dataset splits cleanly into three frequency
# bands (see EDA): ~12k-frequency generic skills (irrelevant to this JD),
# ~5k-frequency general ML/AI skills (relevant but not the JD's core ask),
# and ~1.3k-frequency core IR/ranking/retrieval skills that are *exactly*
# the JD's "things you absolutely need" list. A handful of skills (freq < 10)
# are clearly synonyms planted to reward candidates who describe the same
# work without the trendy buzzword (e.g. "Ranking Systems", "Search
# Infrastructure") -- these get folded into CORE_SKILLS too.
# ---------------------------------------------------------------------------

# Core IR / retrieval / ranking skills -- the JD's explicit "must haves".
# Weighted highest in skill scoring.
CORE_SKILLS = {
    # embeddings & retrieval models
    "embeddings", "sentence transformers", "bm25", "learning to rank",
    "information retrieval", "information retrieval systems", "vector search",
    "vector representations", "semantic search", "search backend",
    "search infrastructure", "search & discovery", "indexing algorithms",
    "text encoders", "content matching", "ranking systems",
    "recommendation systems", "nlp", "natural language processing",
    # vector databases / hybrid search infra (JD names these explicitly)
    "pinecone", "weaviate", "qdrant", "milvus", "pgvector", "opensearch",
    "elasticsearch", "faiss",
    # core ML/eng substance the JD cares about (production-grade, not buzzwords)
    "python", "pytorch", "tensorflow", "scikit-learn", "machine learning",
    "deep learning",
    # LLM integration the JD explicitly mentions
    "rag", "llms", "prompt engineering", "hugging face transformers",
    "langchain", "llamaindex", "haystack",
}

# "Nice to have" per JD: LLM fine-tuning, learning-to-rank models (already
# in core), distributed systems / inference optimization, MLOps tooling.
NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "mlops", "mlflow",
    "kubeflow", "weights & biases", "bentoml", "feature engineering",
    "data science", "statistical modeling", "reinforcement learning",
    "model adaptation", "open-source ml libraries", "workflow orchestration",
    "data pipelines", "spark", "airflow", "kafka", "docker", "kubernetes",
}

# Skills that signal a *different* specialization than what the JD wants.
# The JD explicitly says CV/speech/robotics specialists without NLP/IR
# exposure would be "re-learning fundamentals" -- not a hard reject, but a
# specialization mismatch that should reduce (not zero out) skill fit.
CV_SPEECH_SKILLS = {
    "image classification", "object detection", "computer vision", "cnn",
    "yolo", "gans", "diffusion models", "opencv", "asr", "speech recognition",
    "tts", "reinforcement learning",  # RL only counts here if NLP/IR absent
}

# Generic / off-domain skills carry ~zero weight for this JD. Not an
# exhaustive list -- anything not in the sets above defaults to ~0 weight.
GENERIC_SKILLS = {
    "html", "css", "javascript", "typescript", "react", "angular", "vue.js",
    "redux", "node.js", "next.js", "webpack", "tailwind", "graphql",
    "rest apis", "spring boot", "django", "flask", "fastapi", "microservices",
    "grpc", "java", "go", "rust", "sql", "postgresql", "mongodb", "redis",
    "snowflake", "databricks", "dbt", "bigquery", "etl", "ci/cd", "terraform",
    "aws", "azure", "gcp", "agile", "scrum", "project management", "sales",
    "marketing", "accounting", "excel", "powerpoint", "photoshop",
    "illustrator", "figma", "seo", "content writing", "tally", "six sigma",
    "salesforce crm", "apache beam", "apache flink", "time series",
    "forecasting", "hadoop",
}

# ---------------------------------------------------------------------------
# Career-pattern traps, named explicitly in job_description.md
# ---------------------------------------------------------------------------

# "People who have only worked at consulting firms ... in their entire
# career." Current-company-only would be too aggressive (JD: "if you're
# currently at one of these companies but have prior product-company
# experience, that's fine") -- so this is checked against the FULL
# career_history, not just current_company.
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl",
}

# "Title-chasers ... switching companies every 1.5 years." Operationalized
# as average tenure across roles, in months, below this threshold (with
# >=3 roles so a single short stint doesn't trigger it unfairly).
TITLE_CHASER_AVG_TENURE_MONTHS = 18
TITLE_CHASER_MIN_ROLES = 3

# "Senior engineer who hasn't written production code in the last 18
# months because they've moved into architecture/tech lead." We check the
# CURRENT role's title/description for management-only language.
MANAGEMENT_ONLY_TITLE_MARKERS = {
    "manager", "director", "head of", "vp ", "vice president", "chief",
}
# but NOT these, which can appear in legitimate senior IC titles
IC_TITLE_EXCEPTIONS = {"engineering manager"}  # still counts as management; kept for clarity

# JD: role is in Pune/Noida (hybrid), open to relocation from Tier-1 Indian
# cities, open to Hyderabad/Pune/Mumbai/Delhi NCR, outside India is
# case-by-case with no visa sponsorship.
PREFERRED_LOCATIONS_INDIA = {
    "pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram", "ncr",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

# JD: "We'd love sub-30-day notice. We can buy out up to 30 days."
NOTICE_PERIOD_IDEAL_DAYS = 30

# ---------------------------------------------------------------------------
# Honeypot detection (job_description / README: ~80 honeypots in the pool,
# "subtly impossible profiles", e.g. expert proficiency with ~0 duration)
# ---------------------------------------------------------------------------

HONEYPOT_EXPERT_MAX_DURATION_MONTHS = 3   # "expert" claimed with <=3mo use
HONEYPOT_ADVANCED_MAX_DURATION_MONTHS = 1  # "advanced" claimed with <=1mo use
HONEYPOT_MIN_EXPERT_SKILL_COUNT = 8        # implausibly broad "expert" spread

# ---------------------------------------------------------------------------
# Scoring weights
#
# These combine into a single composite "fit score" per candidate, BEFORE
# the behavioral multiplier is applied. Weights sum to 1.0 for readability;
# the absolute scale doesn't matter since we only need a ranking.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "skill_substance": 0.32,    # trust-weighted core+nice-to-have skill depth
    "career_relevance": 0.30,   # semantic match of career_history to the JD's actual ask
    "experience_fit": 0.13,     # years_of_experience vs 5-9 band + tenure stability
    "seniority_authenticity": 0.10,  # writes code recently, not pure management
    "location_logistics": 0.08, # location/relocation/notice period fit
    "education": 0.07,          # minor signal; JD doesn't emphasize pedigree
}

# Behavioral multiplier range -- bounds how much availability signals can
# move the final score, so a strong candidate with mediocre engagement
# doesn't get buried by multiplier alone (JD: "down-weight ... appropriately",
# not "zero out").
BEHAVIORAL_MULTIPLIER_MIN = 0.55
BEHAVIORAL_MULTIPLIER_MAX = 1.10

# Hard penalty multiplier applied to honeypots and full disqualifiers.
# Not literally zero, to keep scores strictly ordered / debuggable, but
# low enough that they will not appear in any realistic top-100 cut.
HONEYPOT_PENALTY_MULTIPLIER = 0.02
PURE_RESEARCH_PENALTY_MULTIPLIER = 0.15
CONSULTING_ONLY_PENALTY_MULTIPLIER = 0.20
CV_SPEECH_ONLY_PENALTY_MULTIPLIER = 0.55  # soft per JD ("re-learning fundamentals", not a hard no)
