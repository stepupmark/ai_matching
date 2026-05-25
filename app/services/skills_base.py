"""
Comprehensive, extensible skills database for resume/JD parsing.

Maintained as a frozen set of lowercase skill labels for O(1) lookups.
Add new skills by appending to one of the category lists below.
"""

from __future__ import annotations

# ──────────────────────────────────────────────
#  Skill categories — add new skills here
# ──────────────────────────────────────────────

_PROGRAMMING_LANGUAGES = [
    "python", "java", "javascript", "typescript", "go", "golang",
    "rust", "c++", "c#", "csharp", "c", "swift", "kotlin",
    "ruby", "php", "scala", "perl", "haskell", "lua", "r",
    "dart", "elixir", "clojure", "erlang", "julia", "fortran",
    "assembly", "objective-c", "delphi", "cobol", "lisp", "scheme",
    "groovy", "powershell", "shell", "bash", "zsh",
]

_FRAMEWORKS_AND_LIBRARIES = [
    # Frontend
    "react", "react.js", "reactjs", "next.js", "nextjs", "vue", "vue.js",
    "vuejs", "angular", "angular.js", "angularjs", "svelte", "sveltekit",
    "remix", "gatsby", "nuxt", "nuxt.js", "nuxtjs", "solid.js", "solidjs",
    "jquery", "backbone.js", "ember.js", "htmx", "alpine.js", "tailwind",
    "tailwindcss", "bootstrap", "material-ui", "mui", "chakra-ui",
    "styled-components", "sass", "scss", "less", "webpack", "vite",
    "redux", "mobx", "zustand", "react-query", "tanstack query",
    "apollo", "graphql",
    # Backend
    "django", "flask", "fastapi", "spring", "spring boot", "springboot",
    "asp.net", ".net", "dotnet", "express", "express.js", "expressjs",
    "nestjs", "next.js", "nextjs", "rails", "ruby on rails",
    "laravel", "symfony", "phoenix", "gin", "echo", "fiber",
    "actix", "rocket", "tornado", "aiohttp", "starlette",
    "node.js", "nodejs", "deno", "bun",
    # Data / ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "dplyr", "ggplot2", "tidyverse", "jupyter", "jupyter notebook",
    "langchain", "llamaindex", "huggingface", "transformers",
    "spacy", "nltk", "opencv", "pillow",
    # Mobile
    "flutter", "react native", "reactnative", "xamarin", "ionic",
    "android sdk", "ios sdk", "swiftui", "uikit",
]

_DATABASES = [
    "postgresql", "postgres", "pg", "mysql", "sqlite", "sqlite3",
    "mongodb", "mongo", "cassandra", "redis", "elasticsearch", "es",
    "dynamodb", "couchbase", "couchdb", "mariadb", "oracle", "sql server",
    "mssql", "snowflake", "bigquery", "redshift", "clickhouse",
    "neo4j", "influxdb", "timescaledb", "cockroachdb", "firestore",
    "supabase", "fauna", "planetscale", "neon",
]

_CLOUD_AND_DEVOPS = [
    "aws", "amazon web services", "azure", "microsoft azure",
    "gcp", "google cloud", "google cloud platform", "heroku",
    "digitalocean", "linode", "vercel", "netlify", "cloudflare",
    "docker", "kubernetes", "k8s", "terraform", "pulumi",
    "ansible", "chef", "puppet", "jenkins", "github actions",
    "gitlab ci", "circleci", "travis ci", "argocd", "helm",
    "prometheus", "grafana", "datadog", "new relic", "sentry",
    "elk stack", "elastic stack", "logstash", "kibana", "fluentd",
    "nginx", "apache", "traefik", "istio", "envoy",
    "serverless", "lambda", "ecs", "eks", "ec2", "s3",
    "cloudformation", "cdk", "vault", "consul",
]

_DATA_TOOLS = [
    "airflow", "dbt", "spark", "apache spark", "pyspark",
    "hadoop", "hive", "pig", "kafka", "apache kafka",
    "flink", "beam", "databricks", "snowflake", "druid",
    "presto", "trino", "tableau", "power bi", "looker",
    "metabase", "superset", "ollap", "etl", "elt",
]

_VERSION_CONTROL = [
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    "perforce", "azure devops",
]

_METHODOLOGIES_AND_DOMAINS = [
    "agile", "scrum", "kanban", "waterfall", "lean",
    "ci/cd", "cicd", "tdd", "test-driven development",
    "microservices", "soa", "event-driven", "event sourcing",
    "domain-driven design", "ddd", "clean architecture",
    "hexagonal architecture", "onion architecture",
    "rest", "restful", "grpc", "graphql", "soap",
    "machine learning", "deep learning", "nlp", "computer vision",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "data science", "data engineering", "mlops", "data pipeline",
    "a/b testing", "experimentation", "feature flags",
    "oauth", "jwt", "saml", "openid", "sso",
    "pci dss", "hipaa", "gdpr", "soc 2",
]

_PROJECT_MANAGEMENT = [
    "jira", "confluence", "notion", "asana", "trello",
    "monday.com", "basecamp", "linear", "clickup",
    "slack", "teams", "discord",
]

# ──────────────────────────────────────────────
#  Compiled skill set
# ──────────────────────────────────────────────

_ALL_SKILLS: set[str] = set()
for _cat in [
    _PROGRAMMING_LANGUAGES,
    _FRAMEWORKS_AND_LIBRARIES,
    _DATABASES,
    _CLOUD_AND_DEVOPS,
    _DATA_TOOLS,
    _VERSION_CONTROL,
    _METHODOLOGIES_AND_DOMAINS,
    _PROJECT_MANAGEMENT,
]:
    _ALL_SKILLS.update(s.lower() for s in _cat)

SKILLS_SET = frozenset(_ALL_SKILLS)

# ──────────────────────────────────────────────
#  Common location keywords
# ──────────────────────────────────────────────

LOCATIONS = frozenset([
    # US Cities
    "new york", "san francisco", "los angeles", "chicago", "houston",
    "phoenix", "philadelphia", "san antonio", "san diego", "dallas",
    "austin", "seattle", "boston", "denver", "washington dc",
    "miami", "atlanta", "portland", "detroit", "minneapolis",
    "raleigh", "nashville", "salt lake city", "orlando", "tampa",
    "pittsburgh", "baltimore", "charlotte", "sacramento",
    # US States / Regions
    "california", "texas", "new york", "florida", "illinois",
    "massachusetts", "washington", "colorado", "oregon",
    "bay area", "silicon valley", "remote", "hybrid",
    # Global
    "london", "berlin", "paris", "tokyo", "singapore", "sydney",
    "toronto", "vancouver", "amsterdam", "dublin", "zurich",
    "bangalore", "mumbai", "delhi", "shanghai", "beijing",
    "stockholm", "copenhagen", "oslo", "helsinki",
])
