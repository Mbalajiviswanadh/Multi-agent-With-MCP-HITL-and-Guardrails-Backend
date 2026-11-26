# TripMate AI — Production Implementation Flow Log

This document tracks **every implementation step** in chronological order as we build the production-grade multi-agent system from the blueprint. Each step is logged with what was done, what files were created/changed, and what concept it covers.

---

## ✅ Phase 0 — Baseline (What Already Exists)

**What was already built:**
- `backend.py` — Monolithic LangGraph graph with supervisor, guardrail, 5 specialist agents, HITL, final agent.
- `app.py` — FastAPI server with `/api/travel`, `/api/travel/approve`, `/api/history/{thread_id}`, `/health`.
- `mymcp/mcp_client_main.py` — MultiServerMCPClient wiring Tavily (HTTP), AviationStack (stdio), Weather (stdio).
- `mymcp/custom_weather_mcp.py` — Custom FastMCP weather server.
- PostgreSQL checkpointing via `PostgresSaver` from `langgraph-checkpoint-postgres`.
- Structured logging to `agent_runs.log` + console.

**Concepts covered:** Orchestration (LangGraph), MCP tooling, Short-term thread memory, Human-in-the-loop, Basic input guardrail.

---

## ✅ Phase 1 — Modular Architecture Restructure

**What was done:**
Extracted the monolithic `backend.py` into a clean `src/` module hierarchy matching the blueprint directory structure. Each concept lives in its own file.

**Files Created:**
```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py          ← Pydantic Settings (validates all .env keys at startup)
│   ├── database.py        ← psycopg ConnectionPool (production-grade pooling)
│   └── logging_config.py  ← Centralized structured logger setup
├── agents/
│   ├── __init__.py
│   ├── state.py           ← Upgraded TravelState (user_id, context_summary, user_profile, similar_past_trips, query_intent)
│   ├── graph.py           ← StateGraph builder + compiled graph + run/resume helpers
│   ├── supervisor.py      ← Supervisor + 3-intent classifier (chat/travel/blocked)
│   ├── chat.py            ← Conversational chat agent (Mem0-powered memory)
│   ├── specialists/
│   │   ├── __init__.py
│   │   ├── flight.py      ← Flight agent
│   │   ├── hotel.py       ← Hotel agent
│   │   ├── weather.py     ← Weather agent
│   │   └── budget.py      ← Budget agent
│   ├── itinerary.py       ← Itinerary agent
│   └── final.py           ← Final response agent
├── memory/
│   ├── __init__.py
│   ├── checkpointer.py    ← PostgresSaver setup with connection pool
│   ├── profile_store.py   ← Legacy long-term user profile CRUD (users/user_profiles tables)
│   ├── vector_store.py    ← pgvector semantic search (trip_memories_vector table)
│   └── mem0_service.py    ← Mem0 self-evolving memory service (Groq + FastEmbed + Qdrant)
├── guardrails/
│   ├── __init__.py
│   ├── input_guardrail.py  ← 3-intent classifier: chat / travel / blocked
│   └── output_guardrail.py ← Schema + hallucination check on final response
└── reliability/
    ├── __init__.py
    ├── llm_factory.py      ← LLM with fallback providers + timeouts
    ├── retry_policy.py     ← LangGraph RetryPolicy definitions
    └── circuit_breaker.py  ← pybreaker circuit breakers for MCP tools
```

**Concept Covered:** Modular Architecture, Separation of Concerns

---

## ✅ Phase 2 — Core Config, Database Pool & Logging

**What was done:**
- Created `src/core/config.py` using Pydantic `BaseSettings` — validates all environment variables at startup, fails fast with clear error messages if keys are missing.
- Created `src/core/database.py` with `psycopg_pool.ConnectionPool` replacing the single `psycopg.connect()` call. Supports concurrent user sessions without connection exhaustion.
- Created `src/core/logging_config.py` as the single source of logger configuration for all modules.

**Key Change:** The old `backend.py` had one global `psycopg.connect()` — a production server with 50 concurrent users would exhaust this. `ConnectionPool` manages a pool of 5–20 reusable connections.

**Concept Covered:** Production Database Connection Management, Config Validation

---

## ✅ Phase 3 — Upgraded TravelState (3-Tier Memory Fields)

**What was done:**
Updated `src/agents/state.py` to add fields for:
- `user_id` — identifies the logged-in user across sessions
- `context_summary` — condensed summary of old messages to prevent token overflow
- `user_profile` — loaded at graph start (now contains `mem0_context` from Mem0 search)
- `similar_past_trips` — retrieved from pgvector for semantic context
- `query_intent` — `"chat"` | `"travel"` | `"blocked"` (added in Phase 11)

**Concept Covered:** 3-Tier Memory Architecture (State Design)

---

## ✅ Phase 4 — 3-Tier Memory Implementation

**What was done:**
- `src/memory/checkpointer.py` — Wraps `PostgresSaver` with connection pool.
- `src/memory/profile_store.py` — CRUD functions to `user_profiles` table: `load_user_profile()`, `upsert_user_profile()`.
- `src/memory/vector_store.py` — `store_trip_embedding()` and `find_similar_trips()` using `pgvector` cosine distance (`<=>`).

**SQL Migration run:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128),
    email VARCHAR(128) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    home_airport VARCHAR(10) DEFAULT 'HYD',
    preferred_travel_style VARCHAR(64),
    dietary_preferences TEXT[],
    frequent_airlines TEXT[],
    hotel_preferences TEXT[],
    custom_notes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS threads (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trip_memories_vector (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(64) REFERENCES users(id) ON DELETE CASCADE,
    thread_id VARCHAR(64),
    destination VARCHAR(128),
    summary TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS trip_memories_hnsw
    ON trip_memories_vector USING hnsw (embedding vector_cosine_ops);
```

**Concept Covered:** pgvector, Long-Term Memory, Semantic Episodic Memory

---

## ✅ Phase 5 — Reliability Layer

**What was done:**
- `src/reliability/llm_factory.py` — Configured resilient LLM pipeline with primary model `openai/gpt-oss-120b` and fallback model `openai/gpt-oss-20b` via `.with_fallbacks()`. Both run on the same `GROQ_API_KEY`. The 20B fallback automatically activates if the 120B model hits TPM limits (HTTP 413/429).
- `src/reliability/retry_policy.py` — `MCP_RETRY_POLICY` and `LLM_RETRY_POLICY` using LangGraph `RetryPolicy` with exponential backoff and jitter.
- `src/reliability/circuit_breaker.py` — `pybreaker.CircuitBreaker` instances for Tavily, AviationStack, and Weather MCP tools, plus `call_with_timeout()` async helper.

**Concept Covered:** Fallback LLMs, Retries with Exponential Backoff, Timeouts, Circuit Breakers

---

## ✅ Phase 6 — Dual Guardrails (Input + Output)

**What was done:**
- `src/guardrails/input_guardrail.py` — Extracted from supervisor. Pure intent classification function returning `{allowed: bool, reason: str}`.
- `src/guardrails/output_guardrail.py` — Post-processing validator that checks the final response for hallucination patterns (fabricated pricing guarantees, missing safety disclaimers) and applies corrections.

**Concept Covered:** Input & Output Guardrails, Hallucination Detection

---

## ✅ Phase 7 — Threads API & SSE Streaming

**What was done:**
- `src/api/routers/threads.py` — New REST endpoints:
  - `POST /api/threads` — Create a new chat thread
  - `GET /api/threads?user_id=x` — List all threads for a user (chatbot sidebar)
  - `DELETE /api/threads/{thread_id}` — Delete a thread
- `src/api/routers/travel.py` — Refactored travel endpoint using `travel_graph.astream_events()` to stream agent events back to the frontend via Server-Sent Events (SSE).

**Concept Covered:** Chat Sidebar, SSE Streaming, Real-time Feedback

---

## ✅ Phase 8 — Offline Evaluation Suite

**What was done:**
- `evals/datasets/golden_queries.jsonl` — 30 curated test cases (valid trips, ambiguous queries, adversarial prompts, out-of-domain requests).
- `evals/test_routing.py` — Runs supervisor against golden dataset, measures routing precision.
- `evals/test_faithfulness.py` — LLM-as-a-judge: checks if final itinerary hallucinated facts not in MCP tool output.
- `evals/run_evals.py` — CLI runner that outputs a full eval report.

**Concept Covered:** LLM Evaluation, Ragas-style Metrics, CI/CD for AI

---

## ✅ Phase 9 — Modular Agent Graph & FastAPI Integration

**What was done:**
- Created modular specialist agent files in `src/agents/specialists/` (`flight.py`, `hotel.py`, `weather.py`, `budget.py`).
- Created core orchestration agents: `src/agents/supervisor.py`, `src/agents/itinerary.py`, `src/agents/final.py`.
- Assembled `src/agents/graph.py` with LangGraph `StateGraph`, node-level `RetryPolicy` attachments, 3-tier memory injection (`user_profiles` + `pgvector`), and execution runners.
- Integrated `src/api/routers/threads.py` directly into `app.py` for full chatbot sidebar support.
- Updated `backend.py` to use `resilient_llm` (`openai/gpt-oss-120b` with fallback to `openai/gpt-oss-20b`).

**Concept Covered:** Clean Architecture, End-to-End Multi-Agent Execution, Chatbot Session Management

---

## ✅ Phase 10 — Production Server Polish & Local Docker Compose

**What was done:**
- `app.py` Lifespan management: Added async lifespan handler to automatically run `run_schema_migrations()` and `run_vector_schema_migrations()` on server start, and cleanly release connection pools (`close_pool()`) on shutdown.
- Modernized `/api/travel/stream` SSE endpoint: Replaced legacy non-existent nodes with the real LangGraph graph nodes (`supervisor`, `flight_agent`, `hotel_agent`, `weather_agent`, `budget_agent`, `itinerary_agent`, `human_approval`, `final_agent`). Emits real-time progression events and token chunks.
- Added `user_id` cross-session tracking to `TravelRequest`, `ApprovalRequest`, `run_travel_agent`, and `resume_travel_agent`.
- Created `docker-compose.yml` configured with `pgvector/pgvector:pg16` for local containerized development.
- Updated `requirements.txt` and `pyproject.toml` with `pydantic-settings` and `pybreaker`.

**Concept Covered:** FastAPI Lifespan Events, Real-time SSE Streaming, Dockerized Vector DB Infrastructure

---

## ✅ Phase 11 — Intent-Based Routing (3-Way Semantic Router)

**What was done:**

Upgraded the system from a binary guardrail (`allowed/blocked`) to a **3-intent semantic router** that classifies every user message before routing it to the appropriate pipeline.

### Architecture

```
                       User Input
                           │
                           ▼
                  ┌─────────────────┐
                  │   Supervisor    │  (1 LLM call: intent classification)
                  │ Input Guardrail │
                  └────────┬────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
 intent = "chat"   intent = "travel"  intent = "blocked"
       │                   │                   │
       ▼                   ▼                   ▼
 [chat_agent]      [Flight Agent]      [guardrail_blocked]
 Warm reply +      [Hotel Agent]       Safety rejection
 Memory save       [Weather Agent]     (immediate END)
 (fast ~2s)        [Budget Agent]
 → END             [Itinerary Agent]
                   [Human Approval]
                   [Final Agent]
                   → END (~30-60s)
```

### Intent Definitions

| Intent | Trigger | Pipeline | Speed |
|--------|---------|----------|-------|
| `chat` | Greetings, introductions, stated preferences, questions about TripMate | `chat_agent` → END | ~2s |
| `travel` | Any trip planning, destination, flight, hotel, weather, budget query | Full specialist pipeline | ~30-60s |
| `blocked` | Harmful content, jailbreaks, prompt injections | `guardrail_blocked` → END | ~1s |

### Files Changed

- `src/guardrails/input_guardrail.py` — Upgraded from binary `{allowed: bool}` to 3-class `{intent: "chat"/"travel"/"blocked", allowed: bool, reason: str}`.
- `src/agents/supervisor.py` — Updated to route on `intent` field: `chat` → `chat_agent`, `travel` → specialist chain, `blocked` → rejection.
- `src/agents/chat.py` — New conversational agent node. Responds naturally to casual conversation and stores preferences in Mem0 memory.
- `src/agents/graph.py` — Added `chat_agent` node with `chat_agent → END` edge. Added 3-way `route_from_supervisor()` function.
- `src/agents/state.py` — Added `query_intent: str` field to `TravelState`.

### Key Benefit

- **Normal conversations work:** Users can say "Hi", "My name is Viswanadh", "I love vegetarian food" and get natural replies — without triggering a 30-second multi-agent travel pipeline.
- **Preferences are captured:** Even casual mentions ("I prefer budget travel") are silently stored in Mem0 for personalization when the user does request a trip.
- **Safety maintained:** Harmful/off-topic requests are still immediately blocked.

**Concept Covered:** Semantic Intent Routing, Progressive User Profiling, Conditional Graph Branching

---

## ✅ Phase 12 — Mem0 Self-Evolving Memory Integration

**What was done:**

Integrated **Mem0** (mem0ai v2.2.1) as the primary long-term memory engine, replacing the manual SQL-based `user_profiles` CRUD approach for conversational fact extraction.

### Why Mem0 vs Raw SQL

| Capability | Old SQL Approach | Mem0 |
|------------|------------------|------|
| Fact extraction | Manual JSON prompt → parse → SQL UPSERT | Automatic via Groq LLM |
| Conflict resolution | `COALESCE` (additive, never overwrites) | Self-evolving (overwrites old facts) |
| Semantic retrieval | Fixed SQL `SELECT` by column | Built-in vector similarity search |
| Memory scope | Fixed typed columns (home_airport, dietary_preferences…) | Any natural-language fact |
| New preference types | Requires SQL schema change | Zero schema changes needed |

### Mem0 Stack Configuration

```
LLM:          Groq / llama-3.1-8b-instant        ← Lightweight model for fact extraction
Embedder:     FastEmbed / all-MiniLM-L6-v2       ← 384-dim local ONNX (zero API cost)
Vector Store: Qdrant Cloud (Managed Cloud DB)    ← Cloud-hosted persistence across all deploys
Collection:   tripmate_memories
```

### Memory Lifecycle

```
User says: "My name is Viswanadh, I prefer vegetarian food"
                    │
                    ▼
           [chat_agent]
        mem0_add(messages, user_id)
                    │
                    ▼
         Mem0 LLM extracts facts:
         • "User's name is Viswanadh"
         • "Prefers vegetarian food"
                    │
                    ▼
      Stored in Qdrant (local vector DB)

─────────────────────────────────────

User later says: "Actually I eat chicken now"
                    │
                    ▼
         Mem0 self-evolves:
         • DELETES "Prefers vegetarian food"
         • ADDS "Eats chicken"
         (No duplicate facts!)

─────────────────────────────────────

User requests: "Plan a trip to Goa"
                    │
                    ▼
         mem0_search("Plan a trip to Goa", user_id)
                    │
                    ▼
         Returns: ["User's name is Viswanadh", "Eats chicken"]
                    │
                    ▼
         Injected into supervisor/specialist context
         → Trip planned with chicken-friendly restaurants
```

### Files Created / Changed

| File | Change |
|------|--------|
| `src/memory/mem0_service.py` | **New** — Mem0 singleton, `mem0_add()`, `mem0_search()`, `mem0_get_all()`, `mem0_delete_user()`, `mem0_format_for_context()` |
| `src/agents/chat.py` | Replaced manual `_extract_preferences()` JSON prompt + SQL with `mem0_add()` and `mem0_search()` |
| `src/agents/graph.py` | `run_travel_agent()` calls `mem0_search()` instead of `load_user_profile()`. `resume_travel_agent()` calls `mem0_add()` after trip approval |
| `app.py` | Added `GET /api/memory/{user_id}` and `DELETE /api/memory/{user_id}` endpoints |
| `pyproject.toml` | Added `mem0ai>=2.2.1` dependency |

### New API Endpoints

```
GET    /api/memory/{user_id}   → View all Mem0 memories for a user
DELETE /api/memory/{user_id}   → Delete all memories (GDPR right-to-be-forgotten)
```

### Memory Scopes

Mem0 memories are scoped by:
- `user_id` — Persistent cross-session user preferences ("prefers vegetarian food")
- `agent_id` = `"tripmate_ai"` — Namespaced to this application
- `run_id` = `thread_id` — Per-trip context (optional, trip-specific notes)

**Concept Covered:** Self-Evolving Memory, Automatic Fact Extraction, Conflict Resolution, Semantic Memory Search, Memory Scoping

---

---

## ✅ Phase 13 — Production JWT Authentication System

**Goal:** Add a fully production-grade authentication layer so users can register, log in with passwords, and access only their own data via ownership-enforced endpoints.

**What was done:**
- Implemented password hashing with **PBKDF2-HMAC-SHA256** (100,000 iterations, 16-byte random salt) using Python stdlib — no external dependencies like `bcrypt`/`passlib`.
- Issued **signed JWT tokens** (HS256) using `PyJWT`. Token payload: `{sub: user_id, email, name, exp, iat}`.
- Created all auth routes, security utilities, and FastAPI dependencies in a clean modular structure.
- Added DB migration to add `password_hash` column to the `users` table on startup.
- Protected sensitive endpoints with **JWT ownership checks** (users can only read/delete their own data).
- Verified all flows with a full automated test suite — **4/4 tests passed**.

### Architecture

```
POST /api/auth/register
  → Validate email (regex), hash password (PBKDF2), INSERT users + user_profiles
  → Return signed JWT token (7-day expiry, HS256)

POST /api/auth/login
  → Lookup user by email, verify_password (constant-time HMAC compare)
  → Return signed JWT token

GET /api/auth/me
  → Decode JWT Bearer token (Depends: get_current_user)
  → JOIN users + user_profiles, return full profile

DELETE /api/memory/{user_id}      ← Strict auth (get_current_user, 401 if no token)
  → JWT decoded, ownership check: token.sub == user_id (403 if mismatch)

GET /api/memory/{user_id}         ← Optional auth (get_optional_current_user)
POST /api/travel                  ← Optional auth — infers user_id from JWT if present
POST /api/travel/approve          ← Optional auth — infers user_id from JWT if present
```

### Password Hashing Design

```
hash_password("SecurePassword123!"):
  salt = os.urandom(16)   ← Cryptographically secure 128-bit salt
  key  = pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
  → "pbkdf2_sha256$100000$<salt_b64>$<hash_b64>"

verify_password(plain, stored):
  → Constant-time hmac.compare_digest (prevents timing attacks)
```

### JWT Token Flow

```
Register/Login success:
  payload = {sub: "user_abc123", email: "...", name: "...", iat: now, exp: now+7d}
  token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")

Protected endpoint:
  Authorization: Bearer <token>
  → HTTPBearer extracts credentials
  → jwt.decode(token, JWT_SECRET_KEY) → payload
  → user = {id: payload["sub"], email: ..., name: ...}
```

### Files Created / Changed

| File | Change |
|------|--------|
| `src/core/security.py` | **New** — `hash_password()`, `verify_password()`, `create_access_token()`, `decode_access_token()` |
| `src/api/dependencies.py` | **New** — `get_current_user` (strict, 401 on failure), `get_optional_current_user` (returns None on no token) |
| `src/api/routers/auth.py` | **New** — `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` |
| `src/core/config.py` | Added `jwt_secret_key`, `jwt_algorithm`, `jwt_access_token_expire_minutes` settings |
| `src/memory/profile_store.py` | Schema SQL — added `password_hash VARCHAR(255)` column + `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` |
| `src/api/routers/threads.py` | Updated to use `get_optional_current_user` with thread ownership checks |
| `app.py` | Added `auth_router`, `Depends` import, optional auth on `/api/travel` and `/api/travel/approve`, strict auth on `DELETE /api/memory/{user_id}`, ownership checks on `GET /api/memory/{user_id}` |
| `scripts/test_auth_flow.py` | **New** — Full end-to-end auth verification suite (4/4 tests pass against live Render PostgreSQL) |

### Test Results

```
=================================================================
TripMate AI — Production JWT Authentication Verification
=================================================================
[1/4] Register user → 201 Created ✅ (JWT issued, user stored in PostgreSQL)
[2/4] Login + wrong password → 200 OK + 401 Unauthorized ✅ (PBKDF2 verified)
[3/4] GET /api/auth/me with valid token → 200 OK ✅ (identity confirmed from JWT)
[4/4] DELETE /api/memory without token → 401, with token → 200 ✅ (ownership enforced)
=================================================================
ALL PRODUCTION AUTHENTICATION & SECURITY TESTS PASSED 100%!
```

**Concept Covered:** JWT Authentication, PBKDF2 Password Hashing, Bearer Token Authorization, Route Ownership Enforcement, Production Security Middleware

---

*This document is auto-updated after each implementation phase.*

---

## ✅ Phase 14 — DeepEval Evaluation Framework & Rich Console UI

### What Was Built

A production-grade evaluation suite using **DeepEval**-style LLM-as-Judge metrics with a **Rich** terminal dashboard. The eval system now has 4 phases covering guardrail accuracy, routing precision, faithfulness grounding, and DeepEval LLM-judge metrics — all rendered in a styled Rich console table.

### Architecture

```
python -m evals.run_evals
│
├── [Phase 1] Guardrail & Routing Eval
│     └── evals/test_routing.py  →  load_dataset() + run_guardrail_eval() + run_routing_eval()
│
├── [Phase 2] Faithfulness (LLM-as-Judge)
│     └── evals/test_faithfulness.py  →  evaluate_faithfulness() per case
│
└── [Phase 3] DeepEval Metrics (4 LLM-Judge metrics)
      └── evals/test_deepeval.py  →  run_deepeval_suite()
            ├── FaithfulnessMetric     (Is response grounded in retrieved context?)
            ├── AnswerRelevancyMetric  (Does output directly answer the query?)
            ├── GEval Actionability    (Structured G-Eval: day-by-day, logistics, executability)
            └── HallucinationMetric    (Explicit check for fabricated prices/hotel names)
```

### Evaluation Metrics Summary

| Metric | Type | Threshold | Description |
|--------|------|-----------|-------------|
| **Guardrail Accuracy** | Rule-based | ≥ 85% | Supervisor correctly allows/blocks 30 labeled queries |
| **Routing Precision** | Rule-based | ≥ 75% | Supervisor picks correct specialist agents for travel queries |
| **Faithfulness (LLM-Judge)** | Groq-judge | ≥ 75% | Final answers grounded in MCP-retrieved data, no hallucinations |
| **Answer Relevancy** | DeepEval/Groq | ≥ 75% | Output directly answers user query (on-topic, complete) |
| **G-Eval Actionability** | DeepEval/Groq | ≥ 75% | Plan has day-by-day structure, realistic logistics, executable steps |
| **Hallucination Avoidance** | Groq-judge | ≥ 75% | No fabricated prices, hotel names, or flight numbers |

### Dataset: 50 Golden Test Cases

`evals/datasets/golden_queries.jsonl` — expanded to 50 entries covering:
- Core travel planning (Goa, Manali, Shimla, Kerala, Rajasthan, Pondicherry, Visakhapatnam, Gulmarg, etc.)
- Budget queries, luxury queries, multi-city circuits
- Edge cases: conversational chitchat, profile context ("I am Viswanadh", "remember my preference"), non-travel queries
- Adversarial: prompt injection, illegal requests, airport security bypass, passport forgery (all blocked)

### Rich Console Dashboard Results

```
                     ✨ Agent Evaluation Summary Dashboard                     
╔══════════════════════════════╤═════════════════╤══════════════╤═════════════╗
║                              │     Score /     │              │   Target    ║
║ Evaluation Dimension         │    Accuracy     │  Threshold   │   Status    ║
╟──────────────────────────────┼─────────────────┼──────────────┼─────────────╢
║ Input Guardrail Accuracy     │  90.0% (45/50)  │   ≥ 85.0%    │    PASS     ║
║ Supervisor Routing Precision │  81.1% (30/37)  │   ≥ 75.0%    │    PASS     ║
║ Faithfulness (LLM-as-Judge)  │      80.0%      │   ≥ 75.0%    │    PASS     ║
║ DeepEval Answer Relevancy    │      90.3%      │   ≥ 75.0%    │    PASS     ║
║ DeepEval G-Eval              │      80.0%      │   ≥ 75.0%    │    PASS     ║
║ Actionability                │                 │              │             ║
║ DeepEval Hallucination       │      95.0%      │   ≥ 75.0%    │    PASS     ║
║ Avoidance                    │                 │              │             ║
╚══════════════════════════════╧═════════════════╧══════════════╧═════════════╝
 Total Evaluation Runtime: 366.9s | Total Test Cases Evaluated: 61
```

### Key Design Decisions

**DeepEval Compatibility Layer:**
DeepEval 2.5.5 internally calls `sentry_sdk.init(profiles_sample_rate=...)` which is an option removed in newer `sentry_sdk` versions. We patch `sentry_sdk.init()` in-memory before DeepEval imports to silently drop unknown kwargs. This allows DeepEval to load without downgrading sentry_sdk.

**LangChain Schema Bridge:**
DeepEval still references `langchain.schema` (old path). LangChain 1.x moved it to `langchain_core.messages`. We inject a `sys.modules["langchain.schema"]` alias before the import chain.

**Groq as LLM Judge:**
All 4 LLM-judge metrics use the project's existing `resilient_llm` (Groq openai/gpt-oss-120b with fallback to openai/gpt-oss-20b). No external OpenAI key required — consistent with the rest of the project.

**Graceful Fallback:**
If DeepEval's native metric classes fail (network, API mismatch), each metric falls back to our own structured JSON-prompt LLM judge — ensuring the eval suite always produces results.

### Files Created / Changed

| File | Change |
|------|--------|
| `evals/test_deepeval.py` | **Rewritten** — 4 DeepEval-style metrics with Groq judge adapter, Rich table output, 6 test cases covering diverse travel scenarios |
| `evals/run_evals.py` | **Updated** — Added Phase 3 DeepEval runner, 6-metric summary dashboard, hallucination avoidance row |
| `evals/datasets/golden_queries.jsonl` | **Expanded** — 30 → 40 entries (added 10: conversational, adversarial, intent variety) |
| `FLOW_DOC.md` | Added Phase 14 section |

### How to Run

```powershell
# Full 3-phase eval suite
cd d:\Projects\ai-engineer\multi-agent-projects\03-MultiAgent-MCP-Guardrails
uv run python -m evals.run_evals

# DeepEval metrics only (Phase 3)
uv run python evals\test_deepeval.py
```

### Concept Covered

**LLM-as-Judge:** Using a capable LLM (Groq) to evaluate AI outputs against structured criteria — a powerful alternative to human review at scale.

**G-Eval:** A structured criteria-based evaluation method from DeepEval that breaks down quality into measurable sub-dimensions (structure, logistics, executability) and scores them in a single LLM call.

**Faithfulness Grounding:** Verifying that AI responses don't hallucinate facts by checking every claim against a retrieved-context ground truth.

**Hallucination Detection:** Explicit adversarial probe that asks the judge to list any fabricated prices, hotel names, or flight numbers — the most common failure mode in travel AI.

---

*This document is auto-updated after each implementation phase.*

---

## ✅ Phase 15 — Production-Ready Dynamic Routing & HITL Decoupling

### What Was Fixed & Why

**Problem Identified:**
1. **Forced `itinerary_agent`:** Previously, the supervisor was hardcoded to append `itinerary_agent` to *every* travel query. When a user asked a simple standalone question like `"what are the nearest best hotels near Guntur railway station?"`, the system forced an entire 1-day itinerary with timings (07:00, 08:30, etc.) that the user never asked for.
2. **Unnecessary HITL Interrupts:** Because `itinerary_agent` was forced, the system paused on an `interrupt()` for human approval, making the user click "Approve" just to see a hotel search result.
3. **Hardcoded Response Sections in Final Agent:** When the user clicked "Approve", `final_agent` used a hardcoded prompt template that unconditionally outputted `Flight Information`, `Weather Information`, `Day-by-Day Itinerary`, and `Budget`, even when the user never asked for flights and no flight tools were called. This resulted in hallucinated flight disclaimers and duplicate unwanted sections.

### Solution Architecture

```
User Query: "Hotels near Guntur railway station"
     │
     ▼
[Supervisor]  →  Analyzes query with refined routing prompt
     │           Selects ONLY ["hotel_agent"] (itinerary_agent is NOT forced)
     ▼
[hotel_agent] →  Executes live Tavily search
     │
     ▼
[Conditional Edge] → Checks next agents in selected_agents
     │                No more specialist agents?
     │                Is itinerary_agent in selected? NO → Routes directly to [final_agent]!
     ▼
[final_agent] →  Inspects selected_agents (only hotel_agent)
     │           Generates focused Hotel Guide with proximity, amenities, and rates.
     │           Strictly OMITS Flight Information, Weather, and Itinerary sections!
     ▼
[END]        →  Completed directly in ONE turn! requires_approval = False.
```

When a user **does** ask for a full trip (e.g. `"Plan a 3-day trip to Goa with flights and hotel under ₹20k"`):
1. `supervisor` selects `["flight_agent", "hotel_agent", "budget_agent", "itinerary_agent"]`.
2. Specialists execute -> `itinerary_agent` drafts the cohesive multi-day plan.
3. `itinerary_agent` routes to `human_approval` which pauses execution via `interrupt()`.
4. User reviews and approves in the UI.
5. `final_agent` resumes, polishes the approved draft, and presents the final plan.

### Files Changed

| File | Change |
|------|--------|
| `src/agents/supervisor.py` | Removed forced `selected_agents.append("itinerary_agent")`. Added strict routing rules: single-topic queries select only that specialist. |
| `src/agents/graph.py` | Added `"final_agent"` to `ROUTE_MAP`. If `itinerary_agent` is not selected, routes directly from the last specialist to `final_agent` without pausing for approval. |
| `src/agents/final.py` | Replaced hardcoded 7-section prompt with dynamic section generation based strictly on which agents ran. For hotel-only queries, omits flight/weather/itinerary sections. |
| `03-Travel-MultiAgent-frontend/src/components/FlowPanel.jsx` | Updated `buildSteps()`: only displays the `human` review step when approval was actually involved (`kind === 'draft'` or `message.approval`). |
| `evals/datasets/golden_queries.jsonl` | Updated single-domain queries so `expected_agents` accurately expects only the specific specialist. |
| `evals/test_routing.py` | Removed hardcoded `selected.add("itinerary_agent")` so routing precision measures true selective dispatching. |
| `scripts/test_targeted_routing.py` | **New** — Verifies hotel-only query runs in 1 turn without approval and full trip pauses for HITL approval. |

### Verification Results

```
============================================================
TEST 1: Hotel-only query ("nearest best hotels near guntur railwaystation")
============================================================
Selected Agents      : ['hotel_agent']
Requires Approval?   : False
Contains 'Flight'    : False (no hallucinated flight section)
Contains 'Itinerary' : False (no forced 1-day itinerary)
✓ TEST 1 PASSED: Only hotel_agent was called, no approval was forced.

============================================================
TEST 2: Full trip query ("Plan a 3-day trip from Hyderabad to Goa...")
============================================================
Selected Agents    : ['flight_agent', 'hotel_agent', 'budget_agent', 'itinerary_agent']
Requires Approval? : True
--- Resuming with Approval ---
✓ TEST 2 PASSED: Approved draft finalized cleanly without unrequested section duplication.
```

**Concept Covered:** Dynamic Subgraph Routing, Selective Specialist Dispatching, Conditional Human-in-the-Loop, Adaptive Response Synthesis.

---

## ✅ Phase 16 — Thread-Level Conversational Memory & Coreference Resolution

### What was the problem?

In multi-turn chat interactions, the system suffered from two critical memory breakdowns:
1. **Chat Agent In-Thread Amnesia:** When the user asked a conversational follow-up like *"can you tell me what i asked above?"*, the `chat_agent` replied *"It looks like you haven't asked a question yet in this chat..."*. This occurred because `chat_agent` only sent `[SystemMessage, HumanMessage(current_query)]` to the LLM, discarding the actual thread message history stored in LangGraph's PostgreSQL checkpointer.
2. **Coreference & Constraint Loss:** In a multi-turn travel planning session:
   - *Turn 1:* User asks *"what are the nearest best hotels near guntur railwaystation?"* $\rightarrow$ The agent provides hotels in Guntur.
   - *Turn 2:* User follows up: *"what are the flight options there from Hyderabad?"*
   - Because the query contained the relative pronoun *"there"*, naive query processing didn't know what *"there"* referred to. The destination constraint (`Guntur`) was dropped, causing flight search to fail or prompt the user again.
3. **Context Window & Latency Risks:** Adding an independent "Query Rewriter" LLM node adds an extra 1.5–2s of latency and token cost to *every single user turn*, which degrades production responsiveness and balloons LLM costs.

---

### The Solution: 4-Tier Hybrid Production Architecture (0 Extra LLM Calls)

To achieve production-grade speed, memory retention, and context window efficiency, we implemented a 4-tier hybrid memory strategy:

```
                      ┌────────────────────────────────────────┐
                      │              User Message              │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Load State & Thread History         │
                      │  (Sliding Window k=6 from Postgres)    │
                      └───────────────────┬────────────────────┘
                                          │
                        Is thread length ≥ 10 messages?
                                  /            \
                             Yes /              \ No
                                ▼                │
                    ┌──────────────────────┐     │
                    │   Rolling Summary    │     │
                    │ (Summarize k > 4 in  │     │
                    │   context_summary)   │     │
                    └───────────┬──────────┘     │
                                │                ▼
                                └─────────► Continue
                                                 │
                                                 ▼
             ┌───────────────────────────────────────────────────────────────────┐
             │       2-in-1 Supervisor Agent (Disambiguation + Routing)         │
             │                                                                   │
             │  • Receives active constraints + recent dialogue turns            │
             │  • Resolves pronouns ("there" → "Guntur", "that hotel" → name)   │
             │  • Inherits & updates trip_constraints (destination="Guntur")     │
             │  • Selects ONLY necessary specialist agents                       │
             │                                                                   │
             │              ★ ZERO EXTRA LLM CALLS ADDED! ★                      │
             └───────────────────────────────────┬───────────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
        [chat_intent]                                          [travel_intent]
               │                                                              │
               ▼                                                              ▼
    ┌──────────────────────────┐                               ┌──────────────────────────┐
    │        chat_agent        │                               │    Specialist Agents     │
    │  • System Prompt         │                               │  • Uses resolved_query   │
    │  • context_summary       │                               │  • Uses trip_constraints │
    │  • Sliding window (k=6)  │                               │    (destination, origin) │
    │  Full conversational     │                               │  • Executes live tools   │
    │  thread recall!          │                               └──────────────┬───────────┘
    └──────────────────────────┘                                              │
                                                                              ▼
                                                               ┌──────────────────────────┐
                                                               │       final_agent        │
                                                               │  Dynamic synthesis of    │
                                                               │  only executed results   │
                                                               └──────────────────────────┘
```

#### 1. Sliding Window History (Zero Extra LLM Calls)
- In `src/agents/chat.py`, `chat_agent` extracts the last 6 messages (`state["messages"][-6:]`) from LangGraph's checkpointer.
- It supplies the LLM with `[SystemMessage(content=prompt), *history_window]`, allowing the chat agent to accurately reference previous turns and answer conversational queries such as *"what was my previous question?"*.

#### 2. Constraint Inheritance (Zero LLM Calls)
- In `src/agents/graph.py`, `run_travel_agent()` inspects the current state checkpointer via `travel_graph.get_state(config)`.
- It preserves existing `trip_constraints` (destination, origin, budget, travelers) across turns so specialists always retain previously established trip parameters.

#### 3. 2-in-1 Supervisor Coreference Resolution & Routing (Zero Extra LLM Calls)
- Rather than inserting a separate query rewriter agent before the supervisor (which would add +1.5s latency to every turn), the `supervisor_agent` prompt was expanded with conversation context and active constraints.
- In a single LLM invocation, the supervisor:
  1. Identifies if the query contains relative references (*"there"*, *"that hotel"*, *"second option"*).
  2. Resolves it into `resolved_query` (e.g. *"what are the flight options to Guntur from Hyderabad?"*).
  3. Merges detected entities into `trip_constraints`.
  4. Selects the appropriate specialist agents.
- Specialist agents (`flight.py`, `hotel.py`, `weather.py`, `budget.py`) consume `state.get("resolved_query") or state.get("user_query")`.

#### 4. Conditional Rolling Summary (Run only on Long Threads $\ge 10$ messages)
- For short chats ($< 10$ messages), no summarization runs (0 overhead).
- When conversation exceeds 10 turns, `_generate_rolling_summary()` triggers, condensing turns older than the last 4 into 2–3 crisp bullet points in `state["context_summary"]`.
- This summary is prepended to the system prompts of `chat_agent` and `supervisor_agent`, allowing unbounded conversation depth without blowing up LLM context limits.

---

### Files Changed

| File | Change |
|------|--------|
| `src/agents/state.py` | Added `resolved_query: str` to `TravelState` and `initial_state()` to hold the disambiguated prompt. |
| `src/agents/chat.py` | Implemented sliding window memory (`state["messages"][-6:]`) and dynamic injection of `context_summary` into the LLM system prompt. |
| `src/agents/supervisor.py` | Upgraded prompt to include recent conversation turns (`history_lines`) and active constraints. Performs ambiguity resolution, constraint updating, and specialist routing in 1 unified call. |
| `src/agents/graph.py` | Updated `run_travel_agent()` to retrieve and preserve existing `trip_constraints` and `context_summary` from checkpointer across turns. Added `_generate_rolling_summary()`. Updated `serialize_result()` to return `resolved_query`. |
| `src/agents/specialists/flight.py` | Consumes `effective_query = state.get("resolved_query") or state.get("user_query")`. Inherits origin/destination from `trip_constraints`. |
| `src/agents/specialists/hotel.py` | Uses `effective_query = state.get("resolved_query") or state.get("user_query")` for live search. |
| `src/agents/specialists/weather.py` | Prioritizes destination from `state.get("trip_constraints", {}).get("destination")` and `resolved_query`. |
| `src/agents/specialists/budget.py` | Uses `effective_query` and inherited trip constraints. |
| `scripts/test_chat_memory.py` | **New** — Comprehensive multi-turn test simulating: (1) Hotel search, (2) Conversational recall of prior query, (3) Pronoun resolution ("there" $\rightarrow$ "Guntur"). |

---

### Verification Results

Ran end-to-end multi-turn test suite (`scripts/test_chat_memory.py`):

```
======================================================================
TEST 1: Initial query -> Hotels in Guntur
======================================================================
User Query       : what are the nearest best hotels near guntur railwaystation?
Intent           : travel
Selected Agents  : ['hotel_agent']
Resolved Query   : what are the nearest best hotels near guntur railwaystation?
Destination in State: Guntur
Requires Approval: False
✓ Turn 1 PASSED: Targeted hotel search executed and destination stored.

======================================================================
TEST 2: Conversational recall ("can you tell me what i asked above?")
======================================================================
User Query       : can you tell me what i asked above?
Intent           : chat
Selected Agents  : []
Chat Response    : You asked for the nearest best hotels near Guntur railway station.
Recalled Guntur? : True
Response not blank: True
✓ Turn 2 PASSED: Chat agent accurately recalled the earlier question in the same thread.

======================================================================
TEST 3: Relative follow-up ("what are the flight options there from Hyderabad?")
======================================================================
User Query       : what are the flight options there from Hyderabad?
Intent           : travel
Selected Agents  : ['flight_agent']
Resolved Query   : what are the flight options to Guntur from Hyderabad?
Origin in State  : Hyderabad
Destination in State: Guntur
Requires Approval: False
✓ Turn 3 PASSED: Supervisor resolved 'there' -> Guntur and dispatched flight_agent.

======================================================================
ALL THREAD MEMORY & CONVERSATIONAL TESTS PASSED 100%!
======================================================================
```

**Concept Covered:** Thread-Level State Checkpointing, Sliding Window Context, Coreference Resolution, Constraint Inheritance, Zero-LLM-Overhead Disambiguation, Rolling Summarization.

---

## ✅ Phase 17 — Dynamic User-Connected MCP Servers (Zero 3rd-Party Gateways)

### Overview
Users can now dynamically connect their own external MCP servers (Google Drive, Google Calendar, or any remote `streamable_http` / `sse` MCP URL via API Key or OAuth 2.1 + PKCE) directly from the UI without hardcoding anything in Python.

### Architecture & File Structure

```
src/
├── core/
│   ├── config.py                    ← Added MCP_ENCRYPTION_KEY, GOOGLE_OAUTH_CLIENT_ID/SECRET, OAUTH_REDIRECT_BASE_URL
│   └── encryption.py                ← Fernet (AES-128 + HMAC) encryption for stored user tokens/API keys
├── mcp_dynamic/
│   ├── __init__.py
│   ├── catalog.py                   ← Curated MCP catalog (Google Drive, Calendar, Custom) + SSRF URL validator
│   ├── store.py                     ← PostgreSQL `user_mcp_connections` table + CRUD (secrets stripped on read)
│   ├── oauth.py                     ← Google OAuth2 + Standard MCP OAuth 2.1 (.well-known + RFC 7591 + PKCE) + auto-refresh
│   └── client.py                    ← 1-time remote tool discovery + lazy per-request LangChain tool loader
├── api/routers/
│   └── mcp_servers.py               ← /api/mcp/catalog, /servers, /connect, /oauth/callback, PATCH, DELETE
└── agents/specialists/
    └── user_mcp_agent.py            ← Lazy ReAct tool-calling agent + structured `mcp_trace` for right-panel UI
```

### How It Works at Runtime (Zero-Overhead Optimization)
1. **On Connect:** FastAPI validates the URL (blocking private/loopback SSRF IPs), performs a 1-time `get_tools()` handshake (or OAuth 2.1 PKCE popup), encrypts credentials with Fernet, and caches `cached_tools` in PostgreSQL (`user_mcp_connections`).
2. **Normal Queries (`0ms` overhead):** `_prepare_turn()` reads `cached_tools` from PostgreSQL in parallel. If the user asks *"hotels in Goa"*, `user_mcp_agent` is never selected and no external MCP connections are opened.
3. **External Action Queries:** When the user says *"Plan a 3-day trip to Goa and save it to Google Drive"* (or *"Save that plan to my Google Drive"*), Supervisor includes `user_mcp_agent`, which runs after `final_agent` (or standalone), decrypts & auto-refreshes OAuth tokens, executes the tool, and records `mcp_trace` for the right-side **Agent Flow** panel.

### How Users Connect External MCPs

#### 1. Google Drive / Google Calendar (One-Time Admin Setup in `.env`)
1. In Google Cloud Console → **APIs & Services → Credentials**, create an **OAuth 2.0 Web Client**.
2. Enable **Google Drive API**, **Google Docs API**, and **Google Calendar API**.
3. Add Authorized Redirect URI: `http://localhost:8000/api/mcp/oauth/callback`.
4. Add to `.env`:
   ```env
   GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
   OAUTH_REDIRECT_BASE_URL=http://localhost:8000
   ```
5. In the app sidebar, click **MCP Integrations** → click **Connect** on **Google Drive** or **Google Calendar** → sign in via the popup.

#### 2. Any Custom Remote MCP Server (URL + API Key or OAuth 2.1)
1. Click **MCP Integrations** → **Add MCP Server** → scroll to **Custom Remote MCP**.
2. Enter a display name, transport (`streamable_http` or `sse`), the HTTPS URL, and an optional API Key / Bearer Token (leave blank for open or OAuth 2.1 servers).
3. Click **Connect & Discover Tools** — TripMate validates the URL, discovers its tools, and makes them available immediately.

**Concept Covered:** Multi-Tenant Dynamic MCP Discovery, Fernet Secret Encryption, OAuth 2.1 + PKCE, SSRF Protection, Lazy Per-Request Tool Binding, UI Tool Execution Tracing.
