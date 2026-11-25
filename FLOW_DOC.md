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
