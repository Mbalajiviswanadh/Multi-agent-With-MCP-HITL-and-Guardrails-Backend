# TripMate AI — Multi-Agent Travel Planner with MCP, 5-Tier Memory & Guardrails

A production-grade, multi-agent AI travel assistant built with **FastAPI**, **LangGraph**, **Groq**, **Mem0 + Qdrant**, **PostgreSQL + pgvector**, **Model Context Protocol (MCP)**, and a **React + Redux** frontend.

![TripMate AI Architecture](images/architecture_v2.png)

---

## Key Capabilities

- **Multi-Agent Travel Planning:** Supervisor-orchestrated specialist agents (`flight_agent`, `hotel_agent`, `weather_agent`, `budget_agent`, `itinerary_agent`, `final_agent`) powered by LangGraph.
- **Human-in-the-Loop (HITL) Review:** Pauses before `human_approval_agent` / `final_agent` so users can review, edit, or approve draft itineraries before finalization.
- **5-Tier Hybrid Memory with On-Demand `pgvector`:** Combines thread checkpoints, rolling summaries, Mem0 + Qdrant user facts, stateless constraint inheritance, and lazy `pgvector` trip recall.
- **Dynamic User MCP & 3-Stage Tool RAG:** Users can connect Google Drive, Google Calendar, Notion, or custom remote MCPs at runtime while keeping Groq token usage within limits (94% token reduction).
- **4-Intent LLM Router & Guardrails:** Single-call intent classification (`travel` | `mcp` | `chat` | `blocked`) with fail-open safety and post-execution output guardrails.
- **Real-Time SSE Observability:** Streams live agent steps, memory tier diagnostics, and MCP tool filtering traces directly to the frontend `FlowPanel` and `MemoryPanel`.

---

## System Architecture & LangGraph Flow

TripMate AI is organized into 6 layers: **React/Redux Frontend** → **FastAPI + SSE Gateway** → **Guardrails & 4-Intent Router** → **LangGraph Multi-Agent Orchestration** → **5-Tier Memory** → **PostgreSQL/`pgvector`, Qdrant Cloud, and MCP Servers**.

```mermaid
flowchart LR
    START((User Query)) --> IG["Input Guardrail<br/>4-Intent LLM Router"]

    IG -->|"blocked"| BLK["guardrail_blocked"] --> END((END))
    IG -->|"chat"| CHAT["chat_agent<br/>(On-Demand pgvector)"] --> END
    IG -->|"mcp (short-circuit)"| UMCP["user_mcp_agent<br/>(3-Stage Tool RAG)"] --> END

    IG -->|"travel"| SUP["supervisor_agent"]
    SUP --> SPEC["flight / hotel / weather / budget"]
    SPEC --> ITIN["itinerary_agent"]
    ITIN --> HITL{"human_approval_agent<br/>(HITL Interrupt)"}
    HITL -->|"Approved"| FINAL["final_agent"] --> END
    HITL -->|"Revise"| ITIN
```

---

## Core Engineering Strategies & Optimizations

### 1. 5-Tier Hybrid Memory & On-Demand `pgvector`
| Tier | Mechanism | Scope | Storage | Optimization |
|---|---|---|---|---|
| **Tier 1** | Thread Checkpoint + Sliding Window | In-Thread | PostgreSQL (`PostgresSaver`) | Bounds active context while persisting full resumable state |
| **Tier 2** | Rolling Summary | In-Thread | In-State (LLM Summary) | Compresses older turns automatically at **10+ turns** |
| **Tier 3** | Long-Term User Facts | Cross-Session | **Mem0 + Qdrant Cloud** | Extracts durable preferences (diet, home city, travel style) |
| **Tier 4** | Constraint Inheritance | In-Thread | Stateless Merge | Preserves origin, destination, budget, and dates with **0 DB calls** |
| **Tier 5** | Episodic Past Trips | Cross-Session | **PostgreSQL `pgvector`** | **On-Demand Only:** gated by `needs_past_trips` (~500 tokens/turn saved) |

- **On-Demand Gate:** Casual queries (e.g., *"Hi!"*) set `needs_past_trips = False`, resulting in **0 `pgvector` DB calls**. Recall queries (e.g., *"What was my last trip?"*) trigger `get_recent_trips()` (`ORDER BY created_at DESC`) and `find_similar_trips()` (384-dim `FastEmbed all-MiniLM-L6-v2` IVFFlat cosine similarity).

### 2. 3-Stage Tool RAG (77+ MCP Tools → Top-6)
1. **Stage 1 — Server Router:** Uses `target_mcp_servers` from the LLM Router (with `_filter_target_servers()` regex fallback) to connect only to relevant MCP servers.
2. **Stage 2 — `FastEmbed` Semantic RAG:** Embeds the query and cached tool descriptions locally on CPU (`all-MiniLM-L6-v2`), applies cosine similarity + tool-name keyword boost, and selects the **Top-6 tools** (`0` LLM tokens).
3. **Stage 3 — Schema Compaction:** `_compact_tool_for_llm()` strips verbose JSON Schema definitions, reducing Notion schemas from **41,299 tokens → 4,072 tokens** (~90% reduction).

### 3. Guardrails, Routing & Reliability
- **Single-Call 4-Intent Router (`src/guardrails/input_guardrail.py`):** Outputs `intent`, `needs_past_trips`, `target_servers`, and `resolved_query` in one call; fails open to `chat` on errors.
- **MCP Short-Circuit (`src/agents/supervisor.py`):** Pure `mcp` intents route directly to `user_mcp_agent`, saving 1 LLM call.
- **LLM Fallback & Circuit Breaker (`src/reliability/llm_factory.py`):** Primary Groq `gpt-oss-120b` falls back to `gpt-oss-20b` on HTTP 429/413 with a 1-hour TPD circuit breaker and `bind_tools_resilient()`.
- **Output Guardrail (`src/guardrails/output_guardrail.py`):** Scans final responses, detects tool failures via `_is_tool_error()`, and preserves URLs via `_fallback_trace_summary()`.

### 4. Security
- **Fernet AES Encryption (`src/core/encryption.py`):** All OAuth tokens and API keys in `user_mcp_connections` are encrypted at rest.
- **SSRF Protection (`src/mcp_dynamic/catalog.py`):** `validate_remote_mcp_url()` blocks RFC 1918 private networks, loopback, and cloud metadata IPs.
- **JWT Authentication (`src/api/routers/auth.py`):** `bcrypt` password hashing + HS256 JWT tokens.

---

## MCP Integrations

### App-Level MCPs (All Users)
| Server | Transport | Purpose |
|---|---|---|
| **Tavily Search** | `streamable_http` | Live travel web search |
| **AviationStack** | `stdio` (`uvx`) | Real-time flight availability & pricing |
| **Custom Weather MCP** | `stdio` (`FastMCP`) | Weather lookup via `wttr.in` (`mymcp/custom_weather_mcp.py`) |

### User-Level Dynamic MCPs (Per User)
- **Built-In:** Google Drive (6 tools), Google Calendar (2 tools)
- **Remote:** Notion and any custom remote MCP server
- **3 Auth Modes:**
  1. API Key / Bearer Token (Fernet-encrypted)
  2. OAuth 2.1 `.well-known` + RFC 7591 Dynamic Client Registration + PKCE (e.g., Notion)
  3. Google OAuth2 (auto-detected for `*.googleapis.com` URLs)

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend API** | Python 3.11+, FastAPI, Uvicorn, Pydantic Settings, SSE |
| **Orchestration** | LangGraph (`StateGraph`, `PostgresSaver`, HITL), LangChain |
| **LLM & Embeddings** | Groq (`gpt-oss-120b`, `gpt-oss-20b`), `FastEmbed` (`all-MiniLM-L6-v2`, 384-dim) |
| **Memory & Databases** | PostgreSQL 16 + `pgvector`, `psycopg3` pool, Mem0 + Qdrant Cloud |
| **MCP** | `mcp`, `langchain-mcp-adapters` (`MultiServerMCPClient`), `FastMCP` |
| **Frontend** | React 18, Vite, Redux Toolkit, React Router, Tailwind CSS, Framer Motion, Lucide |

---

## Project Structure

```text
03-MultiAgent-MCP-Guardrails/
├── app.py                          # FastAPI entry point & router mounting
├── docker-compose.yml              # Local PostgreSQL 16 + pgvector container
├── pyproject.toml                  # Python dependencies & project metadata
├── mymcp/                          # App-level MCPs (Tavily, AviationStack, Custom Weather)
├── src/
│   ├── agents/                     # graph.py, supervisor.py, chat.py, itinerary.py, final.py, state.py
│   │   └── specialists/            # flight, hotel, weather, budget, user_mcp_agent.py (3-Stage Tool RAG)
│   ├── api/routers/                # auth.py, travel.py (SSE + HITL), threads.py (paginated), mcp_servers.py
│   ├── guardrails/                 # input_guardrail.py (4-intent router), output_guardrail.py
│   ├── memory/                     # vector_store.py, mem0_service.py, thread_memory.py, checkpointer.py, profile_store.py
│   ├── mcp_dynamic/                # catalog.py (SSRF guard), oauth.py, store.py, client.py
│   ├── reliability/                # llm_factory.py (fallback + circuit breaker), retry_policy.py
│   └── core/                       # config.py, database.py, encryption.py, logging_config.py
├── scripts/
│   └── test_memory_recall.py       # 4-part memory & on-demand pgvector test suite
└── images/                         # architecture.jpg, architecture_new.jpg, graph.png
```

---

## Quick Start

### 1. Start PostgreSQL + `pgvector`
```bash
docker compose up -d
```
This launches `pgvector/pgvector:pg16` on `localhost:5432` (database: `tripagentmemo`).

### 2. Install Backend Dependencies
Using `uv` (recommended, `uv.lock` included):
```bash
uv sync
```
Or with `pip`:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/tripagentmemo
GROQ_API_KEY=gsk_...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...
TAVILY_API_KEY=tvly-...
AVIATIONSTACK_API_KEY=...
SECRET_KEY=your-jwt-secret
FERNET_KEY=your-base64-fernet-key
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/api/mcp/oauth/callback
FRONTEND_URL=http://localhost:5173
```

### 4. Run the Backend
```bash
uvicorn app:app --reload --port 8000
```

### 5. Run the Frontend (separate directory)
```bash
cd ../03-Travel-MultiAgent-frontend
npm install
npm run dev
```

---

## Testing & Verification

Run the memory recall & on-demand `pgvector` test suite:
```bash
python scripts/test_memory_recall.py
```
Covers:
1. Short-term thread sliding window + constraint inheritance
2. `chat_agent` on-demand `pgvector` gating (0 DB calls on casual `"Hi!"`)
3. `itinerary_agent` Mem0 + past trips injection
4. Live PostgreSQL `pgvector` round-trip (`store_trip_embedding`, `find_similar_trips`, `get_recent_trips`)

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` & `/api/auth/login` | User registration & JWT login |
| `GET` | `/api/threads?page=1&page_size=15` | Paginated thread list (`total`, `has_more`) |
| `PATCH` | `/api/threads/{thread_id}` | Inline thread rename |
| `POST` | `/api/chat/{thread_id}` | Start agent run |
| `GET` | `/api/stream/{thread_id}` | SSE stream (`agent_started`, `live_step`, `memory_trace`, `mcp_trace`, `done`) |
| `POST` | `/api/approve/{thread_id}` | Resume HITL graph after itinerary review |
| `GET` | `/api/mcp/catalog` | List available dynamic MCP servers |
| `POST` | `/api/mcp/connect` | Connect API-key / Bearer MCP server |
| `GET` | `/api/mcp/oauth/start` & `/callback` | OAuth 2.1 / Google OAuth connection flow |

---

## License

See [LICENSE](LICENSE) for details.
