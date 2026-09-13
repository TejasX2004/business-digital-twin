# B-Twin --- Business Digital Twin & Autonomous Operations Platform

B-Twin is an AI-powered business intelligence and autonomous operations
platform built around a **living Business Digital Twin**.

The system combines transactional business data, deterministic
simulation, Retrieval-Augmented Generation (RAG), multi-agent reasoning,
Red-Team validation, MCP-based tool use, and human-in-the-loop
operational actions.

> **Business data changes → the Digital Twin updates → AI agents
> investigate the new state → scenarios are simulated → company policies
> are retrieved → recommendations are validated → approved actions can
> update the business.**

------------------------------------------------------------------------

## Architecture Overview

``` text
                         ┌──────────────────────┐
                         │       Streamlit      │
                         │      Dashboard       │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
            Standard Mode                      Agentic Mode
                  │                                   │
                  ▼                                   ▼
            Orchestrator                    Operations Agent
                  │                                   │
                  ▼                              MCP Tools
              Planner                              │
                  │                    ┌─────────────┼─────────────┐
                  ▼                    ▼             ▼             ▼
              Scenario             Digital Twin  Simulator       RAG
                  │                    │             │             │
                  ▼                    ▼             ▼             ▼
             Simulator            PostgreSQL    What-If Engine  FAISS
                  │
                  ▼
              Analysis
                  │
                  ▼
             Red-Team
                  │
                  ▼
              Decision
                  │
                  ▼
          Human Approval
                  │
                  ▼
            Event System
                  │
                  ▼
              PostgreSQL
                  │
                  └──────────► Digital Twin Refresh
```

------------------------------------------------------------------------

# 1. Core Architecture

B-Twin deliberately separates **data, computation, knowledge, reasoning,
validation, and action**.

### PostgreSQL --- Source of Truth

PostgreSQL stores the actual business state:

-   Customers
-   Products
-   Sales
-   Inventory
-   Expenses
-   Payments
-   Business events

AI agents never directly execute arbitrary SQL or modify the database.

### Digital Twin --- Derived Business State

The Digital Twin queries PostgreSQL and calculates the current business
state, including:

-   Revenue
-   COGS
-   Gross profit and margin
-   Operating expenses
-   Net profit and margin
-   Cash
-   Accounts receivable
-   Inventory value and units
-   Payment behavior
-   DSO
-   Inventory turnover and coverage
-   Inventory risk
-   Receivable exposure
-   Overall business risk

The Digital Twin is **derived state**, not a second source of truth.

### Deterministic Simulator

The simulator takes the current Digital Twin state and a structured
scenario and calculates projected financial and operational impact.

Supported scenario parameters include:

``` text
sales_growth
price_change
cost_change
payment_delay_days
inventory_change
expense_change
supplier_payment_delay_days
months
```

The simulator is deterministic and does not rely on an LLM for financial
calculations.

------------------------------------------------------------------------

# 2. Standard Decision Pipeline

The Standard Mode uses the validated fixed workflow:

``` text
User Question
      ↓
Planner Agent
      ↓
Structured Scenario
      ↓
Digital Twin
      ↓
Deterministic Simulator
      ↓
Analysis Agent
      ↓
Red-Team Agent
      ↓
Decision Agent
      ↓
Final Recommendation
```

## Planner Agent

Converts natural-language requests into validated structured scenarios.

Example:

``` text
"What happens if sales increase by 30% and customers pay 15 days later?"
```

becomes:

``` json
{
  "sales_growth": 0.30,
  "payment_delay_days": 15
}
```

The Planner validates parameter bounds before the simulator receives the
scenario.

## Analysis Agent

Combines:

``` text
AUTHORITATIVE SIMULATOR FACTS
+
SCENARIO
+
RETRIEVED BUSINESS KNOWLEDGE
```

Simulator values have priority for numerical/current-state facts.

The Analysis Agent produces structured sections such as:

-   Executive summary
-   Key drivers
-   Financial impact
-   Cash-flow impact
-   Working-capital impact
-   Inventory analysis
-   Risk analysis
-   Tradeoffs
-   Recommendations
-   Overall assessment
-   Confidence

## Red-Team Agent

Independently challenges the Analysis Agent for:

-   Numerical inconsistencies
-   Incorrect trends
-   Contradictions
-   Unsupported claims
-   Invented assumptions or benchmarks
-   Incorrect risk interpretation
-   Unsupported recommendations
-   Missing material risks
-   Overconfidence

## Decision Agent

Consumes the scenario, authoritative simulator facts, analysis, and
Red-Team verdict to produce:

-   Decision
-   Priority
-   Rationale
-   Recommended actions
-   Expected benefits
-   Risks
-   Conditions
-   Confidence

------------------------------------------------------------------------

# 3. Event-Driven Digital Twin

The Digital Twin can react to operational events.

Supported event types include:

``` text
SALE_CREATED
PAYMENT_RECEIVED
INVENTORY_UPDATED
EXPENSE_RECORDED
SUPPLIER_PAYMENT_UPDATED
```

The flow is:

``` text
Business Event
      ↓
Event Processor
      ↓
Validation
      ↓
PostgreSQL Transaction
      ↓
COMMIT
      ↓
Digital Twin Rebuild
      ↓
Updated Business State
```

The event does not directly calculate the new Digital Twin values.

> **The event changes PostgreSQL, and the Digital Twin recalculates its
> state from PostgreSQL.**

The Event System provides validation, parameterized SQL, transactions,
rollback, event logging, and idempotency.

------------------------------------------------------------------------

# 4. RAG Knowledge Layer

RAG provides company-specific knowledge without replacing authoritative
business data.

``` text
Business Documents
       ↓
Document Ingestion
       ↓
Chunking + Metadata
       ↓
Local Embeddings
       ↓
FAISS
       ↓
Retriever
       ↓
Relevant Business Knowledge
       ↓
AI Reasoning
```

Example documents:

``` text
inventory_policy.md
payment_policy.md
supplier_policy.md
expense_policy.md
business_rules.md
```

Retrieved knowledge preserves:

-   Source
-   Document
-   Chunk ID
-   Relevance score
-   Content

RAG provides policies, rules, definitions, and context.

It does **not** override PostgreSQL or simulator values.

------------------------------------------------------------------------

# 5. Autonomous Operations Agent

Agentic Mode adds a capability that the fixed pipeline does not have.

Instead of requiring a fully specified scenario, the user can provide a
broad objective:

> "Find the biggest business risk right now."

The Operations Agent dynamically decides which tools it needs.

Example:

``` text
get_business_snapshot()
        ↓
get_customer_exposure()
        ↓
get_payment_metrics()
        ↓
get_inventory_status()
        ↓
search_business_policy()
        ↓
run_business_simulation()
        ↓
compare options
        ↓
recommendation
```

The agent is not required to call every tool. Tool selection is driven
by the investigation.

------------------------------------------------------------------------

# 6. MCP Layer

MCP provides the Operations Agent with a controlled tool interface.

It is **not inserted into every step of the Standard Mode pipeline**.

``` text
                 Operations Agent
                        │
                        ▼
                   MCP Server
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    Business         Simulation        Knowledge
      Tools             Tools            Tools
        │               │                │
        ▼               ▼                ▼
   Digital Twin      Simulator           RAG
        │
        ▼
   Event System
        │
        ▼
   PostgreSQL
```

Representative tools include:

``` text
get_business_snapshot()
get_sales_summary()
get_inventory_status()
get_customer_exposure()
get_cash_position()
get_payment_metrics()

run_business_simulation()
compare_scenarios()

search_business_policy()

create_business_event()
```

MCP tools expose controlled business capabilities rather than arbitrary
database access.

------------------------------------------------------------------------

# 7. Business Actions vs Business Events

The platform distinguishes between:

### Events

Things that happened:

``` text
SALE_CREATED
PAYMENT_RECEIVED
INVENTORY_UPDATED
```

### Actions

Things management proposes to do:

``` text
REQUEST_CREDIT_REVIEW
RESTRICT_CUSTOMER_CREDIT
INITIATE_COLLECTION_REVIEW
FLAG_CUSTOMER_RISK
```

Therefore:

``` text
EVENT  = observed business fact
ACTION = proposed management intervention
```

An action must pass deterministic validation and human approval before
execution.

If an action has no safe automated implementation, it becomes:

``` text
REQUIRES_MANUAL_ACTION
```

rather than pretending that the action was executed.

------------------------------------------------------------------------

# 8. Human-in-the-Loop Safety

Business-changing actions follow:

``` text
Operations Agent
       ↓
Proposed Action
       ↓
Deterministic Action Validator
       ↓
Validated Action
       ↓
Human Approval
       ↓
Action Executor
       ↓
Existing Event System
       ↓
PostgreSQL
       ↓
Digital Twin Refresh
       ↓
Verification
```

The LLM cannot:

-   Execute arbitrary SQL
-   Modify simulator results
-   Invent financial values
-   Invent unsupported thresholds
-   Invent unsupported action parameters
-   Modify business state without approval

The action lifecycle can use states such as:

``` text
PROPOSED
VALIDATING
VALIDATED
AWAITING_APPROVAL
APPROVED
EXECUTED
REQUIRES_MANUAL_ACTION
REJECTED
FAILED
```

------------------------------------------------------------------------

# 9. Autonomous Investigation Loop

The Control Tower follows:

``` text
OBSERVE
   ↓
IDENTIFY ISSUE
   ↓
INVESTIGATE
   ↓
SIMULATE
   ↓
CHECK POLICY
   ↓
COMPARE OPTIONS
   ↓
RECOMMEND
   ↓
HUMAN APPROVAL
   ↓
ACT
   ↓
VERIFY
```

The agent can stop when enough information is available and does not
need to call every tool.

------------------------------------------------------------------------

# 10. Standard Mode vs Agentic Mode

## Standard Mode

Best for predictable scenario analysis.

``` text
Question
  ↓
Planner
  ↓
Scenario
  ↓
Simulator
  ↓
Analysis
  ↓
Red-Team
  ↓
Decision
```

## Agentic Mode

Best for open-ended investigation.

``` text
Objective
   ↓
Operations Agent
   ↓
Dynamic MCP Tool Selection
   ↓
Observe / Investigate / Simulate / Retrieve
   ↓
Compare Options
   ↓
Recommendation
   ↓
Human Approval
   ↓
Action / Manual Workflow
   ↓
Verify
```

Both modes use the same authoritative business systems.

------------------------------------------------------------------------

# 11. Streamlit Dashboard

The Streamlit application provides the operational interface.

Main areas include:

``` text
Dashboard
Business Events
What-If Lab
Scenario Comparison
Autonomous Control Tower
```

### Dashboard

Displays the current Digital Twin state.

### Business Events

Allows users to submit supported operational events and observe
resulting business-state changes.

### What-If Lab

Allows deterministic scenario configuration.

### Scenario Comparison

Compares alternative simulations.

### Autonomous Control Tower

Provides:

``` text
Objective
   ↓
Investigation Timeline
   ↓
Key Findings
   ↓
Simulations & Policies
   ↓
Options Considered
   ↓
Strategic Recommendation
   ↓
Action Validation
   ↓
Human Approval
```

------------------------------------------------------------------------

# 12. Technology Stack

  Layer                Technology
  -------------------- ------------------------------------------------
  Language             Python
  Database             PostgreSQL
  LLM Runtime          Ollama
  LLM                  Qwen
  Vector Database      FAISS
  Embeddings           Local embedding model
  Knowledge            RAG
  Agent Architecture   Multi-agent + Autonomous Operations Agent
  Tool Protocol        MCP
  UI                   Streamlit
  Testing              Python unittest + deterministic scenario tests

------------------------------------------------------------------------

# 13. Project Structure

``` text
business-digital-twin/
│
├── agents/
│   ├── planner_agent.py
│   ├── analysis_agent.py
│   ├── red_team_agent.py
│   ├── decision_agent.py
│   ├── operations_agent.py
│   └── action_validator.py
│
├── backend/
│   ├── digital_twin/
│   │   ├── digital_twin.py
│   │   └── simulation.py
│   ├── orchestration/
│   │   ├── pipeline.py
│   │   └── state.py
│   └── seed_database.py
│
├── events/
│   ├── event_types.py
│   ├── event_store.py
│   ├── event_processor.py
│   └── handlers.py
│
├── actions/
│   ├── action_types.py
│   ├── action_validator.py
│   ├── action_executor.py
│   └── schemas.py
│
├── firewall/
│   └── deterministic validators
│
├── mcp/
│   ├── server.py
│   ├── tools.py
│   ├── schemas.py
│   └── README.md
│
├── rag/
│   ├── documents/
│   │   ├── inventory_policy.md
│   │   ├── payment_policy.md
│   │   ├── supplier_policy.md
│   │   ├── expense_policy.md
│   │   └── business_rules.md
│   ├── ingestion.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── rag_service.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── synthetic business data
│
├── tests/
│   ├── test_pipeline.py
│   ├── scenarios.py
│   └── README.md
│
└── .env
```

------------------------------------------------------------------------

# 14. Complete Data and Decision Flow

``` text
                         USER
                           │
            ┌──────────────┴──────────────┐
            │                             │
       Standard Mode                 Agentic Mode
            │                             │
       Orchestrator                Operations Agent
            │                             │
         Planner                         MCP
            │                             │
         Scenario            ┌────────────┼────────────┐
            │                ▼            ▼            ▼
            └────────────► Digital     Simulator      RAG
                            Twin
                              │
                              ▼
                         PostgreSQL
                              │
                              ▼
                         Event System
                              │
                              ▼
                         Updated State
                              │
                              ▼
                          Analysis
                              │
                              ▼
                          Red-Team
                              │
                              ▼
                           Decision
                              │
                              ▼
                       Action Validator
                              │
                              ▼
                       Human Approval
                              │
                              ▼
                       Business Action
                              │
                              ▼
                       Event System
                              │
                              └──────────► PostgreSQL
```

------------------------------------------------------------------------

# 15. Reliability Model

B-Twin follows a **facts-first architecture**.

  Responsibility           Authoritative source
  ------------------------ -------------------------------------
  Current business state   PostgreSQL / Digital Twin
  Financial calculations   Deterministic simulator
  Company policies         RAG documents
  Reasoning                LLM agents
  Claim validation         Red-Team + deterministic validators
  Business mutations       Event System
  Consequential actions    Human approval

This separation prevents an LLM from becoming an unrestricted database
operator.

------------------------------------------------------------------------

# 16. Testing Strategy

The project uses deterministic and live AI tests.

The scenario suite covers:

-   Baseline
-   Sales growth and decline
-   Price changes
-   Cost changes
-   Expense changes
-   Payment delays
-   Inventory changes
-   Supplier payment delays
-   Multiple simultaneous changes
-   Extreme valid scenarios

Tests verify:

-   Planner parameter extraction
-   Simulator financial reconciliation
-   Analysis schema and numerical fidelity
-   Red-Team validation
-   Decision consistency
-   Cross-pipeline fidelity
-   RAG retrieval
-   MCP tool behavior
-   Event processing
-   Action safety
-   Human-approval boundaries

The deterministic simulator tests are designed to remain stable
independently of LLM output.

------------------------------------------------------------------------

# 17. Example

User:

> "Sales are expected to increase by 30%. Should we increase inventory?"

### Standard Mode

``` text
Planner
→ sales_growth = 0.30

Simulator
→ revenue
→ cash
→ receivables
→ inventory
→ coverage
→ risk

RAG
→ inventory policy

Analysis
→ financial and operational impact

Red-Team
→ validates claims

Decision
→ recommendation
```

### Agentic Mode

The Operations Agent may dynamically perform:

``` text
get_business_snapshot()
        ↓
get_inventory_status()
        ↓
run_business_simulation(+30% sales)
        ↓
run_business_simulation(+30% sales, +10% inventory)
        ↓
run_business_simulation(+30% sales, +20% inventory)
        ↓
search_business_policy("inventory policy")
        ↓
compare options
        ↓
recommendation
```

If a business action is appropriate:

``` text
Recommendation
      ↓
Action Validator
      ↓
Human Approval
      ↓
Event System
      ↓
PostgreSQL
      ↓
Digital Twin
      ↓
Verification
```

------------------------------------------------------------------------

# 18. Key Differentiator

B-Twin is more than:

``` text
LLM + RAG + Dashboard
```

Its core differentiator is a **living business environment** where AI
reasoning is grounded in deterministic business state.

``` text
                 LIVING BUSINESS ENVIRONMENT
                           │
                           ▼
                      Digital Twin
                           │
                           ▼
                  Deterministic Simulation
                           │
                           ▼
                    Multi-Agent Reasoning
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
               RAG                 Red-Team
                │                     │
                └──────────┬──────────┘
                           ▼
                       Decision
                           │
                    Human Approval
                           │
                           ▼
                     Business Action
                           │
                           ▼
                      Event System
                           │
                           ▼
                       PostgreSQL
                           │
                           └──────► Twin Refresh
```

The overall closed-loop concept is:

> **Observe → Understand → Simulate → Reason → Validate → Decide → Act →
> Verify**

------------------------------------------------------------------------

## Running the Project

Configure PostgreSQL and the required environment variables, then run:

``` bash
streamlit run dashboard/app.py
```

Follow the project's environment/dependency configuration for database
setup and model availability.

------------------------------------------------------------------------

## Project Status

-   [x] PostgreSQL business data
-   [x] Business Digital Twin
-   [x] Deterministic simulation engine
-   [x] Natural-language scenario planning
-   [x] Multi-agent analysis pipeline
-   [x] Red-Team validation
-   [x] Decision layer
-   [x] Event-driven business updates
-   [x] RAG knowledge layer
-   [x] Autonomous Operations Agent
-   [x] MCP-based agentic tools
-   [x] Human-in-the-loop action safety
-   [x] Streamlit Control Tower
-   [x] Automated testing

------------------------------------------------------------------------

## Philosophy

> **LLMs should reason over business reality, not invent business
> reality.**

PostgreSQL provides the state.

The Digital Twin derives the state.

The simulator predicts deterministic outcomes.

RAG provides organizational knowledge.

Agents reason over those inputs.

Red-Team challenges the reasoning.

Deterministic validators enforce safety.

Humans approve consequential actions.

The Event System changes the business state.

The Digital Twin observes the new state and the cycle continues.
