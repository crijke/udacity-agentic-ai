# Munder Difflin — Multi-Agent Workflow Diagram

```mermaid
flowchart TD
    REQ["Customer quote request<br/>(free text + request date)"] --> ORC

    subgraph ORC ["ORCHESTRATOR agent"]
        direction TB
        S1[["1. assess_inventory"]] --> S2[["2. generate_quote<br/>(only if ≥1 item fulfillable)"]] --> S3[["3. finalize_order"]]
    end

    S1 --> INV
    S2 --> QUO
    S3 --> ORD

    subgraph INV ["INVENTORY agent"]
        I1[["Extract items + quantities,<br/>map to exact catalog names,<br/>extract needed-by date"]]
        I2[["then, in Python: check_item_fulfillment<br/>per matched item — stock / restock /<br/>deadline / cash (deterministic)"]]
        I1 --> I2
    end

    subgraph QUO ["QUOTING agent"]
        Q1[["tool: calculate_quote<br/>prices + bulk discount<br/>(deterministic, authoritative)"]]
        Q2[["tool: find_similar_quotes<br/>historical precedent"]]
    end

    subgraph ORD ["ORDERING agent"]
        O1[["tool: finalize_sale<br/>restock-to-fulfill + record sales<br/>(deterministic, all-or-nothing)"]]
    end

    ORC --> O2[["after every request, in Python:<br/>restock_low_inventory<br/>top up below-minimum items<br/>(runs even with no sale)"]]

 

    ORC --> REPLY["Customer-facing reply:<br/>quote, discount, delivery date,<br/>declined items + reasons"]
```

**Data flow note:** structured results (assessment, quote) pass between steps
through a Python-side context dict; the orchestrator LLM only sequences the
calls and writes the final reply. Inter-agent messages are JSON text.
