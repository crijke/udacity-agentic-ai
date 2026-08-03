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
        I2[["then, in Python: check_item_fulfillment<br/>per matched item — stock / restock /<br/>deadline / cash (deterministic)<br/><i>helpers: get_stock_level,<br/>get_supplier_delivery_date, get_cash_balance</i>"]]
        I1 --> I2
    end

    subgraph QUO ["QUOTING agent"]
        Q1[["tool: calculate_quote<br/>line prices + subtotal (deterministic,<br/>from inventory table)"]]
        Q2[["tool: find_similar_quotes<br/>discount precedent<br/><i>helper: search_quote_history</i>"]]
        Q3[["tool: apply_discount<br/>agent-chosen % (base rule in prompt,<br/>capped in code), rounds down to whole $"]]
        Q1 --> Q3
        Q2 --> Q3
    end

    subgraph ORD ["ORDERING agent"]
        O1[["tool: finalize_sale<br/>restock-to-fulfill + record sales<br/>(deterministic, all-or-nothing)<br/><i>helpers: get_stock_level,<br/>get_supplier_delivery_date, get_cash_balance,<br/>create_transaction, generate_financial_report</i>"]]
    end

    ORC --> O2[["after every request, in Python:<br/>restock_low_inventory<br/>top up below-minimum items<br/>(runs even with no sale)<br/><i>helpers: get_all_inventory, get_cash_balance,<br/>get_supplier_delivery_date, create_transaction</i>"]]

    ORC --> REPLY["Customer-facing reply:<br/>quote, discount, delivery date,<br/>declined items + reasons"]
```

**Tool → starter-helper map:** `check_item_fulfillment` and `finalize_sale`
build on `get_stock_level`, `get_supplier_delivery_date`, `get_cash_balance`
and `create_transaction` (finalize also reports via
`generate_financial_report`); `find_similar_quotes` wraps
`search_quote_history`; `restock_low_inventory` builds on
`get_all_inventory`, `get_cash_balance`, `get_supplier_delivery_date` and
`create_transaction`. All seven provided helper functions are used.

**Data flow note:** structured results (assessment, quote) pass between steps
through a Python-side context dict; the orchestrator LLM only sequences the
calls and writes the final reply. Inter-agent messages are JSON text.
