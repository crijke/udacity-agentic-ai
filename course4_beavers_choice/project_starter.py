import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            metadata = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df = quotes_df.assign(
                request_metadata=metadata,
                job_type=metadata.apply(lambda x: x.get("job_type", "")),
                order_size=metadata.apply(lambda x: x.get("order_size", "")),
                event_type=metadata.apply(lambda x: x.get("event_type", "")),
            )

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
load_dotenv()

model = OpenAIChatModel(
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

MODEL_SETTINGS = {"temperature": 0.0}

# ---------------------------------------------------------------------------
# Business policies
# ---------------------------------------------------------------------------
# Base bulk discount rule on the order's total unit count:
# 500+ units -> 5%, 1000+ -> 10%. The quoting agent may deviate from this
# base rule based on historical quote precedent, but its chosen discount is
# clamped to MAX_DISCOUNT_PERCENT and the arithmetic (including rounding the
# total down to a whole dollar) always runs in apply_discount, never in the LLM.
BULK_DISCOUNT_TIERS = [(1000, 0.10), (500, 0.05)]
MAX_DISCOUNT_PERCENT = 15.0
# Restock-to-fulfill orders cover the shortfall plus one min_stock_level as buffer.
# Proactive restocks top an item back up to 2x its min_stock_level.
# Stock orders are dated at the SUPPLIER DELIVERY DATE, so stock and the cash
# outflow materialize when goods arrive, not when they are ordered.


def _catalog() -> pd.DataFrame:
    """The sellable catalog: items present in the inventory table."""
    return pd.read_sql(
        "SELECT item_name, category, unit_price, min_stock_level FROM inventory",
        db_engine,
    )


def _catalog_text() -> str:
    cat = _catalog()
    return "\n".join(
        f"- {r.item_name} ({r.category}) — ${r.unit_price:.2f}/unit"
        for r in cat.itertuples()
    )


def _stock_at(item_name: str, as_of_date: str) -> int:
    return int(get_stock_level(item_name, as_of_date)["current_stock"].iloc[0])


class LineItem(BaseModel):
    item_name: str = Field(description="EXACT catalog item name")
    quantity: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Tools for inventory agent
# ---------------------------------------------------------------------------

def check_item_fulfillment(
    item_name: str | None,
    quantity: int | None,
    request_date: str,
    needed_by: str | None = None,
) -> dict:
    """Check whether a catalog item can be fulfilled in the requested quantity.

    Reports whether an item is available or if not if the item can be restocked in time and within budget.
    """
    if not item_name:
        return {
            "error": "item_name is required. Do not call this tool for unmatched "
            "items — just report them as unmatched in your output."
        }
    if not quantity or quantity <= 0:
        return {"error": "quantity must be a positive integer."}
    cat = _catalog().set_index("item_name")
    if item_name not in cat.index:
        return {
            "error": f"'{item_name}' is not a catalog item. Use an exact name from the catalog."
        }

    stock = _stock_at(item_name, request_date)
    result = {
        "item_name": item_name,
        "requested_quantity": quantity,
        "current_stock": stock,
        "in_stock": stock >= quantity,
    }
    if stock >= quantity:
        result.update({"requires_restock": False, "fulfillable": True})
        return result

    min_level = int(cat.loc[item_name, "min_stock_level"])
    unit_price = float(cat.loc[item_name, "unit_price"])
    restock_qty = (quantity - stock) + min_level
    delivery_date = get_supplier_delivery_date(request_date, restock_qty)
    restock_cost = round(restock_qty * unit_price, 2)
    cash = get_cash_balance(request_date)
    meets_deadline = needed_by is None or delivery_date <= needed_by
    affordable = restock_cost <= cash
    result.update(
        {
            "requires_restock": True,
            "restock_quantity": restock_qty,
            "restock_cost": restock_cost,
            "restock_delivery_date": delivery_date,
            "meets_deadline": meets_deadline,
            "cash_sufficient": affordable,
            "fulfillable": meets_deadline and affordable,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Tools for quoting agent
# ---------------------------------------------------------------------------

def calculate_quote(line_items: list[LineItem], request_date: str) -> dict:
    """Compute per-line prices and the undiscounted subtotal for a set of line
    items. Also reports the base-rule bulk discount percent as a starting
    point; the final discount is chosen by the quoting agent and applied with
    apply_discount."""
    cat = _catalog().set_index("item_name")
    line_item_quotes = []
    subtotal = 0.0
    total_units = 0
    for li in line_items:
        if li.item_name not in cat.index:
            return {"error": f"'{li.item_name}' is not a catalog item."}
        unit_price = float(cat.loc[li.item_name, "unit_price"])
        line_total = round(li.quantity * unit_price, 2)
        line_item_quotes.append(
            {
                "item_name": li.item_name,
                "quantity": li.quantity,
                "unit_price": unit_price,
                "line_total": line_total,
            }
        )
        subtotal += line_total
        total_units += li.quantity

    base_rate = 0.0
    for threshold, rate in BULK_DISCOUNT_TIERS:
        if total_units >= threshold:
            base_rate = rate
            break
    return {
        "request_date": request_date,
        "line_items": line_item_quotes,
        "total_units": total_units,
        "subtotal": round(subtotal, 2),
        "base_discount_percent": base_rate * 100,
    }


def apply_discount(subtotal: float, discount_percent: float) -> dict:
    """Apply a bulk discount to a subtotal and round the total DOWN to a whole
    dollar (never rounds up, so the customer is never overcharged). The
    discount percent is clamped to 0..MAX_DISCOUNT_PERCENT. The returned
    total_amount is final and authoritative."""
    pct = max(0.0, min(float(discount_percent), MAX_DISCOUNT_PERCENT))
    subtotal = round(subtotal, 2)
    total = float(int(subtotal * (1 - pct / 100)))
    return {
        "subtotal": subtotal,
        "discount_percent": pct,
        "discount_amount": round(subtotal - total, 2),
        "total_amount": total,
    }


def find_similar_quotes(search_terms: list[str]) -> list:
    """Search historical quotes by keyword (e.g. event type, job type, item names).    """
    return search_quote_history(search_terms)


# ---------------------------------------------------------------------------
# Tools for ordering agent
# ---------------------------------------------------------------------------

def finalize_sale(
    line_items: list[LineItem],
    total_amount: float,
    request_date: str,
    needed_by: str | None = None,
) -> dict:
    """Finalize a quoted sale: place all restock order and record the sale in the transactions table. """
    cat = _catalog().set_index("item_name")
    breakdown = []
    for li in line_items:
        if li.item_name not in cat.index:
            return {"status": "error", "reason": f"'{li.item_name}' is not a catalog item."}
        unit_price = float(cat.loc[li.item_name, "unit_price"])
        breakdown.append((li.item_name, li.quantity, unit_price, li.quantity * unit_price))
    subtotal = sum(b[3] for b in breakdown)
    if subtotal <= 0:
        return {"status": "error", "reason": "Empty or zero-value order."}

    check_date = max(request_date, needed_by) if needed_by else request_date
    restock_plans = []
    for name, qty, unit_price, _ in breakdown:
        available = _stock_at(name, check_date)
        if available < qty:
            min_level = int(cat.loc[name, "min_stock_level"])
            restock_qty = (qty - available) + min_level
            delivery_date = get_supplier_delivery_date(request_date, restock_qty)
            if needed_by and delivery_date > needed_by:
                return {
                    "status": "refused",
                    "reason": (
                        f"'{name}' is short by {qty - available} units and the supplier "
                        f"cannot deliver before {needed_by} (earliest {delivery_date})."
                    ),
                }
            restock_plans.append(
                {
                    "item_name": name,
                    "quantity": restock_qty,
                    "cost": round(restock_qty * unit_price, 2),
                    "delivery_date": delivery_date,
                }
            )

    restock_cost = sum(p["cost"] for p in restock_plans)
    if restock_cost > get_cash_balance(request_date) + total_amount:
        return {
            "status": "refused",
            "reason": f"Insufficient cash for required restock (${restock_cost:.2f}).",
        }

    for p in restock_plans:
        p["transaction_id"] = create_transaction(
            p["item_name"], "stock_orders", p["quantity"], p["cost"], request_date
        )

    sale_ids = []
    allocated = 0.0
    for i, (name, qty, unit_price, line_subtotal) in enumerate(breakdown):
        if i < len(breakdown) - 1:
            share = round(total_amount * line_subtotal / subtotal, 2)
            allocated = round(allocated + share, 2)
        else:
            share = round(total_amount - allocated, 2)
        sale_ids.append(create_transaction(name, "sales", qty, share, request_date))

    report = generate_financial_report(request_date)
    return {
        "status": "success",
        "sale_transaction_ids": sale_ids,
        "restock_orders_placed": restock_plans,
        "cash_after_sale": round(report["cash_balance"], 2),
        "inventory_value_after_sale": round(report["inventory_value"], 2),
    }


def restock_low_inventory(request_date: str) -> dict:
    """Proactively reorder every catalog item projected to sit below its
    min_stock_level (after inbound orders land), topping it up to 2x min_stock_level.
    Skips items cash cannot cover. Runs after every request, independent of
    whether a sale was made. Safe to call repeatedly: inbound stock orders are
    dated at delivery, so the projection already counts them."""
    horizon = (
        datetime.fromisoformat(request_date) + timedelta(days=14)
    ).strftime("%Y-%m-%d")
    cash = get_cash_balance(request_date)
    projected_stock = get_all_inventory(horizon)
    orders, skipped = [], []
    for r in _catalog().itertuples():
        projected = int(projected_stock.get(r.item_name, 0))
        if projected >= r.min_stock_level:
            continue
        qty = int(2 * r.min_stock_level - projected)
        cost = round(qty * r.unit_price, 2)
        if cost > cash:
            skipped.append({"item_name": r.item_name, "cost": cost})
            continue
        delivery_date = get_supplier_delivery_date(request_date, qty)
        txn_id = create_transaction(r.item_name, "stock_orders", qty, cost, delivery_date)
        cash = round(cash - cost, 2)
        orders.append(
            {
                "item_name": r.item_name,
                "quantity": qty,
                "cost": cost,
                "delivery_date": delivery_date,
                "transaction_id": txn_id,
            }
        )
    return {"restock_orders": orders, "skipped_for_cash": skipped, "cash_remaining": cash}


# ---------------------------------------------------------------------------
# Agents: inventory, quoting, ordering + an orchestrator that manages them
# ---------------------------------------------------------------------------

class ParsedItem(BaseModel):
    requested: str = Field(description="The item as the customer described it")
    matched_item_name: str | None = Field(
        description="EXACT catalog name, or null if nothing in the catalog reasonably matches"
    )
    quantity: int
    note: str = Field(default="", description="Short reason, e.g. why unmatched")


class ParsedRequest(BaseModel):
    items: list[ParsedItem]
    needed_by: str | None = Field(
        description="ISO date the customer needs delivery by, if stated"
    )


inventory_agent = Agent(
    model,
    output_type=ParsedRequest,
    model_settings=MODEL_SETTINGS,
    retries=3,
)


@inventory_agent.instructions
def inventory_instructions() -> str:
    return f"""You are the inventory agent of Munder Difflin, a paper company.

Given a customer request, extract EVERY requested item with its quantity, then
map each one to the closest item in our sellable catalog below. Matching rules:
- Use EXACT catalog names in matched_item_name.
- Reasonable substitutions are allowed within the same kind of product
  (e.g. "printer paper" or "white copy paper" -> "A4 paper"; "poster boards"
  -> "Large poster paper (24x36 inches)").
- Ignore size, color, finish and packaging qualifiers when matching
  (e.g. '8.5"x11" colored paper' -> "Colored paper"; "cardstock in various
  colors" -> "Cardstock"; "rolls of washi tape" -> "Decorative adhesive tape
  (washi tape)").
- Prefer the closest catalog match over null. Only set matched_item_name to
  null when the item is a different kind of product entirely (e.g. balloons)
  or truly absent from the catalog.
- Convert reams to sheets: 1 ream = 500 sheets.
- If nothing in the catalog is a reasonable match, set matched_item_name to
  null and explain in the note.
Also extract the customer's needed-by/delivery date if stated (ISO format).

SELLABLE CATALOG:
{_catalog_text()}"""


class QuoteResult(BaseModel):
    line_items: list[LineItem]
    discount_percent: float = Field(
        ge=0,
        description="Final bulk discount percent you chose (0 if none)",
    )
    discount_reason: str = Field(
        default="",
        description="Short reason: base rule, or which historical precedent justified deviating",
    )
    total_amount: float = Field(description="Exactly the total from apply_discount")
    quote_explanation: str = Field(
        description="Customer-facing explanation: itemized prices, any bulk discount, total"
    )


quoting_agent = Agent(
    model,
    output_type=QuoteResult,
    tools=[calculate_quote, find_similar_quotes, apply_discount],
    model_settings=MODEL_SETTINGS,
    retries=3,
    instructions=f"""You are the quoting agent of Munder Difflin, a paper company.

You receive a list of fulfillable line items (exact catalog names) for a
customer request. Steps:
1. Call calculate_quote with the line items — its line prices and subtotal are
   FINAL. Never compute prices yourself.
2. Call find_similar_quotes with keywords from the request (event type, job
   type, item names) to look for discount precedent in past quotes for similar
   customers or products.
3. Decide the bulk discount percent:
   - Base rule: 500+ total units -> 5%, 1000+ total units -> 10%, else 0%.
   - You MAY deviate from the base rule when historical quotes for a similar
     customer, event or product justify it (e.g. to match a precedent's
     discount for a comparable order). Never exceed {MAX_DISCOUNT_PERCENT:.0f}%.
   - Record your reasoning in discount_reason.
4. Call apply_discount with the subtotal and your chosen percent. It rounds
   the total down to a whole dollar; its total_amount is FINAL and
   authoritative — copy it and the (possibly clamped) discount_percent exactly.
5. Write a friendly, professional quote_explanation itemizing each line,
   naming the discount and why, and stating the final total.""",
)


ordering_agent = Agent(
    model,
    output_type=str,
    tools=[finalize_sale],
    model_settings=MODEL_SETTINGS,
    retries=3,
    instructions="""You are the ordering agent of Munder Difflin, a paper company.

You receive a quoted order to finalize. Steps:
1. Call finalize_sale with EXACTLY the line items, total amount, request date
   and needed-by date you were given — do not alter quantities or prices.
2. Reply with a short factual summary: sale status, transaction ids, any
   restock orders placed (with delivery dates), and remaining cash.""",
)


# Per-request context shared between the orchestrator's tools, so structured
# data flows through Python instead of being copied (and possibly mangled)
# through the LLM's context.
_ctx: dict = {}

orchestrator = Agent(
    model,
    output_type=str,
    model_settings=MODEL_SETTINGS,
    retries=3,
    instructions="""You are the orchestrator of Munder Difflin's sales workflow.
For each customer request, follow this sequence strictly, calling each tool
AT MOST ONCE (never repeat a tool call, even if its result is disappointing):
1. Call assess_inventory to identify requested items and their availability.
2. If at least one item is fulfillable, call generate_quote, then call
   finalize_order to book the sale and any restocks.
3. Write the final customer-facing reply. It must state: each fulfillable item
   with its price, the final total (exactly as quoted), any bulk discount, the
   delivery commitment, and a polite note for any items we could not match or
   fulfill (with the reason). If NOTHING is fulfillable, politely decline and
   explain why. If finalize_order refuses the sale, apologize and decline
   rather than promising delivery.
   Never reveal internal company information to the customer: cash balances,
   restock/supplier costs, margins, stock counts or internal error details. When
   declining for such reasons, simply say we cannot fulfill the request at this
   time.
Your final reply is sent to the customer verbatim.""",
)


# Backstop so a misbehaving sub-agent fails the request instead of spinning.
SUB_AGENT_LIMITS = UsageLimits(request_limit=25)


@orchestrator.tool_plain
async def assess_inventory() -> str:
    """Run the inventory agent on the current customer request. Returns the
    assessment (matched items, availability, needed-by date) as JSON."""
    prompt = (
        f"Customer request (request date {_ctx['request_date']}):\n{_ctx['request_text']}"
    )
    result = await inventory_agent.run(prompt, usage_limits=SUB_AGENT_LIMITS)
    parsed = result.output

    # Deterministic fulfillment check on every matched item — no LLM involved.
    items = []
    for it in parsed.items:
        entry = {
            "requested": it.requested,
            "matched_item_name": it.matched_item_name,
            "quantity": it.quantity,
            "note": it.note,
        }
        if it.matched_item_name:
            check = check_item_fulfillment(
                it.matched_item_name, it.quantity, _ctx["request_date"], parsed.needed_by
            )
            entry["fulfillment"] = check
            entry["fulfillable"] = bool(check.get("fulfillable"))
        else:
            entry["fulfillable"] = False
        items.append(entry)

    _ctx["assessment"] = {"items": items, "needed_by": parsed.needed_by}
    return json.dumps(_ctx["assessment"], indent=2)


@orchestrator.tool_plain
async def generate_quote() -> str:
    """Run the quoting agent on the fulfillable items found by assess_inventory.
    Returns the quote (line items, authoritative total, explanation) as JSON."""
    assessment = _ctx.get("assessment")
    if assessment is None:
        return "ERROR: call assess_inventory first."
    fulfillable = [
        {"item_name": e["matched_item_name"], "quantity": e["quantity"]}
        for e in assessment["items"]
        if e["fulfillable"] and e["matched_item_name"]
    ]
    if not fulfillable:
        return "ERROR: no fulfillable items to quote."
    prompt = (
        f"Request date: {_ctx['request_date']}\n"
        f"Original customer request:\n{_ctx['request_text']}\n\n"
        f"Fulfillable line items (exact catalog names):\n{json.dumps(fulfillable)}"
    )
    result = await quoting_agent.run(prompt, usage_limits=SUB_AGENT_LIMITS)
    quote = result.output
    # Guardrail: re-derive the authoritative total from the agent's chosen
    # discount percent, so the booked amount always matches deterministic
    # pricing + rounding policy even if the agent mistranscribed a number.
    priced = calculate_quote(quote.line_items, _ctx["request_date"])
    if "error" not in priced:
        check = apply_discount(priced["subtotal"], quote.discount_percent)
        quote.discount_percent = check["discount_percent"]
        quote.total_amount = check["total_amount"]
    _ctx["quote"] = quote
    return quote.model_dump_json(indent=2)


@orchestrator.tool_plain
async def finalize_order() -> str:
    """Run the ordering agent to book the quoted sale and needed restocks.
    Returns the ordering agent's factual summary."""
    quote = _ctx.get("quote")
    if quote is None:
        return "ERROR: call generate_quote first."
    assessment = _ctx["assessment"]
    prompt = (
        f"Request date: {_ctx['request_date']}\n"
        f"Needed by: {assessment['needed_by']}\n"
        f"Line items: {json.dumps([li.model_dump() for li in quote.line_items])}\n"
        f"Quoted total amount: {quote.total_amount}\n"
        f"Finalize this sale."
    )
    result = await ordering_agent.run(prompt, usage_limits=SUB_AGENT_LIMITS)
    return result.output


def process_quote_request(request_text: str, request_date: str) -> str:
    """Entry point: run one customer request through the multi-agent system."""
    _ctx.clear()
    _ctx["request_text"] = request_text
    _ctx["request_date"] = request_date
    try:
        result = orchestrator.run_sync(
            f"Customer request (request date {request_date}):\n{request_text}",
            usage_limits=UsageLimits(request_limit=100),
        )
        response = result.output
    except Exception as e:
        # One misbehaving request must not abort the whole scenario run.
        print(f"ERROR processing request dated {request_date}: {e}")
        response = (
            "We are sorry — we were unable to process your request due to an "
            "internal error. Please contact us to try again."
        )

    # Proactive restock runs on EVERY request date
    try:
        restock = restock_low_inventory(request_date)
        if restock["restock_orders"]:
            print(f"Proactive restock on {request_date}: {restock['restock_orders']}")
        if restock["skipped_for_cash"]:
            print(f"Restock skipped (insufficient cash): {restock['skipped_for_cash']}")
    except Exception as e:
        print(f"ERROR during proactive restock on {request_date}: {e}")

    return response


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)

   
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
       
        
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")

    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return



    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = process_quote_request(request_with_date, request_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            } 
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
