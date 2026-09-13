import os
from datetime import date
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL not found.\n"
        "Please create a .env file containing:\n"
        "DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/business_twin"
    )


# Opening cash is not present in our seed transaction data.
# We therefore define an initial cash position for the twin.
OPENING_CASH = 5_000_000


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(DATABASE_URL)


# ============================================================
# HELPER
# ============================================================

def decimal_to_float(value):
    """
    PostgreSQL NUMERIC values are returned as Decimal.
    Convert them to normal Python floats for JSON/UI use.
    """
    if isinstance(value, Decimal):
        return float(value)

    if value is None:
        return 0.0

    return float(value)


# ============================================================
# REVENUE
# ============================================================

def get_revenue(cursor):
    cursor.execute(
        """
        SELECT
            COALESCE(SUM(quantity * unit_price), 0) AS revenue
        FROM sales;
        """
    )

    result = cursor.fetchone()

    return decimal_to_float(result["revenue"])


# ============================================================
# COGS
# ============================================================

def get_cogs(cursor):
    """
    Cost of Goods Sold.

    COGS = quantity sold × product cost price
    """

    cursor.execute(
        """
        SELECT
            COALESCE(
                SUM(s.quantity * p.cost_price),
                0
            ) AS cogs
        FROM sales s
        JOIN products p
            ON s.product_id = p.product_id;
        """
    )

    result = cursor.fetchone()

    return decimal_to_float(result["cogs"])


# ============================================================
# EXPENSES
# ============================================================

def get_operating_expenses(cursor):
    cursor.execute(
        """
        SELECT
            COALESCE(SUM(amount), 0) AS expenses
        FROM expenses;
        """
    )

    result = cursor.fetchone()

    return decimal_to_float(result["expenses"])


# ============================================================
# ACCOUNTS RECEIVABLE
# ============================================================

def get_accounts_receivable(cursor):
    """
    AR = sales that have not yet been paid.
    """

    cursor.execute(
        """
        SELECT
            COALESCE(
                SUM(s.quantity * s.unit_price),
                0
            ) AS accounts_receivable
        FROM sales s
        WHERE s.payment_status = 'Pending';
        """
    )

    result = cursor.fetchone()

    return decimal_to_float(result["accounts_receivable"])


# ============================================================
# PAYMENTS RECEIVED
# ============================================================

def get_total_payments(cursor):
    cursor.execute(
        """
        SELECT
            COALESCE(SUM(amount), 0) AS total_payments
        FROM payments;
        """
    )

    result = cursor.fetchone()

    return decimal_to_float(result["total_payments"])


# ============================================================
# INVENTORY
# ============================================================

def get_inventory(cursor):
    """
    Calculates the current inventory value.

    Inventory Value =
        stock quantity × product cost price
    """

    cursor.execute(
        """
        SELECT
            COALESCE(
                SUM(i.stock_quantity * p.cost_price),
                0
            ) AS inventory_value,

            COALESCE(
                SUM(i.stock_quantity),
                0
            ) AS inventory_units

        FROM inventory i
        JOIN products p
            ON i.product_id = p.product_id;
        """
    )

    result = cursor.fetchone()

    return {
        "value": decimal_to_float(result["inventory_value"]),
        "units": int(result["inventory_units"])
    }


# ============================================================
# SALES STATISTICS
# ============================================================

def get_sales_statistics(cursor):
    cursor.execute(
        """
        SELECT
            COUNT(*) AS number_of_sales,
            COALESCE(SUM(quantity), 0) AS units_sold,
            COALESCE(AVG(unit_price), 0) AS average_sale_price
        FROM sales;
        """
    )

    result = cursor.fetchone()

    return {
        "transactions": int(result["number_of_sales"]),
        "units_sold": int(result["units_sold"]),
        "average_sale_price": decimal_to_float(
            result["average_sale_price"]
        )
    }


# ============================================================
# PAYMENT BEHAVIOR
# ============================================================

def get_payment_metrics(cursor):
    """
    Calculate average payment delay.

    PostgreSQL DATE - DATE returns an integer number of days,
    so we can directly average the difference.
    """

    cursor.execute(
        """
        SELECT
            COUNT(*) AS payments_count,

            COALESCE(
                AVG(
                    p.payment_date - s.sale_date
                ),
                0
            ) AS average_payment_days

        FROM payments p

        JOIN sales s
            ON p.sale_id = s.sale_id;
        """
    )

    result = cursor.fetchone()

    return {
        "payments_count": int(result["payments_count"]),
        "average_payment_days": round(
            decimal_to_float(
                result["average_payment_days"]
            ),
            2
        )
    }

# ============================================================
# CUSTOMER PAYMENT TERMS
# ============================================================

def get_average_payment_terms(cursor):
    cursor.execute(
        """
        SELECT
            COALESCE(
                AVG(payment_terms_days),
                0
            ) AS average_payment_terms
        FROM customers;
        """
    )

    result = cursor.fetchone()

    return round(
        decimal_to_float(result["average_payment_terms"]),
        2
    )


# ============================================================
# DSO
# ============================================================

def calculate_dso(
    accounts_receivable,
    revenue,
    period_days=365
):
    """
    Days Sales Outstanding.

    Approximation:

        DSO = AR / Revenue × number of days

    This uses the complete historical dataset.
    """

    if revenue <= 0:
        return 0.0

    return round(
        (accounts_receivable / revenue) * period_days,
        2
    )


# ============================================================
# GROSS MARGIN
# ============================================================

def calculate_gross_margin(
    revenue,
    cogs
):
    if revenue <= 0:
        return 0.0

    gross_profit = revenue - cogs

    return round(
        (gross_profit / revenue) * 100,
        2
    )


# ============================================================
# NET MARGIN
# ============================================================

def calculate_net_margin(
    revenue,
    cogs,
    expenses
):
    if revenue <= 0:
        return 0.0

    net_profit = revenue - cogs - expenses

    return round(
        (net_profit / revenue) * 100,
        2
    )


# ============================================================
# INVENTORY TURNOVER
# ============================================================

def calculate_inventory_turnover(
    cogs,
    inventory_value
):
    """
    Simplified inventory turnover.

    Annual historical COGS / current inventory value.
    """

    if inventory_value <= 0:
        return 0.0

    return round(
        cogs / inventory_value,
        2
    )


# ============================================================
# DAYS INVENTORY OUTSTANDING
# ============================================================

def calculate_inventory_days(
    inventory_turnover
):
    if inventory_turnover <= 0:
        return 0.0

    return round(
        365 / inventory_turnover,
        2
    )


# ============================================================
# CASH
# ============================================================

def calculate_cash(
    opening_cash,
    payments,
    expenses
):
    """
    Simplified cash model.

    Cash =
        Opening Cash
        + Payments Received
        - Operating Expenses
    """

    return round(
        opening_cash
        + payments
        - expenses,
        2
    )


# ============================================================
# CASH FLOW
# ============================================================

def calculate_operating_cash_flow(
    payments,
    expenses
):
    return round(
        payments - expenses,
        2
    )


# ============================================================
# STOCKOUT RISK
# ============================================================

def calculate_inventory_risk(cursor):
    """
    Detect products whose stock is at or below reorder level.
    """

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_products,

            COUNT(*) FILTER (
                WHERE stock_quantity <= reorder_level
            ) AS products_at_risk

        FROM inventory;
        """
    )

    result = cursor.fetchone()

    total_products = int(result["total_products"])
    products_at_risk = int(result["products_at_risk"])

    if total_products == 0:
        return {
            "products_at_risk": 0,
            "risk_percentage": 0.0
        }

    risk_percentage = (
        products_at_risk / total_products
    ) * 100

    return {
        "products_at_risk": products_at_risk,
        "risk_percentage": round(
            risk_percentage,
            2
        )
    }


# ============================================================
# PENDING RECEIVABLE RISK
# ============================================================

def calculate_receivable_risk(
    accounts_receivable,
    revenue
):
    if revenue <= 0:
        return 0.0

    percentage = (
        accounts_receivable / revenue
    ) * 100

    return round(
        percentage,
        2
    )


# ============================================================
# OVERALL RISK SCORE
# ============================================================

def calculate_risk_score(
    dso,
    inventory_risk_percentage,
    receivable_percentage,
    net_margin
):
    """
    Deterministic risk score from 0–100.

    Higher = greater business risk.
    """

    score = 0

    # --------------------------------------------------------
    # DSO risk
    # --------------------------------------------------------

    if dso > 60:
        score += 30

    elif dso > 45:
        score += 20

    elif dso > 30:
        score += 10

    # --------------------------------------------------------
    # Inventory risk
    # --------------------------------------------------------

    if inventory_risk_percentage > 30:
        score += 25

    elif inventory_risk_percentage > 15:
        score += 15

    elif inventory_risk_percentage > 5:
        score += 5

    # --------------------------------------------------------
    # Receivables risk
    # --------------------------------------------------------

    if receivable_percentage > 30:
        score += 25

    elif receivable_percentage > 20:
        score += 15

    elif receivable_percentage > 10:
        score += 5

    # --------------------------------------------------------
    # Profitability risk
    # --------------------------------------------------------

    if net_margin < 0:
        score += 20

    elif net_margin < 5:
        score += 15

    elif net_margin < 10:
        score += 5

    return min(
        score,
        100
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 70:
        return "Critical"

    if score >= 50:
        return "High"

    if score >= 30:
        return "Medium"

    return "Low"


# ============================================================
# BUILD DIGITAL TWIN
# ============================================================

def build_digital_twin():

    conn = get_connection()

    try:

        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            # ------------------------------------------------
            # Core financial metrics
            # ------------------------------------------------

            revenue = get_revenue(cursor)

            cogs = get_cogs(cursor)

            expenses = get_operating_expenses(cursor)

            gross_profit = revenue - cogs

            net_profit = (
                revenue
                - cogs
                - expenses
            )

            # ------------------------------------------------
            # Working capital
            # ------------------------------------------------

            accounts_receivable = (
                get_accounts_receivable(cursor)
            )

            inventory = get_inventory(cursor)

            payments = get_total_payments(cursor)

            # ------------------------------------------------
            # Cash
            # ------------------------------------------------

            cash = calculate_cash(
                OPENING_CASH,
                payments,
                expenses
            )

            operating_cash_flow = (
                calculate_operating_cash_flow(
                    payments,
                    expenses
                )
            )

            # ------------------------------------------------
            # Ratios
            # ------------------------------------------------

            dso = calculate_dso(
                accounts_receivable,
                revenue
            )

            gross_margin = calculate_gross_margin(
                revenue,
                cogs
            )

            net_margin = calculate_net_margin(
                revenue,
                cogs,
                expenses
            )

            inventory_turnover = (
                calculate_inventory_turnover(
                    cogs,
                    inventory["value"]
                )
            )

            inventory_days = (
                calculate_inventory_days(
                    inventory_turnover
                )
            )

            # ------------------------------------------------
            # Payment behavior
            # ------------------------------------------------

            payment_metrics = (
                get_payment_metrics(cursor)
            )

            average_payment_terms = (
                get_average_payment_terms(cursor)
            )

            # ------------------------------------------------
            # Inventory risk
            # ------------------------------------------------

            inventory_risk = (
                calculate_inventory_risk(cursor)
            )

            # ------------------------------------------------
            # Receivable risk
            # ------------------------------------------------

            receivable_percentage = (
                calculate_receivable_risk(
                    accounts_receivable,
                    revenue
                )
            )

            # ------------------------------------------------
            # Overall risk
            # ------------------------------------------------

            risk_score = calculate_risk_score(
                dso=dso,
                inventory_risk_percentage=inventory_risk[
                    "risk_percentage"
                ],
                receivable_percentage=receivable_percentage,
                net_margin=net_margin
            )

            risk_level = get_risk_level(
                risk_score
            )

            # ------------------------------------------------
            # Sales statistics
            # ------------------------------------------------

            sales_statistics = (
                get_sales_statistics(cursor)
            )

            # ------------------------------------------------
            # Digital Twin
            # ------------------------------------------------

            twin = {

                "metadata": {
                    "company": "Acme Retail Pvt. Ltd.",
                    "generated_on": str(date.today()),
                    "currency": "INR"
                },

                "financials": {

                    "revenue": round(
                        revenue,
                        2
                    ),

                    "cogs": round(
                        cogs,
                        2
                    ),

                    "gross_profit": round(
                        gross_profit,
                        2
                    ),

                    "gross_margin_percent": gross_margin,

                    "operating_expenses": round(
                        expenses,
                        2
                    ),

                    "net_profit": round(
                        net_profit,
                        2
                    ),

                    "net_margin_percent": net_margin
                },

                "cash_flow": {

                    "opening_cash": OPENING_CASH,

                    "payments_received": round(
                        payments,
                        2
                    ),

                    "operating_cash_flow": (
                        operating_cash_flow
                    ),

                    "current_cash": cash
                },

                "working_capital": {

                    "accounts_receivable": round(
                        accounts_receivable,
                        2
                    ),

                    "inventory_value": round(
                        inventory["value"],
                        2
                    ),

                    "inventory_units": inventory[
                        "units"
                    ],

                    "receivable_percentage": (
                        receivable_percentage
                    )
                },

                "operations": {

                    "sales_transactions": (
                        sales_statistics[
                            "transactions"
                        ]
                    ),

                    "units_sold": (
                        sales_statistics[
                            "units_sold"
                        ]
                    ),

                    "average_sale_price": (
                        sales_statistics[
                            "average_sale_price"
                        ]
                    ),

                    "inventory_turnover": (
                        inventory_turnover
                    ),

                    "inventory_days": (
                        inventory_days
                    )
                },

                "payment_behavior": {

                    "average_payment_days": (
                        payment_metrics[
                            "average_payment_days"
                        ]
                    ),

                    "average_payment_terms": (
                        average_payment_terms
                    ),

                    "total_payments": (
                        payment_metrics[
                            "payments_count"
                        ]
                    ),
                    # Days Sales Outstanding: Receivables Collection Velocity = (AR / Revenue) * 365
                    "dso_days": dso
                },

                "risk": {

                    "overall_score": risk_score,

                    "level": risk_level,

                    "inventory": inventory_risk,

                    "receivables_percentage": (
                        receivable_percentage
                    )
                }
            }

            return twin

    finally:

        conn.close()


# ============================================================
# PRINT DIGITAL TWIN
# ============================================================

def print_digital_twin(twin):

    print("\n")
    print("=" * 70)
    print("              BUSINESS DIGITAL TWIN")
    print("=" * 70)

    print(
        f"\nCompany: "
        f"{twin['metadata']['company']}"
    )

    print("\nFINANCIALS")
    print("-" * 70)

    print(
        f"Revenue              : "
        f"₹{twin['financials']['revenue']:,.2f}"
    )

    print(
        f"COGS                 : "
        f"₹{twin['financials']['cogs']:,.2f}"
    )

    print(
        f"Gross Profit         : "
        f"₹{twin['financials']['gross_profit']:,.2f}"
    )

    print(
        f"Gross Margin         : "
        f"{twin['financials']['gross_margin_percent']}%"
    )

    print(
        f"Operating Expenses   : "
        f"₹{twin['financials']['operating_expenses']:,.2f}"
    )

    print(
        f"Net Profit           : "
        f"₹{twin['financials']['net_profit']:,.2f}"
    )

    print(
        f"Net Margin           : "
        f"{twin['financials']['net_margin_percent']}%"
    )

    print("\nCASH FLOW")
    print("-" * 70)

    print(
        f"Opening Cash        : "
        f"₹{twin['cash_flow']['opening_cash']:,.2f}"
    )

    print(
        f"Payments Received   : "
        f"₹{twin['cash_flow']['payments_received']:,.2f}"
    )

    print(
        f"Operating Cash Flow : "
        f"₹{twin['cash_flow']['operating_cash_flow']:,.2f}"
    )

    print(
        f"Current Cash        : "
        f"₹{twin['cash_flow']['current_cash']:,.2f}"
    )

    print("\nWORKING CAPITAL")
    print("-" * 70)

    print(
        f"Accounts Receivable : "
        f"₹{twin['working_capital']['accounts_receivable']:,.2f}"
    )

    print(
        f"Inventory Value     : "
        f"₹{twin['working_capital']['inventory_value']:,.2f}"
    )

    print(
        f"Inventory Units     : "
        f"{twin['working_capital']['inventory_units']:,}"
    )

    print("\nOPERATIONS")
    print("-" * 70)

    print(
        f"Sales Transactions  : "
        f"{twin['operations']['sales_transactions']}"
    )

    print(
        f"Units Sold          : "
        f"{twin['operations']['units_sold']:,}"
    )

    print(
        f"Inventory Turnover  : "
        f"{twin['operations']['inventory_turnover']}x"
    )

    print(
        f"Inventory Days      : "
        f"{twin['operations']['inventory_days']} days"
    )

    print("\nPAYMENT BEHAVIOR")
    print("-" * 70)

    print(
        f"Average Payment     : "
        f"{twin['payment_behavior']['average_payment_days']} days"
    )

    print(
        f"Average Terms       : "
        f"{twin['payment_behavior']['average_payment_terms']} days"
    )

    print("\nRISK")
    print("-" * 70)

    print(
        f"Risk Score          : "
        f"{twin['risk']['overall_score']}/100"
    )

    print(
        f"Risk Level          : "
        f"{twin['risk']['level']}"
    )

    print(
        f"Inventory Risk      : "
        f"{twin['risk']['inventory']['risk_percentage']}%"
    )

    print(
        f"Receivable Exposure : "
        f"{twin['risk']['receivables_percentage']}%"
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        twin = build_digital_twin()

        print_digital_twin(twin)

    except Exception as e:

        print("\nFailed to build Digital Twin.")
        print("-" * 70)
        print(str(e))

        raise