# backend/database/seed_database.py

import os
from datetime import date, timedelta
from decimal import Decimal

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL not found. Create a .env file with:\n"
        "DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/business_twin"
    )


# ============================================================
# SEED DATA
# ============================================================

CUSTOMERS = [
    ("C001", "Alpha Electronics", "Retail", "Mumbai", 30, 500000),
    ("C002", "TechWorld Solutions", "Corporate", "Pune", 45, 800000),
    ("C003", "SmartBuy Retail", "Retail", "Nashik", 30, 350000),
    ("C004", "Digital Hub", "Corporate", "Nagpur", 60, 1000000),
    ("C005", "FutureTech", "Retail", "Aurangabad", 30, 400000),
    ("C006", "Metro Computers", "Corporate", "Mumbai", 45, 750000),
    ("C007", "Vision Electronics", "Retail", "Pune", 30, 300000),
    ("C008", "Galaxy Stores", "Retail", "Kolhapur", 30, 250000),
    ("C009", "NextGen Systems", "Corporate", "Nashik", 60, 900000),
    ("C010", "Prime Digital", "Corporate", "Pune", 45, 600000),
    ("C011", "QuickTech", "Retail", "Mumbai", 30, 300000),
    ("C012", "Elite Systems", "Corporate", "Nagpur", 60, 850000),
]


PRODUCTS = [
    ("P001", "Laptop Pro 14", "Laptop", 52000, 69999, 20, 14),
    ("P002", "Laptop Air 15", "Laptop", 42000, 57999, 25, 14),
    ("P003", "Gaming Laptop X", "Laptop", 68000, 89999, 15, 21),
    ("P004", "Office Laptop 13", "Laptop", 35000, 47999, 30, 14),
    ("P005", "Mechanical Keyboard", "Accessories", 3500, 5499, 50, 10),
    ("P006", "Wireless Mouse", "Accessories", 900, 1499, 100, 7),
    ("P007", "27 Inch Monitor", "Monitor", 14000, 19999, 30, 14),
    ("P008", "24 Inch Monitor", "Monitor", 9500, 13999, 35, 14),
    ("P009", "USB-C Dock", "Accessories", 4500, 6999, 40, 10),
    ("P010", "Wireless Headset", "Accessories", 2800, 4499, 60, 7),
    ("P011", "External SSD 1TB", "Storage", 6500, 8999, 40, 10),
    ("P012", "External SSD 2TB", "Storage", 10500, 13999, 30, 10),
]


SALES = [
    ("S0001", "2025-09-03", "C001", "P001", 3, 69999, "Paid"),
    ("S0002", "2025-09-05", "C003", "P006", 15, 1499, "Paid"),
    ("S0003", "2025-09-08", "C002", "P003", 2, 89999, "Paid"),
    ("S0004", "2025-09-12", "C007", "P008", 8, 13999, "Paid"),
    ("S0005", "2025-09-15", "C004", "P004", 10, 47999, "Pending"),
    ("S0006", "2025-09-19", "C005", "P005", 12, 5499, "Paid"),
    ("S0007", "2025-09-22", "C006", "P007", 5, 19999, "Pending"),
    ("S0008", "2025-09-27", "C008", "P010", 10, 4499, "Paid"),

    ("S0009", "2025-10-02", "C009", "P002", 8, 57999, "Paid"),
    ("S0010", "2025-10-06", "C010", "P001", 4, 69999, "Pending"),
    ("S0011", "2025-10-11", "C001", "P011", 10, 8999, "Paid"),
    ("S0012", "2025-10-16", "C003", "P006", 20, 1499, "Paid"),
    ("S0013", "2025-10-21", "C004", "P003", 3, 89999, "Pending"),
    ("S0014", "2025-10-25", "C007", "P008", 10, 13999, "Paid"),
    ("S0015", "2025-10-29", "C002", "P007", 6, 19999, "Paid"),

    ("S0016", "2025-11-03", "C005", "P002", 6, 57999, "Paid"),
    ("S0017", "2025-11-07", "C006", "P004", 12, 47999, "Pending"),
    ("S0018", "2025-11-13", "C008", "P006", 25, 1499, "Paid"),
    ("S0019", "2025-11-18", "C009", "P003", 4, 89999, "Pending"),
    ("S0020", "2025-11-24", "C010", "P001", 5, 69999, "Paid"),

    ("S0021", "2025-12-02", "C001", "P001", 6, 69999, "Paid"),
    ("S0022", "2025-12-05", "C003", "P010", 20, 4499, "Paid"),
    ("S0023", "2025-12-10", "C004", "P002", 12, 57999, "Pending"),
    ("S0024", "2025-12-15", "C007", "P007", 8, 19999, "Paid"),
    ("S0025", "2025-12-21", "C002", "P003", 5, 89999, "Paid"),
    ("S0026", "2025-12-27", "C006", "P011", 15, 8999, "Pending"),

    ("S0027", "2026-01-04", "C005", "P004", 10, 47999, "Paid"),
    ("S0028", "2026-01-08", "C008", "P006", 30, 1499, "Paid"),
    ("S0029", "2026-01-14", "C009", "P001", 7, 69999, "Pending"),
    ("S0030", "2026-01-19", "C010", "P008", 12, 13999, "Paid"),
    ("S0031", "2026-01-25", "C001", "P007", 7, 19999, "Paid"),

    ("S0032", "2026-02-02", "C002", "P002", 10, 57999, "Paid"),
    ("S0033", "2026-02-07", "C004", "P003", 5, 89999, "Pending"),
    ("S0034", "2026-02-13", "C007", "P005", 25, 5499, "Paid"),
    ("S0035", "2026-02-18", "C006", "P004", 15, 47999, "Pending"),
    ("S0036", "2026-02-24", "C003", "P006", 35, 1499, "Paid"),

    ("S0037", "2026-03-03", "C009", "P001", 8, 69999, "Paid"),
    ("S0038", "2026-03-09", "C010", "P011", 20, 8999, "Paid"),
    ("S0039", "2026-03-15", "C001", "P003", 3, 89999, "Pending"),
    ("S0040", "2026-03-22", "C005", "P008", 15, 13999, "Paid"),

    ("S0041", "2026-04-02", "C002", "P001", 9, 69999, "Paid"),
    ("S0042", "2026-04-08", "C004", "P004", 18, 47999, "Pending"),
    ("S0043", "2026-04-15", "C007", "P006", 40, 1499, "Paid"),
    ("S0044", "2026-04-21", "C006", "P007", 10, 19999, "Paid"),
    ("S0045", "2026-04-27", "C009", "P002", 12, 57999, "Pending"),

    ("S0046", "2026-05-04", "C010", "P003", 6, 89999, "Paid"),
    ("S0047", "2026-05-10", "C001", "P011", 25, 8999, "Paid"),
    ("S0048", "2026-05-16", "C003", "P008", 18, 13999, "Paid"),
    ("S0049", "2026-05-23", "C004", "P001", 10, 69999, "Pending"),
    ("S0050", "2026-05-29", "C002", "P004", 20, 47999, "Paid"),

    ("S0051", "2026-06-03", "C005", "P002", 15, 57999, "Paid"),
    ("S0052", "2026-06-09", "C006", "P003", 7, 89999, "Pending"),
    ("S0053", "2026-06-15", "C008", "P006", 45, 1499, "Paid"),
    ("S0054", "2026-06-21", "C009", "P007", 12, 19999, "Paid"),
    ("S0055", "2026-06-27", "C010", "P001", 11, 69999, "Pending"),

    ("S0056", "2026-07-04", "C001", "P003", 5, 89999, "Paid"),
    ("S0057", "2026-07-10", "C002", "P002", 18, 57999, "Paid"),
    ("S0058", "2026-07-16", "C004", "P004", 22, 47999, "Pending"),
    ("S0059", "2026-07-22", "C007", "P006", 50, 1499, "Paid"),
    ("S0060", "2026-07-28", "C006", "P007", 15, 19999, "Paid"),

    ("S0061", "2026-08-03", "C009", "P001", 12, 69999, "Pending"),
    ("S0062", "2026-08-09", "C010", "P011", 30, 8999, "Paid"),
    ("S0063", "2026-08-15", "C003", "P008", 22, 13999, "Paid"),
    ("S0064", "2026-08-21", "C005", "P002", 20, 57999, "Paid"),
    ("S0065", "2026-08-27", "C004", "P003", 8, 89999, "Pending"),
    ("S0066", "2026-08-30", "C002", "P004", 25, 47999, "Paid"),
]


INVENTORY = [
    ("I001", "P001", 32, "WH_Mumbai", 20, "2026-08-15"),
    ("I002", "P002", 45, "WH_Pune", 25, "2026-08-18"),
    ("I003", "P003", 18, "WH_Mumbai", 15, "2026-08-20"),
    ("I004", "P004", 55, "WH_Pune", 30, "2026-08-21"),
    ("I005", "P005", 82, "WH_Mumbai", 50, "2026-08-22"),
    ("I006", "P006", 180, "WH_Pune", 100, "2026-08-23"),
    ("I007", "P007", 38, "WH_Mumbai", 30, "2026-08-17"),
    ("I008", "P008", 52, "WH_Pune", 35, "2026-08-19"),
    ("I009", "P009", 48, "WH_Mumbai", 40, "2026-08-16"),
    ("I010", "P010", 75, "WH_Pune", 60, "2026-08-24"),
    ("I011", "P011", 65, "WH_Mumbai", 40, "2026-08-25"),
    ("I012", "P012", 36, "WH_Pune", 30, "2026-08-26"),
]


EXPENSES = [
    ("E001", "2026-04-01", "Rent", 350000, "Warehouse and office rent"),
    ("E002", "2026-04-05", "Salaries", 850000, "Employee salaries"),
    ("E003", "2026-04-10", "Marketing", 180000, "Digital marketing"),
    ("E004", "2026-04-15", "Utilities", 75000, "Electricity and internet"),

    ("E005", "2026-05-01", "Rent", 350000, "Warehouse and office rent"),
    ("E006", "2026-05-05", "Salaries", 850000, "Employee salaries"),
    ("E007", "2026-05-10", "Marketing", 220000, "Digital marketing"),
    ("E008", "2026-05-15", "Utilities", 78000, "Electricity and internet"),

    ("E009", "2026-06-01", "Rent", 350000, "Warehouse and office rent"),
    ("E010", "2026-06-05", "Salaries", 875000, "Employee salaries"),
    ("E011", "2026-06-10", "Marketing", 200000, "Digital marketing"),
    ("E012", "2026-06-15", "Utilities", 80000, "Electricity and internet"),

    ("E013", "2026-07-01", "Rent", 350000, "Warehouse and office rent"),
    ("E014", "2026-07-05", "Salaries", 875000, "Employee salaries"),
    ("E015", "2026-07-10", "Marketing", 240000, "Digital marketing"),
    ("E016", "2026-07-15", "Utilities", 82000, "Electricity and internet"),

    ("E017", "2026-08-01", "Rent", 350000, "Warehouse and office rent"),
    ("E018", "2026-08-05", "Salaries", 900000, "Employee salaries"),
    ("E019", "2026-08-10", "Marketing", 250000, "Digital marketing"),
    ("E020", "2026-08-15", "Utilities", 85000, "Electricity and internet"),
]


# ============================================================
# DATABASE SCHEMA
# ============================================================

CREATE_TABLES = """

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    customer_name VARCHAR(150) NOT NULL,
    customer_type VARCHAR(50) NOT NULL,
    city VARCHAR(100),
    payment_terms_days INTEGER,
    credit_limit NUMERIC(14,2)
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name VARCHAR(150) NOT NULL,
    category VARCHAR(100),
    cost_price NUMERIC(12,2),
    selling_price NUMERIC(12,2),
    reorder_level INTEGER,
    lead_time_days INTEGER
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id VARCHAR(20) PRIMARY KEY,
    sale_date DATE NOT NULL,
    customer_id VARCHAR(20) REFERENCES customers(customer_id),
    product_id VARCHAR(20) REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL,
    payment_status VARCHAR(30) NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id VARCHAR(20) PRIMARY KEY,
    product_id VARCHAR(20) REFERENCES products(product_id),
    stock_quantity INTEGER NOT NULL,
    warehouse VARCHAR(100),
    reorder_level INTEGER,
    last_restock_date DATE
);

CREATE TABLE IF NOT EXISTS expenses (
    expense_id VARCHAR(20) PRIMARY KEY,
    expense_date DATE NOT NULL,
    category VARCHAR(100),
    amount NUMERIC(14,2),
    description TEXT
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id VARCHAR(20) PRIMARY KEY,
    sale_id VARCHAR(20) REFERENCES sales(sale_id),
    payment_date DATE NOT NULL,
    amount NUMERIC(14,2),
    payment_method VARCHAR(50)
);

"""


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(DATABASE_URL)


# ============================================================
# SEED DATABASE
# ============================================================

def seed_database():

    print("\n" + "=" * 60)
    print("BUSINESS DIGITAL TWIN - DATABASE SEED")
    print("=" * 60)

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            print("\n[1/7] Creating tables...")
            cursor.execute(CREATE_TABLES)

            # ------------------------------------------------
            # Clear existing seed data
            # ------------------------------------------------

            print("[2/7] Clearing existing data...")

            cursor.execute("""
                TRUNCATE TABLE
                    payments,
                    sales,
                    inventory,
                    expenses,
                    products,
                    customers
                CASCADE;
            """)

            # ------------------------------------------------
            # Customers
            # ------------------------------------------------

            print("[3/7] Inserting customers...")

            execute_values(
                cursor,
                """
                INSERT INTO customers
                (
                    customer_id,
                    customer_name,
                    customer_type,
                    city,
                    payment_terms_days,
                    credit_limit
                )
                VALUES %s
                """,
                CUSTOMERS
            )

            # ------------------------------------------------
            # Products
            # ------------------------------------------------

            print("[4/7] Inserting products...")

            execute_values(
                cursor,
                """
                INSERT INTO products
                (
                    product_id,
                    product_name,
                    category,
                    cost_price,
                    selling_price,
                    reorder_level,
                    lead_time_days
                )
                VALUES %s
                """,
                PRODUCTS
            )

            # ------------------------------------------------
            # Sales
            # ------------------------------------------------

            print("[5/7] Inserting sales...")

            execute_values(
                cursor,
                """
                INSERT INTO sales
                (
                    sale_id,
                    sale_date,
                    customer_id,
                    product_id,
                    quantity,
                    unit_price,
                    payment_status
                )
                VALUES %s
                """,
                SALES
            )

            # ------------------------------------------------
            # Inventory
            # ------------------------------------------------

            print("[6/7] Inserting inventory and expenses...")

            execute_values(
                cursor,
                """
                INSERT INTO inventory
                (
                    inventory_id,
                    product_id,
                    stock_quantity,
                    warehouse,
                    reorder_level,
                    last_restock_date
                )
                VALUES %s
                """,
                INVENTORY
            )

            execute_values(
                cursor,
                """
                INSERT INTO expenses
                (
                    expense_id,
                    expense_date,
                    category,
                    amount,
                    description
                )
                VALUES %s
                """,
                EXPENSES
            )

            # ------------------------------------------------
            # Payments
            # ------------------------------------------------

            print("[7/7] Generating payment records...")

            payments = []

            payment_methods = [
                "Bank Transfer",
                "UPI",
                "Card"
            ]

            payment_counter = 1

            for sale in SALES:

                sale_id = sale[0]
                sale_date = date.fromisoformat(sale[1])
                customer_id = sale[2]
                quantity = sale[4]
                unit_price = sale[5]
                status = sale[6]

                # Only some sales have been paid.
                # Pending invoices remain as accounts receivable.
                if status == "Paid":

                    amount = quantity * unit_price

                    # Different customers pay at different speeds.
                    customer = next(
                        c for c in CUSTOMERS
                        if c[0] == customer_id
                    )

                    payment_terms = customer[4]

                    # Synthetic actual payment delay.
                    delay = max(
                        5,
                        min(
                            payment_terms,
                            payment_terms - 5 + (payment_counter % 11)
                        )
                    )

                    payment_date = sale_date + timedelta(days=delay)

                    payment_method = payment_methods[
                        payment_counter % len(payment_methods)
                    ]

                    payments.append(
                        (
                            f"PM{payment_counter:04d}",
                            sale_id,
                            payment_date,
                            amount,
                            payment_method
                        )
                    )

                    payment_counter += 1

            execute_values(
                cursor,
                """
                INSERT INTO payments
                (
                    payment_id,
                    sale_id,
                    payment_date,
                    amount,
                    payment_method
                )
                VALUES %s
                """,
                payments
            )

        conn.commit()

        print("\n" + "=" * 60)
        print("DATABASE SEEDED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nCustomers : {len(CUSTOMERS)}")
        print(f"Products  : {len(PRODUCTS)}")
        print(f"Sales     : {len(SALES)}")
        print(f"Inventory : {len(INVENTORY)}")
        print(f"Expenses  : {len(EXPENSES)}")
        print(f"Payments  : {len(payments)}")

        print("\nDatabase is ready for the Digital Twin.")

    except Exception as e:

        conn.rollback()

        print("\nDATABASE SEED FAILED")
        print("-" * 60)
        print(str(e))

        raise

    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    seed_database()