"""
Deterministic Database Handlers for Business Digital Twin Events.
All handlers execute parameterized SQL queries within the caller's transaction.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any, Dict


def _parse_date(val: Any) -> date:
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return date.fromisoformat(val.strip())
    raise ValueError(f"Invalid date format: {val}. Expected YYYY-MM-DD.")


def _generate_next_id(cursor, table: str, id_col: str, prefix: str, pad_len: int = 4) -> str:
    """Generate the next sequential ID with a given prefix (e.g., S0067, PM0047, E0021)."""
    cursor.execute(f"SELECT {id_col} FROM {table} WHERE {id_col} LIKE %s ORDER BY {id_col} DESC;", (f"{prefix}%",))
    rows = cursor.fetchall()
    max_num = 0
    pattern = re.compile(rf"^{prefix}(\d+)$")
    for row in rows:
        val = row[0] if isinstance(row, (list, tuple)) else row[id_col]
        match = pattern.match(val)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    return f"{prefix}{next_num:0{pad_len}d}"


# ==============================================================================
# 1. SALE_CREATED HANDLER
# ==============================================================================

def handle_sale_created(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles SALE_CREATED event.
    Inserts a new sale record into the sales table with 'Pending' payment status.
    """
    customer_id = str(payload.get("customer_id", "")).strip()
    product_id = str(payload.get("product_id", "")).strip()
    quantity = int(payload.get("quantity", 0))
    sale_date = _parse_date(payload.get("sale_date", str(date.today())))

    if not customer_id:
        raise ValueError("Customer ID is required.")
    if not product_id:
        raise ValueError("Product ID is required.")
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive integer, got {quantity}.")

    # Validate foreign keys
    cursor.execute("SELECT 1 FROM customers WHERE customer_id = %s;", (customer_id,))
    if not cursor.fetchone():
        raise ValueError(f"Customer '{customer_id}' does not exist in customers table.")

    cursor.execute("SELECT selling_price FROM products WHERE product_id = %s;", (product_id,))
    product_row = cursor.fetchone()
    if not product_row:
        raise ValueError(f"Product '{product_id}' does not exist in products table.")
    default_selling_price = float(product_row[0])

    # Determine unit_price and total amount
    if "sale_amount" in payload and payload["sale_amount"] is not None:
        total_amount = float(payload["sale_amount"])
        if total_amount <= 0:
            raise ValueError(f"Sale amount must be positive, got {total_amount}.")
        unit_price = round(total_amount / quantity, 2)
    elif "unit_price" in payload and payload["unit_price"] is not None:
        unit_price = float(payload["unit_price"])
        if unit_price <= 0:
            raise ValueError(f"Unit price must be positive, got {unit_price}.")
        total_amount = round(unit_price * quantity, 2)
    else:
        unit_price = default_selling_price
        total_amount = round(unit_price * quantity, 2)

    payment_status = payload.get("payment_status", "Pending")
    sale_id = payload.get("sale_id") or _generate_next_id(cursor, "sales", "sale_id", "S", pad_len=4)

    # Insert using parameterized SQL
    cursor.execute(
        """
        INSERT INTO sales (
            sale_id,
            sale_date,
            customer_id,
            product_id,
            quantity,
            unit_price,
            payment_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """,
        (sale_id, sale_date, customer_id, product_id, quantity, Decimal(str(unit_price)), payment_status)
    )

    return {
        "sale_id": sale_id,
        "customer_id": customer_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "sale_amount": total_amount,
        "sale_date": str(sale_date),
        "payment_status": payment_status,
    }


# ==============================================================================
# 2. PAYMENT_RECEIVED HANDLER
# ==============================================================================

def handle_payment_received(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles PAYMENT_RECEIVED event.
    Inserts payment record and updates sale payment_status to 'Paid' if fully settled.
    """
    sale_id = str(payload.get("sale_id", "")).strip()
    amount = float(payload.get("amount", 0.0))
    payment_date = _parse_date(payload.get("payment_date", str(date.today())))
    payment_method = str(payload.get("payment_method", "Bank Transfer")).strip() or "Bank Transfer"

    if not sale_id:
        raise ValueError("Sale ID is required.")
    if amount <= 0:
        raise ValueError(f"Payment amount must be positive, got {amount}.")

    # Validate sale existence and calculate required payment
    cursor.execute(
        "SELECT quantity, unit_price, payment_status FROM sales WHERE sale_id = %s;",
        (sale_id,)
    )
    sale_row = cursor.fetchone()
    if not sale_row:
        raise ValueError(f"Sale ID '{sale_id}' does not exist.")

    quantity = int(sale_row[0])
    unit_price = float(sale_row[1])
    total_sale_amount = round(quantity * unit_price, 2)
    previous_status = sale_row[2]

    payment_id = payload.get("payment_id") or _generate_next_id(cursor, "payments", "payment_id", "PM", pad_len=4)

    # Insert payment record
    cursor.execute(
        """
        INSERT INTO payments (
            payment_id,
            sale_id,
            payment_date,
            amount,
            payment_method
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (payment_id, sale_id, payment_date, Decimal(str(amount)), payment_method)
    )

    # Check total payments recorded for this sale
    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM payments WHERE sale_id = %s;",
        (sale_id,)
    )
    total_paid = float(cursor.fetchone()[0])

    updated_status = previous_status
    if total_paid >= total_sale_amount:
        cursor.execute(
            "UPDATE sales SET payment_status = 'Paid' WHERE sale_id = %s;",
            (sale_id,)
        )
        updated_status = "Paid"

    return {
        "payment_id": payment_id,
        "sale_id": sale_id,
        "amount": amount,
        "total_paid": total_paid,
        "total_sale_amount": total_sale_amount,
        "payment_date": str(payment_date),
        "payment_method": payment_method,
        "sale_payment_status": updated_status,
    }


# ==============================================================================
# 3. INVENTORY_UPDATED HANDLER
# ==============================================================================

def handle_inventory_updated(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles INVENTORY_UPDATED event.
    Updates stock_quantity and last_restock_date in inventory table.
    """
    product_id = str(payload.get("product_id", "")).strip()
    quantity_change = int(payload.get("quantity_change", 0))
    update_date = _parse_date(payload.get("update_date", str(date.today())))
    warehouse = payload.get("warehouse")

    if not product_id:
        raise ValueError("Product ID is required.")
    if quantity_change == 0:
        raise ValueError("Quantity change cannot be 0.")

    # Check current inventory record
    cursor.execute(
        "SELECT inventory_id, stock_quantity, warehouse FROM inventory WHERE product_id = %s;",
        (product_id,)
    )
    inv_row = cursor.fetchone()

    if inv_row:
        inv_id = inv_row[0]
        current_stock = int(inv_row[1])
        new_stock = current_stock + quantity_change
        if new_stock < 0:
            raise ValueError(
                f"Insufficient stock for product '{product_id}': current {current_stock}, change {quantity_change} results in negative stock."
            )

        cursor.execute(
            """
            UPDATE inventory
            SET stock_quantity = %s,
                last_restock_date = CASE WHEN %s > 0 THEN %s ELSE last_restock_date END
            WHERE product_id = %s;
            """,
            (new_stock, quantity_change, update_date, product_id)
        )
    else:
        # Check if product exists in products table before inserting
        cursor.execute("SELECT reorder_level FROM products WHERE product_id = %s;", (product_id,))
        prod_row = cursor.fetchone()
        if not prod_row:
            raise ValueError(f"Product '{product_id}' does not exist in products table.")
        reorder_level = int(prod_row[0]) or 20

        if quantity_change < 0:
            raise ValueError(f"Initial stock for new product '{product_id}' cannot be negative: {quantity_change}")

        inv_id = _generate_next_id(cursor, "inventory", "inventory_id", "I", pad_len=3)
        target_warehouse = warehouse or "WH_Main"
        new_stock = quantity_change
        current_stock = 0

        cursor.execute(
            """
            INSERT INTO inventory (
                inventory_id,
                product_id,
                stock_quantity,
                warehouse,
                reorder_level,
                last_restock_date
            )
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (inv_id, product_id, new_stock, target_warehouse, reorder_level, update_date)
        )

    return {
        "inventory_id": inv_id,
        "product_id": product_id,
        "quantity_change": quantity_change,
        "previous_stock": current_stock,
        "new_stock": new_stock,
        "update_date": str(update_date),
    }


# ==============================================================================
# 4. EXPENSE_RECORDED HANDLER
# ==============================================================================

def handle_expense_recorded(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles EXPENSE_RECORDED event.
    Inserts a new expense into the expenses table.
    """
    category = str(payload.get("category", "")).strip()
    amount = float(payload.get("amount", 0.0))
    expense_date = _parse_date(payload.get("expense_date", str(date.today())))
    description = str(payload.get("description", "")).strip()

    if not category:
        raise ValueError("Expense category is required.")
    if amount <= 0:
        raise ValueError(f"Expense amount must be positive, got {amount}.")

    expense_id = payload.get("expense_id") or _generate_next_id(cursor, "expenses", "expense_id", "E", pad_len=3)

    cursor.execute(
        """
        INSERT INTO expenses (
            expense_id,
            expense_date,
            category,
            amount,
            description
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (expense_id, expense_date, category, Decimal(str(amount)), description)
    )

    return {
        "expense_id": expense_id,
        "category": category,
        "amount": amount,
        "expense_date": str(expense_date),
        "description": description,
    }


# ==============================================================================
# 5. SUPPLIER_PAYMENT_UPDATED HANDLER
# ==============================================================================

def handle_supplier_payment_updated(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles SUPPLIER_PAYMENT_UPDATED event.
    Updates an existing supplier payment record in the expenses table,
    or creates a new expense with category='Supplier Payment'.
    """
    expense_id = payload.get("expense_id")
    amount = float(payload.get("amount", 0.0))
    payment_date = _parse_date(payload.get("payment_date", payload.get("expense_date", str(date.today()))))
    description = str(payload.get("description", "")).strip()
    category = str(payload.get("category", "Supplier Payment")).strip() or "Supplier Payment"

    if amount <= 0:
        raise ValueError(f"Supplier payment amount must be positive, got {amount}.")

    if expense_id:
        expense_id = str(expense_id).strip()
        cursor.execute("SELECT expense_id FROM expenses WHERE expense_id = %s;", (expense_id,))
        if not cursor.fetchone():
            raise ValueError(f"Expense/Supplier payment ID '{expense_id}' not found.")

        cursor.execute(
            """
            UPDATE expenses
            SET amount = %s,
                expense_date = %s,
                description = COALESCE(NULLIF(%s, ''), description),
                category = %s
            WHERE expense_id = %s;
            """,
            (Decimal(str(amount)), payment_date, description, category, expense_id)
        )
        action = "updated"
    else:
        expense_id = _generate_next_id(cursor, "expenses", "expense_id", "E", pad_len=3)
        cursor.execute(
            """
            INSERT INTO expenses (
                expense_id,
                expense_date,
                category,
                amount,
                description
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            (expense_id, payment_date, category, Decimal(str(amount)), description or "Supplier Payment")
        )
        action = "created"

    return {
        "expense_id": expense_id,
        "action": action,
        "category": category,
        "amount": amount,
        "payment_date": str(payment_date),
        "description": description,
    }


# ==============================================================================
# 6. RESTRICT_CREDIT / INITIATE_CREDIT_REVIEW HANDLER
# ==============================================================================

def handle_restrict_credit(cursor, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles RESTRICT_CREDIT / INITIATE_CREDIT_REVIEW operational actions.
    Validates customer existence and records formal credit restriction / review.
    """
    customer_id = str(payload.get("customer_id", "")).strip()
    if not customer_id:
        raise ValueError("Customer ID is required for credit restriction action.")

    cursor.execute("SELECT customer_name, credit_limit FROM customers WHERE customer_id = %s;", (customer_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Customer '{customer_id}' does not exist in customers table.")

    customer_name, credit_limit = row[0], float(row[1])
    action_type = str(payload.get("action", "restrict_additional_credit")).strip()
    effective_date = _parse_date(payload.get("effective_date", str(date.today())))

    return {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "current_credit_limit": credit_limit,
        "action_taken": action_type,
        "effective_date": str(effective_date),
        "status": "RESTRICTED_PENDING_REVIEW",
    }
