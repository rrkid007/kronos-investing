"""Approval queue CLI: list, approve, or reject pending paper orders.

Usage:
  python scripts/approve_trades.py                 # list pending orders
  python scripts/approve_trades.py --approve ID [ID...]
  python scripts/approve_trades.py --reject ID [ID...]
  python scripts/approve_trades.py --approve-all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.execution.approval import approve_and_submit
from trading_platform.execution.orders import list_pending, reject_order


def show_pending(conn) -> None:
    pending = list_pending(conn)
    if not pending:
        print("no orders awaiting approval")
        return
    print(f"{'order_id':14s} {'date':12s} {'ticker':7s} {'side':5s} {'qty':>8s} "
          f"{'est_value':>10s}  score/reason")
    for o in pending:
        notes = json.loads(o["notes"] or "{}")
        value = notes.get("est_value")
        score = notes.get("final_score")
        reason = (notes.get("reason") or "")[:50]
        print(f"{o['order_id']:14s} {o['run_date']:12s} {o['ticker']:7s} "
              f"{o['side']:5s} {o['qty']:8g} "
              f"{'$' + format(value, ',.0f') if value else '-':>10s}  "
              f"{score if score is not None else '-'} {reason}")
        memo = conn.execute(
            "SELECT recommendation FROM research_memos WHERE order_id = ?",
            (o["order_id"],),
        ).fetchone()
        if memo:
            print(f"{'':14s} memo: analyst recommends '{memo['recommendation']}' "
                  f"(advisory) — view with --memo {o['order_id']}")
    print(f"\n{len(pending)} order(s) awaiting approval. "
          f"Unapproved orders expire at the next daily run.")


def show_memo(conn, order_id: str) -> None:
    row = conn.execute(
        "SELECT memo_md FROM research_memos WHERE order_id = ?", (order_id,)
    ).fetchone()
    print(row["memo_md"] if row else f"no memo for order {order_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--approve", nargs="+", metavar="ORDER_ID")
    parser.add_argument("--reject", nargs="+", metavar="ORDER_ID")
    parser.add_argument("--approve-all", action="store_true")
    parser.add_argument("--memo", metavar="ORDER_ID",
                        help="print the advisory research memo for an order")
    args = parser.parse_args()

    config = load_config(args.config_dir)
    conn = connect(config.db_path)
    init_db(conn)

    if args.memo:
        show_memo(conn, args.memo)
    elif args.approve_all:
        pending = list_pending(conn)
        for o in pending:
            print(approve_and_submit(conn, config, o["order_id"]))
        if not pending:
            print("nothing to approve")
    elif args.approve or args.reject:
        for order_id in args.approve or []:
            print(approve_and_submit(conn, config, order_id))
        for order_id in args.reject or []:
            reject_order(conn, order_id)
            print(f"rejected {order_id}")
    else:
        show_pending(conn)
    conn.close()


if __name__ == "__main__":
    main()
