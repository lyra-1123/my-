# -*- coding: utf-8 -*-
"""
安全追加 CSV 日志：日志写入失败（例如 Windows 上文件被 Excel 打开而锁住）绝不能让交易程序崩溃。
写不进去时先写到 <文件名>.pending.csv，下次主文件可写时自动并回。查看日志请用只读方式，或先复制一份再用 Excel 打开。
"""
from __future__ import annotations

import csv
import os

import pandas as pd


def _write(path: str, rows: list[dict], fields: list[str]):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def append_row(path: str, row: dict, fields: list[str] | None = None) -> bool:
    """追加一行；返回是否写进了主文件。fields 固定时，旧表头不同会原地升级（保留历史行）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pending = path[:-4] + ".pending.csv" if path.endswith(".csv") else path + ".pending"
    try:
        if fields is None:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    fields = fh.readline().strip().split(",")
            else:
                fields = list(row)
        elif os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
            if header != fields:
                pd.read_csv(path).reindex(columns=fields).to_csv(path, index=False, encoding="utf-8")
        rows = []
        if os.path.exists(pending):
            rows = pd.read_csv(pending, dtype=str, keep_default_na=False).to_dict("records")
        _write(path, rows + [row], fields)
        if rows:
            os.remove(pending)
        return True
    except (PermissionError, OSError) as e:
        try:
            _write(pending, [row], fields or list(row))
            print(f"!!! 日志文件 {path} 被占用（是否用 Excel 打开了？请关闭），本条暂存到 {pending}：{e.__class__.__name__}")
        except Exception as e2:
            print(f"!!! 日志写入失败（不影响交易）：{path}：{e2!r}")
        return False


def flush_pending(path: str) -> None:
    """把暂存文件并回主文件（主文件仍被占用时什么也不做）。"""
    pending = path[:-4] + ".pending.csv" if path.endswith(".csv") else path + ".pending"
    if not os.path.exists(pending):
        return
    try:
        rows = pd.read_csv(pending, dtype=str, keep_default_na=False).to_dict("records")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                fields = fh.readline().strip().split(",")
        else:
            fields = list(rows[0]) if rows else []
        _write(path, rows, fields)
        os.remove(pending)
    except (PermissionError, OSError):
        pass
