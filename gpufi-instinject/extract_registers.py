#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import os

# 自然排序：把字符串中的数字片段按数值排序
def natural_key(s: str):
    name = s.lstrip('%')
    parts = re.split(r'(\d+)', name)
    return tuple(int(p) if p.isdigit() else p for p in parts)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <basename>")
        sys.exit(1)

    base = sys.argv[1]
    ptx_file = f"{base}.ptx"
    txt_file = "register_used.txt"

    if not os.path.exists(ptx_file):
        print(f"Error: {ptx_file} not found.")
        sys.exit(1)

    with open(ptx_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    registers = set()

    # 需要排除的“基名”前缀（不带 %，不区分大小写）
    # 你要求排除 %ctaid，即使没有 .x；这里用 startswith 覆盖 ctaid 的复合形式。
    blocked_prefixes = ("ctaid",)

    def is_blocked(reg: str) -> bool:
        """reg 形如 %name[...]；屏蔽带点的，以及基名前缀命中 blocked_prefixes 的。"""
        if '.' in reg:
            return True
        m = re.match(r"%([A-Za-z_]+)", reg)
        if not m:
            return False
        base = m.group(1).lower()
        return any(base.startswith(bp) for bp in blocked_prefixes)

    # 1) 展开 %name<COUNT> 形式（如 %r<122>、%rd<13>）
    for base_name, count_str in re.findall(r"%(?P<base>[A-Za-z_][A-Za-z0-9_]*)<(?P<count>\d+)>", text):
        if any(base_name.lower().startswith(bp) for bp in blocked_prefixes):
            continue  # 跳过被屏蔽基名的范围声明
        count = int(count_str)
        for i in range(count):
            reg = f"%{base_name}{i}"
            if not is_blocked(reg):
                registers.add(reg)

    # 2) 捕获一般寄存器名（明确排除 %name<COUNT> 和任何带点号的）
    general_pattern = re.compile(
        r"%(?![A-Za-z_][A-Za-z0-9_]*<)"      # 非 %name<COUNT>
        r"[A-Za-z_][A-Za-z0-9_]*"            # 主体：字母/数字/下划线
        r"(?:_[A-Za-z0-9_]+)*"               # 允许 _suffix（如 _sh1）
        r"(?:\d+)?"
    )

    for m in general_pattern.finditer(text):
        reg = m.group(0)
        if not is_blocked(reg):
            registers.add(reg)

    # 去重 + 自然排序
    registers = sorted(registers, key=natural_key)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(registers))

    print(f"Extracted {len(registers)} registers -> {txt_file}")

if __name__ == "__main__":
    main()
