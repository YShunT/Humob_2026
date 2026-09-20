"""被害情報PDFから市町別の断水推移を抽出する。

実行: python3 src/preprocessing/water_extract.py
入力: events/被害情報_危機管理監室/*_被害情報.pdf
出力: data/processed/events/断水.csv
依存: preprocessing/shelter_extract.py、pdftotext（poppler）
"""

import csv
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from preprocessing.shelter_extract import MUNICIPALITIES, MATCH_ORDER, pdf_lines

PDF_DIR = ROOT / "events" / "被害情報_危機管理監室"
OUT_CSV = ROOT / "data" / "processed" / "events" / "断水.csv"
YEAR = 2024

FIELDNAMES = ("日付", "市町名", "断水戸数", "概数", "状態", "備考")

TOTAL_LABEL = "計"
NAME_ALTERNATION = "|".join(MATCH_ORDER)

# 「輪島市  約10,000戸  ※輪島、門前地区の一部で通水エリアを拡大  浄水施設の修繕…」
ROW_RE = re.compile(
    rf"^\s*({NAME_ALTERNATION}|{TOTAL_LABEL})\s+(約)?\s*([\d,]+)\s*戸\s*(.*)$"
)
HEADER_RE = re.compile(r"市町\s+断水状況")
# 脚注「※断水解消 1月：白山市・加賀市（2日）、津幡町（7日）」と、その続き行「2月：羽咋市（2日）」
RESOLVED_HEAD_RE = re.compile(r"※\s*断水解消")
MONTH_RE = re.compile(r"(\d{1,2})月\s*[：:]")
ENTRY_RE = re.compile(rf"((?:{NAME_ALTERNATION})(?:\s*[・、,]\s*(?:{NAME_ALTERNATION}))*)\s*[（(]\s*(\d{{1,2}})\s*日")

STATE_OUTAGE = "断水中"
STATE_RESOLVED = "解消済み"
STATE_UNREPORTED = "未報告"


# PDF解析
def to_int(text):
    return int(text.replace(",", ""))


def parse_table(lines):
    """断水表の市町行と計行を返す。表が無ければ ``(None, None)``。"""
    for index, line in enumerate(lines):
        if not HEADER_RE.search(line):
            continue
        rows, total = {}, None
        for candidate in lines[index:index + 40]:
            found = ROW_RE.match(candidate)
            if not found:
                continue
            name, approx, digits, remark = found.groups()
            record = {
                "断水戸数": to_int(digits),
                "概数": "T" if approx else "F",
                "備考": re.sub(r"\s{2,}", " ", remark).strip(),
            }
            if name == TOTAL_LABEL:
                total = record
                break
            if name in rows:      # 同じ市町が2度出たら別の表に入っている
                break
            rows[name] = record
        if rows:
            return rows, total
    return None, None


def parse_resolved(lines):
    """脚注から ``{市町名: YYYYMMDD or None}`` を返す。日付なしで列挙される版もある。

    1月の版は「※ 断水解消：白山市、加賀市、…」と名前だけを並べる。2月以降は
    「※断水解消 1月：白山市・加賀市（2日）、…」と月と日を持ち、過去分も累積して載る。
    日付が読めた場合は日付、名前だけの場合は ``None`` を返し、呼び出し側で解決する。
    """
    resolved = {}
    for index, line in enumerate(lines):
        if not RESOLVED_HEAD_RE.search(line):
            continue
        # 脚注は「1月：…」「2月：…」と複数行に折り返す
        for candidate in lines[index:index + 8]:
            parts = MONTH_RE.split(candidate)
            if len(parts) > 1:
                for i in range(1, len(parts) - 1, 2):
                    month = int(parts[i])
                    for names, day in ENTRY_RE.findall(parts[i + 1]):
                        for name in re.split(r"[・、,]", names):
                            name = name.strip()
                            if name in MUNICIPALITIES:
                                resolved[name] = YEAR * 10000 + month * 100 + int(day)
            elif candidate is lines[index]:
                # 日付を持たない版。名前だけ拾って未確定として置く
                for name in re.findall(NAME_ALTERNATION, candidate):
                    resolved.setdefault(name, None)
        break
    return resolved


def check_total(rows, total):
    """市町行の合計と表末尾の「計」行を突き合わせる。"""
    if total is None:
        return ["計行なし"]
    got = sum(record["断水戸数"] for record in rows.values())
    want = total["断水戸数"]
    # 原本は各行が概数のため、計と数百戸ずれることがある。1%を許容幅とする。
    if want and abs(got - want) > max(10, 0.01 * want):
        return [f"断水戸数 合計{got:,} != 計{want:,}"]
    return []


# 全ファイルの集約
def collect():
    """全PDFを日付順に処理し、``(CSV行, 検算メモ, 解消日)`` を返す。

    1パス目で全PDFの脚注から解消日を集める。日付つきの記載は後のPDFに現れるため、
    先に全部読んでからでないと1月分を正しく埋められない。
    """
    paths = sorted(PDF_DIR.glob("*_被害情報.pdf"))
    parsed = {}
    resolved = {}
    for path in paths:
        stem = path.stem.split("_")[0]
        month, day = (int(part) for part in stem.split("-"))
        date_int = YEAR * 10000 + month * 100 + day
        lines = pdf_lines(path)
        rows, total = parse_table(lines)
        if rows is None:
            continue
        parsed[date_int] = (rows, total)
        for name, when in parse_resolved(lines).items():
            if when is not None:
                resolved[name] = when          # 日付つきが最優先
            elif name not in resolved:
                resolved[name] = date_int      # 日付なし版はそのPDFの日付で代用

    records, notes = [], []
    for date_int in sorted(parsed):
        rows, total = parsed[date_int]
        date_text = f"{date_int // 10000}-{date_int // 100 % 100:02d}-{date_int % 100:02d}"
        problems = check_total(rows, total)
        if problems:
            notes.append(f"{date_text}  検算NG: {' / '.join(problems)}")
        for name in MUNICIPALITIES:
            if name in rows:
                record = rows[name]
                state = STATE_OUTAGE
                households, approx, remark = (
                    record["断水戸数"], record["概数"], record["備考"]
                )
            elif name in resolved and resolved[name] <= date_int:
                state, households, approx, remark = STATE_RESOLVED, 0, "F", ""
            else:
                state, households, approx, remark = STATE_UNREPORTED, "", "", ""
            records.append({
                "日付": date_text,
                "市町名": name,
                "断水戸数": households,
                "概数": approx,
                "状態": state,
                "備考": remark,
            })
    records.sort(key=lambda row: (row["日付"], MUNICIPALITIES.index(row["市町名"])))
    return records, notes, resolved


# 実行
def run():
    records, notes, resolved = collect()
    dates = sorted({row["日付"] for row in records})
    counts = {}
    for row in records:
        counts[row["状態"]] = counts.get(row["状態"], 0) + 1
    print(f"[parse] {len(records)}行 / {len(dates)}日付 ({dates[0]} 〜 {dates[-1]})")
    print("  状態の内訳: " + " / ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("  読めた解消日: " + ", ".join(
        f"{k} {v // 100 % 100}/{v % 100}" for k, v in sorted(resolved.items(), key=lambda kv: kv[1])
    ))
    if notes:
        print(f"[note] {len(notes)}件")
        for note in notes:
            print(f"  {note}")
    else:
        print("[note] 計行との検算はすべて一致")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"[write] {OUT_CSV}")


if __name__ == "__main__":
    run()
