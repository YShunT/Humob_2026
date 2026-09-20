"""被害情報PDFから市町別の1次避難所推移を抽出する。

実行: python3 src/preprocessing/shelter_extract.py
入力: events/被害情報_危機管理監室/*_被害情報.pdf
出力: data/processed/events/避難所.csv
依存: pdftotext（poppler）、標準ライブラリ
"""

import csv
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

PDF_DIR = ROOT / "events" / "被害情報_危機管理監室"
OUT_CSV = ROOT / "data" / "processed" / "events" / "避難所.csv"
YEAR = 2024

FIELDNAMES = (
    "日付",
    "市町名",
    "開設数(箇所)",
    "避難者数(人)",
    "広域避難所(箇所)",
    "広域避難者数(人)",
    "閉鎖",
)

# 原本の表に載る19市町を、原本の並び順のまま持つ（CSVの行順もこれに合わせる）
MUNICIPALITIES = (
    "金沢市", "七尾市", "小松市", "輪島市", "珠洲市", "加賀市", "羽咋市", "かほく市",
    "白山市", "能美市", "野々市市", "川北町", "津幡町", "内灘町", "志賀町",
    "宝達志水町", "中能登町", "穴水町", "能登町",
)
# 正規表現の選択肢は長い名前を先に並べ、部分一致での取り違えを防ぐ
MATCH_ORDER = tuple(sorted(MUNICIPALITIES, key=len, reverse=True))
TOTAL_LABEL = "計"

# 全角数字・全角カンマ・全角括弧・全角空白を半角へ寄せてから正規表現をかける
ZEN2HAN = str.maketrans("０１２３４５６７８９，．（）　", "0123456789,.() ")

NAME_ALTERNATION = "|".join(MATCH_ORDER)
# 発災直後は「約3,700」のような概数で載る日がある（1/2・1/3）。約は落として数値を採る。
ROW_RE = re.compile(
    rf"^\s*({NAME_ALTERNATION}|{TOTAL_LABEL})\s+約?([\d,]+)\s+約?([\d,]+)\s*(.*)$"
)
# 箇所数と人数の区切りは「・」が基本だが、1/21・1/22 の羽咋市だけ「、」になっている
WIDE_RE = re.compile(r"他に広域避難所\s*([\d,]+)\s*カ所\s*[・、,]\s*([\d,]+)\s*人")
ASOF_RE = re.compile(r"令和6年\s*(\d{1,2})月\s*(\d{1,2})日[^】]*?現在")
SECTION_RE = re.compile(r"避難所の開設状況")
HEADER_RE = re.compile(r"市町名.*開設数")


# PDF解析
def to_int(text):
    return int(text.replace(",", ""))


def pdf_lines(path):
    """pdftotext -layout の出力を、全角を寄せた行リストで返す。"""
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, check=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return [line.translate(ZEN2HAN) for line in text.split("\n")]


def find_table_start(lines):
    """「避難所の開設状況」の後にある市町表のヘッダー行番号を返す。無ければ None。

    見出しは年月で書式が変わる（`（１）市町１次避難所` / `（１）市町避難所` /
    `○市町１次避難所` / 見出し無し）ため、節見出しの後に最初に現れる
    「市町名…開設数」のヘッダー行を表の開始とみなす。
    """
    for index, line in enumerate(lines):
        if not SECTION_RE.search(line):
            continue
        for offset in range(index, min(index + 12, len(lines))):
            if HEADER_RE.search(lines[offset]):
                return offset + 1
    return None


def parse_remarks(remarks):
    """備考欄から広域避難所の箇所数・人数と、閉鎖済みかどうかを取り出す。"""
    wide = WIDE_RE.search(remarks)
    wide_count = to_int(wide.group(1)) if wide else 0
    wide_evacuees = to_int(wide.group(2)) if wide else 0
    return wide_count, wide_evacuees, "閉鎖" in remarks


def skip_reason(lines):
    """表を取れなかったPDFについて、その理由を1行で返す。"""
    japanese = sum(1 for line in lines for char in line if "\u3040" <= char <= "\u9fff")
    if japanese < 50:
        return "テキスト層が壊れている（CIDフォントにToUnicodeなし・要OCR）"
    if not any(SECTION_RE.search(line) for line in lines):
        return "避難所の開設状況の節がない"
    return "節はあるが別書式（箇所数のみの箇条書きで避難者数なし）"


def parse_pdf(path):
    """1つのPDFから ``(日付, 市町行のリスト, 計行)`` を返す。表が無ければ理由を返す。"""
    lines = pdf_lines(path)

    as_of = None
    for line in lines[:40]:
        found = ASOF_RE.search(line)
        if found:
            as_of = f"{YEAR}-{int(found.group(1)):02d}-{int(found.group(2)):02d}"
            break

    start = find_table_start(lines)
    if start is None:
        return skip_reason(lines)

    rows, total, seen = [], None, set()
    for line in lines[start:start + 40]:
        found = ROW_RE.match(line)
        if not found:
            # 空行やページヘッダーは読み飛ばし、市町行が続く限り表とみなす
            if line.strip() and rows and TOTAL_LABEL in line:
                break
            continue
        name, opened, evacuees, remarks = found.groups()
        wide_count, wide_evacuees, closed = parse_remarks(remarks)
        record = {
            "市町名": name,
            "開設数(箇所)": to_int(opened),
            "避難者数(人)": to_int(evacuees),
            "広域避難所(箇所)": wide_count,
            "広域避難者数(人)": wide_evacuees,
            "閉鎖": "T" if closed else "F",
        }
        if name == TOTAL_LABEL:
            total = record
            break
        if name in seen:      # 同じ市町が2度出たら別表に入っている
            break
        seen.add(name)
        rows.append(record)

    if not rows:
        return skip_reason(lines)
    return as_of, rows, total


def check_total(rows, total):
    """市町行の合計と表末尾の「計」行を突き合わせ、食い違いの一覧を返す。"""
    if total is None:
        return ["計行なし"]
    problems = []
    for column in ("開設数(箇所)", "避難者数(人)", "広域避難所(箇所)", "広域避難者数(人)"):
        got = sum(row[column] for row in rows)
        want = total[column]
        if got != want:
            problems.append(f"{column} 合計{got} != 計{want}")
    return problems


# 全ファイルの集約
def collect():
    """全PDFを日付順に処理し、``(CSV行, 検算メモ)`` を返す。"""
    records, notes = [], []
    for path in sorted(PDF_DIR.glob("*_被害情報.pdf")):
        stem = path.stem.split("_")[0]
        month, day = (int(part) for part in stem.split("-"))
        file_date = f"{YEAR}-{month:02d}-{day:02d}"

        parsed = parse_pdf(path)
        if isinstance(parsed, str):
            notes.append(f"{file_date}  収録せず: {parsed}")
            continue
        as_of, rows, total = parsed

        if as_of is None:
            notes.append(f"{file_date}  本文の「現在」日時を読めずファイル名の日付を採用")
            as_of = file_date
        elif as_of != file_date:
            notes.append(f"{file_date}  本文は {as_of} 現在（ファイル名と不一致・本文を採用）")

        problems = check_total(rows, total)
        if problems:
            notes.append(f"{file_date}  検算NG: {' / '.join(problems)}")

        for row in rows:
            records.append({"日付": as_of, **row})

    records.sort(key=lambda row: (row["日付"], MUNICIPALITIES.index(row["市町名"])))
    return records, notes


# 実行
def run():
    records, notes = collect()
    dates = sorted({row["日付"] for row in records})
    print(f"[parse] {len(records)}行 / {len(dates)}日付 "
          f"({dates[0]} 〜 {dates[-1]}) / {len({r['市町名'] for r in records})}市町")
    if notes:
        print(f"[note] {len(notes)}件")
        for note in notes:
            print(f"  {note}")
    else:
        print("[note] 検算はすべて一致")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"[write] {OUT_CSV}")


if __name__ == "__main__":
    run()
