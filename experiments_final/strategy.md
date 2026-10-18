# FINAL — EXP04 採用モデル（N-B）を単独で再現できるコードに再構成する

- **コード**: [`src/main.py`](../src/main.py)（前処理コードは非公開）
- **構成の説明**: [`src/README.md`](../src/README.md)
- **比較元**: EXP04（実験記録は非公開）

```bash
python3 src/main.py --validate
python3 src/main.py --submit
```

## 狙い

新しい手法は試さない。EXP04 で採用した N-B（対角 D5 ＋ 出発側・到着側の総量×配分を1:1平均）だけを、
EXP03/EXP04 の実験コードに依存せず、前処理から提出物まで1から実行できる構成にする。
他の実験の再現や候補比較は対象外。

## 固定する条件

- 特徴群・ハイパーパラメータ・EM上限200回・震災日の状態リセット・CV14日・提出58日は EXP04 と同一。
- 前処理CSVは既存の `data/processed/` とバイト単位で同じものを作る。
- 出力は `experiments_final/` に書き、EXP04 の記録は変更しない。

## 合格条件

1. 前処理コード（非公開）で再生成した8ファイルが、既存の `data/processed/` とすべて一致する。
2. `--validate` の14日・1月・4月スコアと5成分の収束診断が EXP04 の metrics.json と小数6桁で一致する。
3. `--submit` の提出物が公式validatorを通り、EXP04 の submission.tsv と浮動小数点の誤差内で一致する。
