# EXP02 避難所到着地重み付け診断

## 仮説・手法・検証

避難所なしdestinationへの配分を抑えることでEXP02の過大予測を改善できるかを確認した。
登録施設の有無で重み付けし、評価域内のorigin別非対角総量を再正規化で保存。
対角、範囲外、候補集合は固定。指定14日のみを使用し、追加学習は行っていない。
元TSVのSHA256不変、保存TSVの再読込一致、全origin非対角総量の保存、非負性を検査した。

| lambda | Combined | 対角NRMSE | 非対角NRMSE | jan | apr | EXP02との差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000000 | 0.220699 | 0.108684 | 0.332715 | 0.231813 | 0.209586 | 0.000000 |
| 0.800000 | 0.221730 | 0.108684 | 0.334776 | 0.233858 | 0.209601 | 0.001031 |
| 0.500000 | 0.233598 | 0.108684 | 0.358511 | 0.246673 | 0.220523 | 0.012898 |
| 0.100000 | 0.293063 | 0.108684 | 0.477443 | 0.308239 | 0.277887 | 0.072364 |

## 結果と解釈

4候補の最小はlambda=1.0。
抑制候補はいずれも基準を改善せず、今回の補正は不採用。
施設あり・なしの両方へ流れるoriginだけ配分が変わる。どちらか一方だけなら重みが相殺される。
今回の結果だけで施設数の連続値補正、時変の避難情報、別の総量調整の有効性は判断できない。

## 保存則の検算

- lambda=1.0: {"mixed_origin_days": 550, "changed_origin_days": 0, "max_abs_off_total_delta": 0.0, "max_abs_daily_total_delta": 0.0}
- lambda=0.8: {"mixed_origin_days": 550, "changed_origin_days": 550, "max_abs_off_total_delta": 3.552713678800501e-15, "max_abs_daily_total_delta": 0.0}
- lambda=0.5: {"mixed_origin_days": 550, "changed_origin_days": 550, "max_abs_off_total_delta": 7.105427357601002e-15, "max_abs_daily_total_delta": 0.0}
- lambda=0.1: {"mixed_origin_days": 550, "changed_origin_days": 550, "max_abs_off_total_delta": 3.552713678800501e-15, "max_abs_daily_total_delta": 0.0}

EXP02本体のmetrics.json、元CV予測、提出物は更新していない。
