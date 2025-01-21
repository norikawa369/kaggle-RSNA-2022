# kaggle-RSNA-2022-Cervical-Spine-Fracture-Detection
RSNA2022 spine fracture detectionコンペのリポジトリ  
[kaggle日記](https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068)を参考にしてつけていく

## Result
- PL…0.4233(46位)
- PL…0.4694(42位)
- silver medal !!

## Log
### 2022/09/07
- 初submit!!
- とりあえず,Effnetのnotebookを参考に作成.LB 0.47
- 計算環境はcolabを使う予定, 実験管理はwandb, notebook単位で管理予定.
### 2022/09/08
- localでスクリプト作成し、colabで実行する
- 実験管理はwandbでやりたい
- gitでversion管理していこう

### 2022/10/29
- 結局GCPを使って計算した
- 今回はVSCode使わず、バージョン管理もしてないので、完成版のコードのみをlocalで管理することにした
- 上位のgold modalをとるためには独自の視点で戦う必要あり。スコアの高いnotebookにとらわれないというのも必要な気がする。