# lab-tutorial-nmt
二宮・梶原研究室の機械翻訳班のチュートリアル

# LSTMCellを用いたLSTM Seq2Seqの実装
英日対訳コーパス ASPEC(https://jipsti.jst.go.jp/aspec/) を用いて、英語から日本語への翻訳を実装しました。

実装にはtorch.LSTMCellを利用しています。
## ファイルの場所
配置しているファイルの場所を示します。なお、学習データと外部ソフト(Kyteaとmoses)のスクリプトは付属していません。

| path | Description |
| --- | --- |
| `data/model_weight/` | 学習済みのモデルウェイト(state_dict)、`{epoch数}_{epoch bleu(nltk)}`の書式 |
| `data/test_output/` | testデータを翻訳して出力した英語文 |
| `script/python/` | 学習などに用いたスクリプトやクラスファイルが配置されている |
| `script/python/_train.ipynb` | モデルを学習するスクリプト |
| `script/python/_test.ipynb` | testデータを翻訳し、`data/test_output/`に結果を書き出すスクリプト |
| `script/shell/` | トークナイズやmosesを用いたBLEU評価などを行うshellスクリプト |

## 実験設定(チュートリアルで制限)
| パラメータ | 設定値 |
| --- | --- |
| 学習データ数 | 20,000文 |
| 学習データの最大文長 | 50 |
| バッチサイズ | 64 |
| エポック数 | 20と24についてmosesでBLEU値を計算 |
| Embeddingの次元数 | 256 |
| LSTMCellの隠れ層 | 256 |
| Optimizer | Adam |
| 学習率 | 0.001 |
