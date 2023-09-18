# lab-tutorial-nmt
二宮・梶原研究室の機械翻訳班のチュートリアル

# LSTMCellを用いたLSTM Seq2Seqの実装
英日対訳コーパス ASPEC(https://jipsti.jst.go.jp/aspec/) を用いて、英語から日本語への翻訳を実装しました。

実装にはPytorchを利用しています。
## ファイルの場所
配置しているファイルの場所を示します。なお、学習データと外部ソフト(Kyteaとmoses)のスクリプトは付属していません。

| path | Description |
| --- | --- |
| `my_nmt/` | pythonアプリのルート |
| `output/` | モデルの設定値(config)やweight(.pth)、testデータに対する出力値(test_{bleu値}.txt)を保存 |
| `python/` | pythonアプリ制作前のコード |
| `shell/` | 各種shellスクリプト |
| `resource/` | 資源(付属していません) |

## 実験設定(チュートリアルで制限)
...整備中...

## 実験結果
...整備中...
