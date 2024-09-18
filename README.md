# フレーム画像予測

[参考にしたもの](https://github.com/m-mejiaj/next-frame-prediction/blob/main/moving_mnist.ipynb)

過去数フレームから未来のフレーム画像を予測する

## 学習

```
python train.py --root_dir='dataset' --seqlen='seqence_num'
```
## 予測

```
python predict.py --root_dir='dataset' --seqlen='seqence_num'
```

