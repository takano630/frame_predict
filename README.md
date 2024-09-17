# フレーム画像予測

[参考にしたもの](https://github.com/m-mejiaj/next-frame-prediction/blob/main/moving_mnist.ipynb)

過去数フレームから未来のフレーム画像を予測する

```
python frame_redict_aug.py --root_dir='dataset' --seqlen='seqence_num'
```

## 結果
MDVDデータセット　Normalラベルのものを4フレームごとに取得　←　べつのプログラムで行った 

train:val:test = 6:2:2　の割合

9枚のフレーム画像から次の時刻を予測した　画像はlogに
```
loss: 0.6222 - mean_squared_error: 0.0058
Mse of the model is 0.005831328686326742
```
