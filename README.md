# フレーム画像予測

[参考にしたもの](https://github.com/m-mejiaj/next-frame-prediction/blob/main/moving_mnist.ipynb)

過去数フレームから未来のフレーム画像を予測する

## 学習

```
python train.py --root_dir='dataset' --seqlen='seqence_num' --save='save_file' --batch='batch_size'
```
## 予測

```
python predict.py --root_dir='dataset' --seqlen='seqence_num' --load='load_file'
```

# 実験結果
MDVDとUCSDped1に対して、4フレームごとに取得したもの（毎フレーム使用すると膨大になる上にあまりフレーム間で変化がないため）
を用意し、正常クラスのもののみで学習させた。過去19フレームから次のフレームを予測する。

## MDVD　（MDVD_train_each_4.h5）
### 正常クラス
![image](https://github.com/user-attachments/assets/ff6676dc-fd09-404d-81ce-ff50130a63ae)
![image](https://github.com/user-attachments/assets/2fa81a3a-3bc3-4e47-a1cf-059a2b4574e2)

### 異常クラス
![image](https://github.com/user-attachments/assets/66d3546f-2fb4-43d0-98c9-8e4e0d49306c)
![image](https://github.com/user-attachments/assets/c60936c6-a952-40ce-9169-78d27c1cd0aa)

### 疑惑クラス
![image](https://github.com/user-attachments/assets/af546a7b-c591-4ff7-bbb2-18f5d61362f1)
![image](https://github.com/user-attachments/assets/2429851e-9007-44cc-b888-45d3f8558317)

## UCSDped1　（UCSDped1_train_each_4.h5）
### 正常クラス
![image](https://github.com/user-attachments/assets/769579d3-baad-48b9-b985-fc4ac5e29c72)
![image](https://github.com/user-attachments/assets/91e87ad0-f9f6-46ee-b01c-356665621422)

### 異常クラス
![image](https://github.com/user-attachments/assets/7279bd7e-36fe-44b2-b51a-0a0c6a63ef28)
![image](https://github.com/user-attachments/assets/fc372b85-fae9-47e1-a339-a3e5ffa512fb)




