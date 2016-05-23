# -*- coding: utf-8 -*-

# テスト用データを持ってくる @MNIST
# 0～9 を手書きした数値をデジタル化した情報が 70,000個含まれる
# そのうち55,000個を学習に用い、5,000個は検証、10,000個を精度評価に用いる
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# 入力テンソル確保。Noneには学習用データ数(55,000)が入る
x = tf.placeholder(tf.float32, [None, 784]) # 28*28=784ピクセルの画像データ
                                            # 簡単のため画像を一次元にする
y_ = tf.placeholder(tf.float32, [None, 10]) # 正解集(0～9の1つだけが1で他は0となる配列)

# 学習用パラメータバッファ(初期値0)
W = tf.Variable(tf.zeros([784, 10]))    # 重み(784次元 -> 10次元へ)
b = tf.Variable(tf.zeros([10]))         # バイアス

# 活性化関数
y = tf.nn.softmax(tf.matmul(x, W) + b)  # x dot W + b で得られる10次元のパラメタに対し
                                        # softmax(全出力の総和が1となる)を計算

# 学習方法
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    # 計算結果と答えの誤差の計算方法
# バックプロパゲーションの設定。学習率0.01でcross_entropyを最小化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# ※今回のソースでは100画像ごとにまとめて誤差を計算していることに注意


# 各種変数の初期化実行
init = tf.initialize_all_variables()

# セッション開始
sess = tf.Session()
sess.run(init)

# 学習ループ1000回。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 学習データランダムに100個取り出して・・・
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 学習実行

# 一致率判定
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # 正解集と計算結果がequalか？
# 結果の集計
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# テストデータ(10,000個)のラベリング精度を求める
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# だいたい91%一致する

# セッション終了
sess.close()
