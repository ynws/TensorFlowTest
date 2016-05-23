# -*- coding: utf-8 -*-

# �e�X�g�p�f�[�^�������Ă��� @MNIST
# 0�`9 ���菑���������l���f�W�^����������� 70,000�܂܂��
# ���̂���55,000���w�K�ɗp���A5,000�͌��؁A10,000�𐸓x�]���ɗp����
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# ���̓e���\���m�ہBNone�ɂ͊w�K�p�f�[�^��(55,000)������
x = tf.placeholder(tf.float32, [None, 784]) # 28*28=784�s�N�Z���̉摜�f�[�^
                                            # �ȒP�̂��߉摜���ꎟ���ɂ���
y_ = tf.placeholder(tf.float32, [None, 10]) # �����W(0�`9��1������1�ő���0�ƂȂ�z��)

# �w�K�p�p�����[�^�o�b�t�@(�����l0)
W = tf.Variable(tf.zeros([784, 10]))    # �d��(784���� -> 10������)
b = tf.Variable(tf.zeros([10]))         # �o�C�A�X

# �������֐�
y = tf.nn.softmax(tf.matmul(x, W) + b)  # x dot W + b �œ�����10�����̃p�����^�ɑ΂�
                                        # softmax(�S�o�͂̑��a��1�ƂȂ�)���v�Z

# �w�K���@
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    # �v�Z���ʂƓ����̌덷�̌v�Z���@
# �o�b�N�v���p�Q�[�V�����̐ݒ�B�w�K��0.01��cross_entropy���ŏ�������
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# ������̃\�[�X�ł�100�摜���Ƃɂ܂Ƃ߂Č덷���v�Z���Ă��邱�Ƃɒ���


# �e��ϐ��̏��������s
init = tf.initialize_all_variables()

# �Z�b�V�����J�n
sess = tf.Session()
sess.run(init)

# �w�K���[�v1000��B
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # �w�K�f�[�^�����_����100���o���āE�E�E
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # �w�K���s

# ��v������
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # �����W�ƌv�Z���ʂ�equal���H
# ���ʂ̏W�v
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# �e�X�g�f�[�^(10,000��)�̃��x�����O���x�����߂�
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# ��������91%��v����

# �Z�b�V�����I��
sess.close()
