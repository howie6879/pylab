import time

from math import sqrt

import tensorflow as tf


class CharConvNet(object):
    def __init__(
        self,
        *,
        conv_layers: list = None,
        fully_layers: list = None,
        input_size=1014,
        alphabet_size=69,
        num_of_classes=4,
        th=1e-6
    ):

        if conv_layers is None:
            conv_layers = [
                [256, 7, 3],
                [256, 7, 3],
                [256, 3, None],
                [256, 3, None],
                [256, 3, None],
                [256, 3, 3],
            ]

        if fully_layers is None:
            fully_layers = [1024, 1024]

        seed = time.time()

        tf.set_random_seed(seed)
        with tf.name_scope("Input-Layer"):
            # Model inputs
            self.input_x = tf.placeholder(
                tf.int64, shape=[None, input_size], name="input_x"
            )
            self.input_y = tf.placeholder(
                tf.float32, shape=[None, num_of_classes], name="input_y"
            )
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob"
            )

        with tf.name_scope("Embedding-Layer"), tf.device("/cpu:0"):
            # Quantization layer
            Q = tf.concat(
                [
                    # # Zero padding vector for out of alphabet characters
                    tf.zeros([1, alphabet_size]),
                    # one-hot vector representation for alphabets
                    tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0),
                ],
                0,
                name="Q",
            )

            x = tf.nn.embedding_lookup(Q, self.input_x)
            # Add the channel dim, thus the shape of x is [batch_size, input_size, alphabet_size, 1]
            x = tf.expand_dims(x, -1)
            # (128, 1014, 69, 1)

        var_id = 0

        # Convolution layers
        for i, cl in enumerate(conv_layers):
            var_id += 1
            with tf.name_scope("ConvolutionLayer"):
                # 第一层 x (128, 1014, 69, 1)
                # 第二层 x (128, 336, 256, 1)
                # 第三层 x (128, 110, 256, 1)
                # 第四层 x (128, 108, 256, 1)
                # 第五层 x (128, 106, 256, 1)
                # 第六层 x (128, 104, 256, 1)
                filter_width = x.get_shape()[2].value
                # 第一层 卷积配置 [256, 7, 3]    结果为 [7, 69, 1, 256]
                # 第二层 卷积配置 [256, 7, 3]    结果为 [7, 256, 1, 256]
                # 第三层 卷积配置 [256, 3, None] 结果为 [3, 256, 1, 256]
                # 第四层 卷积配置 [256, 3, None] 结果为 [3, 256, 1, 256]
                # 第五层 卷积配置 [256, 3, None] 结果为 [3, 256, 1, 256]
                # 第六层 卷积配置 [256, 3, 3]    结果为 [3, 256, 1, 256]
                filter_shape = [cl[1], filter_width, 1, cl[0]]
                # Convolution layer
                stdv = 1 / sqrt(cl[0] * cl[1])
                # # The kernel of the conv layer is a trainable vraiable
                W = tf.Variable(
                    tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),
                    dtype="float32",
                    name="W",
                )
                # # and the biases as well
                b = tf.Variable(
                    tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv),
                    name="b",
                )

                # Perform the convolution operation
                # 第一层 x (128, 1014, 69, 1) W (7, 69, 1, 256) [1, 1, 1, 1] 表示步长
                # 第二层 x (128, 336, 256, 1) W (7, 256, 1, 256)
                # 第三层 x (128, 110, 256, 1) W (3, 256, 1, 256)
                # 第四层 x (128, 108, 256, 1) W (3, 256, 1, 256)
                # 第五层 x (128, 106, 256, 1) W (3, 256, 1, 256)
                # 第六层 x (128, 104, 256, 1) W [3, 256, 1, 256]
                conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID", name="Conv")
                # 第一层 (128, 1008, 1, 256)
                # 第一层 (128, 330, 1, 256)
                # 第三层 (128, 108, 1, 256)
                # 第四层 (128, 106, 1, 256)
                # 第五层 (128, 104, 1, 256)
                # 第六层 (128, 102, 1, 256)
                x = tf.nn.bias_add(conv, b)

            # Threshold
            with tf.name_scope("ThresholdLayer"):
                # 向前传播过程中
                # 对x的值判断是否小于th 然后转化成bool类型
                # 如果小于th 值取0 反之不变
                x = tf.where(tf.less(x, th), tf.zeros_like(x), x)

            if not cl[-1] is None:
                with tf.name_scope("MaxPoolingLayer"):
                    # Maxpooling over the outputs
                    # 第一层 (128, 336, 1, 256)  1008/3 = 336
                    # 第二层 (128, 110, 1, 256)  330/3 = 110
                    # 第六层 (128, 34, 1, 256)   102/3 = 34
                    pool = tf.nn.max_pool(
                        x,
                        ksize=[1, cl[-1], 1, 1],
                        strides=[1, cl[-1], 1, 1],
                        padding="VALID",
                    )
                    # [batch_size, img_width, img_height, 1]
                    x = tf.transpose(pool, [0, 1, 3, 2])
            else:
                # [batch_size, img_width, img_height, 1]
                # 转置操作
                # 第三层直接进行转置 (128, 108, 256, 1)
                # 第四层直接进行转置 (128, 106, 256, 1)
                # 第五层直接进行转置 (128, 104, 256, 1)
                x = tf.transpose(x, [0, 1, 3, 2], name="tr%d" % var_id)
            # 第一层结束 (128, 336, 256, 1)
            # 第二层结束 (128, 110, 256, 1)
            # 第三层结束 (128, 108, 256, 1)
            # 第四层结束 (128, 106, 256, 1)
            # 第五层结束 (128, 104, 256, 1)
            # 第六层结束 (128, 34, 256, 1)

        with tf.name_scope("ReshapeLayer"):
            # Reshape layer
            # x (128, 34, 256, 1)
            # vec_dim = 34 * 256 = 8704
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value
            # shape=(128, 8704)
            x = tf.reshape(x, [-1, vec_dim])

        # The connection from reshape layer to fully connected layers
        # [8704, 1024, 1024]
        weights = [vec_dim] + list(fully_layers)

        for i, fl in enumerate(fully_layers):
            var_id += 1
            with tf.name_scope("LinearLayer"):
                # Fully-Connected layer
                stdv = 1 / sqrt(weights[i])
                # 第一个全连接层 W (8704, 1024)
                # 第二个全连接层 W (1024, 1024)
                W = tf.Variable(
                    tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv),
                    dtype="float32",
                    name="W",
                )
                # 第一次 (1024, )
                # 第二次 (1024, )
                b = tf.Variable(
                    tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv),
                    dtype="float32",
                    name="b",
                )

                # 第一次 (128, 1024)
                # 第二次 (128, 1024)
                x = tf.nn.xw_plus_b(x, W, b)

            with tf.name_scope("ThresholdLayer"):
                x = tf.where(tf.less(x, th), tf.zeros_like(x), x)

            with tf.name_scope("DropoutLayer"):
                # Add dropout
                x = tf.nn.dropout(x, self.dropout_keep_prob)

        with tf.name_scope("OutputLayer"):
            stdv = 1 / sqrt(weights[-1])
            # 输出层
            # (1024, 4)
            W = tf.Variable(
                tf.random_uniform(
                    [weights[-1], num_of_classes], minval=-stdv, maxval=stdv
                ),
                dtype="float32",
                name="W",
            )
            # (4, )
            b = tf.Variable(
                tf.random_uniform(shape=[num_of_classes], minval=-stdv, maxval=stdv),
                name="b",
            )
            # (128, 4)
            self.p_y_given_x = tf.nn.xw_plus_b(x, W, b, name="scores")
            # 返回向量中的最大值的索引号
            self.predictions = tf.argmax(self.p_y_given_x, 1)

        with tf.name_scope("loss"):
            # 交叉熵作为损失函数
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.p_y_given_x, labels=self.input_y
            )
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("Accuracy"):
            # 计算准确率
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy"
            )
