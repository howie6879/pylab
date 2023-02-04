#!/usr/bin/env python
"""
 Created by howie.hu at 2018/9/21.
"""

import numpy as np


class Perceptron:
    """
    李航老师统计学习方法第二章感知机例2.2对偶形式代码实现
    """

    def __init__(self, alpha_length=3):
        self.alpha = np.zeros(alpha_length)
        # 权重
        self.w = np.zeros(2)
        # 偏置项
        self.b = 0.0

    def fit(self, input_vectors, labels, learn_nums=7):
        """
        训练出合适的 w 和 b
        :param input_vectors: 样本训练数据集
        :param labels: 标记值
        :param learn_nums: 学习多少次
        """
        gram = np.matmul(input_vectors, input_vectors.T)

        for i in range(learn_nums):

            for input_vector_index, input_vector in enumerate(input_vectors):
                label = labels[input_vector_index]
                delta = 0.0
                for alpha_index, a in enumerate(self.alpha):
                    delta += (
                        a * labels[alpha_index] * gram[input_vector_index][alpha_index]
                    )
                delta = label * delta + self.b
                if delta <= 0:
                    self.alpha[input_vector_index] += 1
                    self.b += label
                    break
        self.w = sum(
            [j * input_vectors[i] * labels[i] for i, j in enumerate(self.alpha)]
        )
        print("最终结果：此时感知器权重为{0}，偏置项为{1}".format(self.w, self.b))
        return self

    def predict(self, input_vector):
        if isinstance(input_vector, list):
            input_vector = np.array(input_vector)
        y = sum(self.w * input_vector) + self.b
        return 1 if y > 0 else -1


if __name__ == "__main__":
    input_vectors = np.array([[3, 3], [4, 3], [1, 1]])
    labels = np.array([1, 1, -1])
    p = Perceptron()
    model = p.fit(input_vectors, labels)
    print(model.predict([3, 3]))
    print(model.predict([4, 3]))
    print(model.predict([1, 1]))
