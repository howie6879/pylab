import csv
import re

import numpy as np


class DataUtils(object):
    """
    此类用于加载原始数据
    """

    def __init__(
        self,
        data_source: str,
        *,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        batch_size=128,
        input_size: int = 1014,
        num_of_classes: int = 4
    ):
        """
        数据初始化
        :param data_source: 原始数据路径
        :param alphabet: 索引字母表
        :param input_size: 输入特征 相当于论文中说的l0
        :param num_of_classes: 据类别
        """

        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.batch_size = batch_size
        self.data_source = data_source
        self.length = input_size
        self.num_of_classes = num_of_classes

        # 将每个字符映射成int
        self.char_dict = {}
        for idx, char in enumerate(self.alphabet):
            self.char_dict[char] = idx + 1

    def get_batch_to_indices(self, batch_num=0):
        """
        返回随机样本
        :param batch_num: 每次随机分配的样本数目
        :return: (data, classes)
        """
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        # 最长不超过数据集本身大小
        end_index = (
            data_size
            if self.batch_size == 0
            else min((batch_num + 1) * self.batch_size, data_size)
        )
        batch_texts = self.shuffled_data[start_index:end_index]
        # 类别 one hot 编码
        one_hot = np.eye(self.num_of_classes, dtype="int64")

        batch_indices, classes = [], []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            # 类别数字减一就是 one hot 编码的index
            classes.append(one_hot[int(c) - 1])

        return np.asarray(batch_indices, dtype="int64"), classes

    def get_length(self):
        """
        返回数据集长度
        :return:
        """
        return len(self.data)

    def load_data(self):
        """
        从文件加载原始数据
        Returns: None
        """
        data = []
        with open(self.data_source, "r", encoding="utf-8") as f:
            rdr = csv.reader(f, delimiter=",", quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = (
                        txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                    )
                data.append((int(row[0]), txt))
        self.data = np.array(data)
        self.shuffled_data = self.data
        print("Data loaded from " + self.data_source)

    def shuffle_data(self):
        """
        将数据集打乱
        :return:
        """
        data_size = len(self.data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def str_to_indexes(self, s):
        """
        根据字符字典对数据进行转化
        :param s: 即将转化的字符
        :return: numpy.ndarray 长度为：self.length
        """
        # 论文中表明 对于比较大的数据可以考虑不用区分大小写
        s = s.lower()
        # 最大长度不超过 input_size 此处为 1014
        max_length = min(len(s), self.length)
        # 初始化数组
        str2idx = np.zeros(self.length, dtype="int64")
        for i in range(1, max_length + 1):
            # 逆序映射
            c = s[-i]
            if c in self.char_dict:
                str2idx[i - 1] = self.char_dict[c]
        return str2idx


if __name__ == "__main__":
    train_data_ins = DataUtils(data_source="./ag_news_csv/train.csv")
    train_data_ins.load_data()
    train_data_ins.shuffle_data()

    batch_indices, classes = train_data_ins.get_batch_to_indices()
    print(classes)

    exit()
    with open("test.vec", "w") as fo:
        for i in range(len(train_data_ins.data)):
            # 类别
            c = train_data_ins.data[i][0]
            # 文本
            txt = train_data_ins.data[i][1]
            # 生成向量
            vec = ",".join(map(str, train_data_ins.str_to_indexes(txt)))

            fo.write("{}\t{}\n".format(c, vec))
