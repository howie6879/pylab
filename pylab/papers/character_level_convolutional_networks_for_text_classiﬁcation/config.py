class TrainingConfig(object):
    p = 0.9
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 5000
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    # 卷积层定义
    conv_layers = [
        [256, 7, 3],
        [256, 7, 3],
        [256, 3, None],
        [256, 3, None],
        [256, 3, None],
        [256, 3, 3],
    ]
    # 全连接层
    fully_connected_layers = [1024, 1024]
    # 阈值
    th = 1e-6


class Config(object):
    # 字母表
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    input_size = 1014
    batch_size = 128
    num_of_classes = 4

    train_data_source = "ag_news_csv/train.csv"
    dev_data_source = "ag_news_csv/test.csv"

    training = TrainingConfig()
    model = ModelConfig()


config = Config()
