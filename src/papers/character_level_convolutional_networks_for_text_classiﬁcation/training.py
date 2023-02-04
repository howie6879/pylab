import datetime
import os
import time

import tensorflow as tf

from pylab.papers.character_level_convolutional_networks_for_text_classiﬁcation.config import (
    config,
)
from pylab.papers.character_level_convolutional_networks_for_text_classiﬁcation.char_cnn import (
    CharConvNet,
)
from pylab.papers.character_level_convolutional_networks_for_text_classiﬁcation.data_utils import (
    DataUtils,
)

if __name__ == "__main__":
    # 加载数据集
    train_data = DataUtils(
        data_source=config.train_data_source,
        alphabet=config.alphabet,
        input_size=config.input_size,
        batch_size=config.batch_size,
        num_of_classes=config.num_of_classes,
    )
    train_data.load_data()
    dev_data = DataUtils(
        data_source=config.dev_data_source,
        alphabet=config.alphabet,
        input_size=config.input_size,
        batch_size=config.batch_size,
        num_of_classes=config.num_of_classes,
    )
    dev_data.load_data()

    # 每次迭代数据
    num_batches_per_epoch = int(train_data.get_length() / config.batch_size) + 1

    num_batch_dev = dev_data.get_length()

    print("Training ===>")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False
        )

        sess = tf.Session(config=session_conf)

        with sess.as_default():

            char_cnn = CharConvNet(
                conv_layers=config.model.conv_layers,
                fully_layers=config.model.fully_connected_layers,
                input_size=config.input_size,
                alphabet_size=config.alphabet_size,
                num_of_classes=config.num_of_classes,
                th=config.model.th,
            )

            global_step = tf.Variable(0, trainable=False)
            learning_rate = 0.001

            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(char_cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step
            )

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g
                    )
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g)
                    )
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", char_cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", char_cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged]
            )
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    char_cnn.input_x: x_batch,
                    char_cnn.input_y: y_batch,
                    char_cnn.dropout_keep_prob: config.training.p,
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [
                        train_op,
                        global_step,
                        train_summary_op,
                        char_cnn.loss,
                        char_cnn.accuracy,
                    ],
                    feed_dict,
                )
                time_str = datetime.datetime.now().isoformat()
                print(
                    "{}: step {}, loss {:g}, acc {:g}".format(
                        time_str, step, loss, accuracy
                    )
                )
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    char_cnn.input_x: x_batch,
                    char_cnn.input_y: y_batch,
                    char_cnn.dropout_keep_prob: 1.0,  # Disable dropout
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, char_cnn.loss, char_cnn.accuracy],
                    feed_dict,
                )
                time_str = datetime.datetime.now().isoformat()
                print(
                    "{}: step {}, loss {:g}, acc {:g}".format(
                        time_str, step, loss, accuracy
                    )
                )
                if writer:
                    writer.add_summary(summaries, step)

            for e in range(config.training.epoches):

                train_data.shuffle_data()
                for k in range(num_batches_per_epoch):

                    batch_x, batch_y = train_data.get_batch_to_indices(k)
                    train_step(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % config.training.evaluate_every == 0:
                        xin, yin = dev_data.get_batch_to_indices()
                        print("\nEvaluation:")
                        dev_step(xin, yin, writer=dev_summary_writer)
                        print("")

                    if current_step % config.training.checkpoint_every == 0:
                        path = saver.save(
                            sess, checkpoint_prefix, global_step=current_step
                        )
                        print("Saved model checkpoint to {}\n".format(path))
