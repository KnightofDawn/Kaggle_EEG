import tensorflow as tf
from eeg_dataset import EEG_Dataset
import numpy as np
from model import cnn

BATCH_SIZE = 128
BATCH_LAG = 32
NUM_CHANNELS = 32

def log_training(batch_index, valid_loss, valid_auc=None):
    """
    Logs the validation accuracy and loss to the terminal
    """
    print('Batch {}'.format(batch_index))
    if valid_auc != None:
        print('\tCross entropy validation loss: {}'.format(valid_loss))
        print('\tAccuracy: {}'.format(valid_auc))
    else:
        print('\tMean squared error loss: {}'.format(valid_loss))


def main():
    global BATCH_SIZE, BATCH_LAG, NUM_CHANNELS
    
    features = tf.placeholder(tf.float32, shape=[None, NUM_CHANNELS, BATCH_LAG, 1], name="features")
    labels = tf.placeholder(tf.float32, shape=[None, 6], name="labels")

    logits = cnn(features)
    
    auc = tf.metrics.auc(labels=labels, predictions=logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    init = tf.global_variables_initializer()
   

    with tf.Session() as sess:
        print("in main function")
        sess.run(tf.local_variables_initializer())
        #writer = tf.summary.FileWriter('logs', sess.graph)
        print("local variables initialized")
        sess.run(tf.global_variables_initializer())
        print("global variables initialized")

        #saver, save_path = utils.restore(sess, get('cnn.checkpoint'))
        eeg_data = EEG_Dataset(batch_size=BATCH_SIZE, batch_lag=BATCH_LAG)
        print("Calling train cnn")
        
        #train_cnn(features, labels, loss, train_op, auc, eeg_data)
        for subj in range(1,13):
            eeg_data.load_training_data(sub=subj)
            for batch_index in range(0, eeg_data.get_total_batches()):
                #print("Running Batch: ", batch_index)
                # Run one step of training
                batch_features, batch_labels = eeg_data.get_batch()

                #print("-----DEBUG-----")
                #print(batch_features.shape)
                #print(batch_labels.shape)
                sess.run(train_op, feed_dict={features: list(batch_features), 
                    labels: list(batch_labels)})

                if batch_index % 100 == 0:
                    batch_auc, batch_loss = sess.run([auc, loss], feed_dict={features : batch_features, labels : batch_labels})
                    log_training(batch_index, batch_loss, batch_auc)

        # test
        eeg_data.load_testing_data()
        test_auc, test_loss = sess.run([auc, loss], feed_dict={features : test_features, labels : test_labels})
        print('----------TESTING DATA--------------')
        print('\tCross entropy validation loss: {}'.format(valid_loss))
        print('\tAccuracy: {}'.format(valid_acc))


        #print('saving trained model...\n')
        #saver.save(sess, save_path)
        #writer.close()
        #utils.hold_training_plot()




if __name__ == '__main__':
    main()