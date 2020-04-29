import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pandas as pd
import tensorflow as tf


class tf_roc(object):
    def __init__(self, y_pred_prob, y_true, threshold_num, save_dir):
        '''
        file format: dataid,predict_score,label
        predict_score should be between 0 and 1
        label should be 0 or 1
        threshold_num: number of threshold will plot
        '''
        self.predicts = y_pred_prob.tolist()
        self.labels = y_true.tolist()
        self.total = y_true.shape[0]

        self.threshold_num = threshold_num
        self.trues = 0  # total of True labels
        self.fpr = []  # false positive
        self.tpr = []  # true positive
        self.ths = []  # thresholds
        self.save_dir = save_dir
        self.writer = tf.summary.FileWriter(self.save_dir)

    def calc(self):
        for label in self.labels:
            if label:
                self.trues += 1
        threshold_step = 1. / self.threshold_num
        for t in range(self.threshold_num + 1):
            th = 1 - threshold_step * t
            tn, tp, fp, fpr, tpr = self._calc_once(th)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.ths.append(th)
            self._save(fpr, tpr)
        print(self.fpr)
        print(self.tpr)
        print(self.ths)

    def _save(self, fpr, tpr):
        summt = tf.Summary()
        summt.value.add(tag="roc", simple_value=tpr)
        self.writer.add_summary(summt, fpr * 100)  # for tensorboard step drawable
        self.writer.flush()

    def _calc_once(self, t):
        esp = 1e-5
        fp = 0
        tp = 0
        tn = 0
        for i in range(self.total):
            if not self.labels[i]:
                if self.predicts[i] >= t:
                    fp += 1
                else:
                    tn += 1
            elif self.predicts[i] >= t:
                tp += 1
        # fpr = fp / float(fp + tn) #precision
        fpr = fp / float(fp + tp + esp)  # detection
        tpr = tp / float(self.trues)
        return tn, tp, fp, fpr, tpr


def classification_reports(y_true, y_pred):
    print("classification_report(left:labels):")
    print(classification_report(y_true=y_true, y_pred=y_pred))


def confusion_matrixs(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    print("confusion _matrixs(left labels:y_true,up labels:y_pred):")
    print(conf_mat)


def roc_auc_scores(y_true, y_pred_prob):
    return roc_auc_score(y_true=y_true, y_score=y_pred_prob)


def classify_metric_message(predict_label_file, name=None):
    """
    :param predict_label_file:should have type:predict_label,predict_prob,true_label
    :param name:
    :return:
    """
    csvimagedata = pd.read_csv(predict_label_file)
    data = csvimagedata.iloc[:, :].values
    predict_labels = data[:, 0]
    predict_probs = data[:, 1]
    true_labels = data[:, 2]

    if name == "roc_auc_score":
        print(roc_auc_scores(true_labels, predict_probs))
    elif name == "confusion_matric":
        confusion_matrixs(true_labels, predict_labels)
    elif name == "main_classify":
        classification_reports(true_labels, predict_labels)
    elif name == "roc_curve":
        threshold_num = 2000
        save_dir = "log"
        roc = tf_roc(predict_probs, true_labels, int(threshold_num), save_dir)
        roc.calc()


if __name__ == '__main__':
    predict_label_file = "classify_metrics.csv"
    classify_metric_message(predict_label_file, name="roc_curve")
