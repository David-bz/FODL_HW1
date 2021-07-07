import torch
import sklearn
import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as tvtf
from config import HW1_Dataset
import json


class Part_1():
    def __init__(self):
        self.ds = HW1_Dataset()
        self.ds.transform_for_baseline()
        self.linear_svm = sklearn.svm.SVC(kernel='linear')
        self.rbf_svm = sklearn.svm.SVC(kernel='rbf')

    def evaluate_baseline_model(self, model):
        result_dict = {'Name' : 'SVM', 'Kernel' : model.kernel}
        model.fit(self.ds.X_train, self.ds.y_train)
        train_pred = model.predict(self.ds.X_train)
        test_pred = model.predict(self.ds.X_test)
        result_dict['training_accuracy'] = sklearn.metrics.accuracy_score(train_pred, self.ds.y_train)
        result_dict['testing_accuracy'] = sklearn.metrics.accuracy_score(test_pred, self.ds.y_test)
        with open('./results/svm_' + model.kernel + '.json', "w+") as f:
            json.dump(result_dict, f)
        print(result_dict)
        return result_dict


if __name__ == '__main__':
    baseline = Part_1()
    baseline.evaluate_baseline_model(baseline.linear_svm)
    baseline.evaluate_baseline_model(baseline.rbf_svm)