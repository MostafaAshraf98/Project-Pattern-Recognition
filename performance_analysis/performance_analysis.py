import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import hmmlearn.hmm as hmm
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

class PerformanceAnalysis:
    def __init__(self, modelName, predictions, true_labels):
        self.modelName = modelName
        self.predictions = predictions
        self.true_labels = true_labels

        # Weights
        self.weights = np.zeros(6)
        for i in range(6):
            self.weights[i] = np.sum(self.true_labels == i) / len(self.true_labels)

        
        # Performance Metrics
        
        ## Confusion Matrix Parameters
        self.true_positives = np.zeros(6)
        self.false_positives = np.zeros(6)
        self.true_negatives = np.zeros(6)
        self.false_negatives = np.zeros(6)
        self.confusion_matrix_computed = False
        
        ## Micro Average
        self.micro_avg_precision_ = 0
        self.micro_avg_recall_ = 0
        self.micro_avg_f1_ = 0
        self.total_TP = 0
        self.total_FP = 0
        self.total_TN = 0
        self.total_FN = 0
        
        ## Macro Average
        self.macro_avg_precision_ = 0
        self.macro_avg_recall_ = 0
        self.macro_avg_f1_ = 0
        self.avg_TP = 0
        self.avg_FP = 0
        self.avg_TN = 0
        self.avg_FN = 0
        
        ## Weighted Macro Average
        self.weighted_macro_avg_precision_ = 0
        self.weighted_macro_avg_recall_ = 0
        self.weighted_macro_avg_f1_ = 0
        self.weighted_avg_TP = 0
        self.weighted_avg_FP = 0
        self.weighted_avg_TN = 0
        self.weighted_avg_FN = 0
        
    def calculate_performance_metrics(self):
        # Calculate the confusion matrix
        self.confusion_matrix()
        
        # Write them to a file
        with open('performance_metrics.txt', 'a') as f:
            f.write('========================================\n')
            f.write('Timestamp: '+ str(datetime.datetime.now()) + '\n')
            f.write('Model: ' + self.modelName + '\n')
            f.write('Accuracy: ' + str(np.round(self.accuracy() * 100, 2)) + '%\n\n')
            f.write('Micro Average Precision: ' + str(self.micro_avg_precision()) + '\n')
            f.write('Micro Average Recall: ' + str(self.micro_avg_recall()) + '\n')
            f.write('Micro Average F1: ' + str(np.round(self.micro_avg_f1(),2)) + '\n\n')
            f.write('Macro Average Precision: ' + str(self.macro_avg_precision()) + '\n')
            f.write('Macro Average Recall: ' + str(self.macro_avg_recall()) + '\n')
            f.write('Macro Average F1: ' + str(self.macro_avg_f1()) + '\n\n')
            f.write('Weighted Macro Average Precision: ' + str(self.weighted_macro_avg_precision()) + '\n')
            f.write('Weighted Macro Average Recall: ' + str(self.weighted_macro_avg_recall()) + '\n')
            f.write('Weighted Macro Average F1: ' + str(self.weighted_macro_avg_f1()) + '\n')
            f.write('========================================\n\n')
        
        # Print them to the console
        print('========================================')
        print('Model: ' + self.modelName)
        print('Accuracy: ' + str(np.round(self.accuracy() * 100, 2)) + '%\n')
        print('Micro Average Precision: ' + str(self.micro_avg_precision()))
        print('Micro Average Recall: ' + str(self.micro_avg_recall()))
        print('Micro Average F1: ' + str(np.round(self.micro_avg_f1(),2)) + '\n\n')
        print('Macro Average Precision: ' + str(self.macro_avg_precision()))
        print('Macro Average Recall: ' + str(self.macro_avg_recall()))
        print('Macro Average F1: ' + str(self.macro_avg_f1()) + '\n')
        print('Weighted Macro Average Precision: ' + str(self.weighted_macro_avg_precision()))
        print('Weighted Macro Average Recall: ' + str(self.weighted_macro_avg_recall()))
        print('Weighted Macro Average F1: ' + str(self.weighted_macro_avg_f1()))
        print('========================================\n\n')


        
  
    def confusion_matrix(self):
        # print('Predictions shape: ' + str(self.predictions.shape))
        for i in range(6):
            self.true_positives[i] = np.sum((self.predictions == i) & (self.true_labels == i))
            self.false_positives[i] = np.sum((self.predictions == i) & (self.true_labels != i))
            self.true_negatives[i] = np.sum((self.predictions != i) & (self.true_labels != i))
            self.false_negatives[i] = np.sum((self.predictions != i) & (self.true_labels == i))
        self.total_TP = np.sum(self.true_positives)
        self.total_FP = np.sum(self.false_positives)
        self.total_TN = np.sum(self.true_negatives)
        self.total_FN = np.sum(self.false_negatives)
        self.avg_TP = np.mean(self.true_positives)
        self.avg_FP = np.mean(self.false_positives)
        self.avg_TN = np.mean(self.true_negatives)
        self.avg_FN = np.mean(self.false_negatives)
        self.confusion_matrix_computed = True
        # print('TP = ' + str(self.total_TP))
        # print('FP = ' + str(self.total_FP))
        # print('FN = ' + str(self.total_FN))




    def accuracy(self):
        return np.sum(self.predictions == self.true_labels) / len(self.true_labels)



    #========================================MICRO AVERAGE========================================

    def micro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            self.micro_avg_precision_ = self.total_TP / (self.total_TP + self.total_FP)
            # print('Micro Average Precision: ' + str(self.micro_avg_precision_))
            return self.micro_avg_precision_

    def micro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            self.micro_avg_recall_ =  self.total_TP / (self.total_TP + self.total_FN)
            # print('Micro Average Recall: ' + str(self.micro_avg_recall_))
            return self.micro_avg_recall_
    
    def micro_avg_f1(self):
        return 2 * (self.micro_avg_precision_ * self.micro_avg_recall_) / (self.micro_avg_precision_ + self.micro_avg_recall_)
        
    
    #========================================MACRO AVERAGE========================================

    def macro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            temp_arr = self.true_positives / (self.true_positives + self.false_positives)
            temp_arr = temp_arr[~np.isnan(temp_arr)]
            self.macro_avg_precision_ = np.mean(temp_arr)
            # print('True Positives: ' + str(self.true_positives))
            # print('False Positives: ' + str(self.false_positives))
            # print('Macro Average Precision: ' + str(self.macro_avg_precision_))
            return self.macro_avg_precision_

    def macro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            temp_arr = self.true_positives / (self.true_positives + self.false_negatives)
            temp_arr = temp_arr[~np.isnan(temp_arr)]
            self.macro_avg_recall_ = np.mean(temp_arr)
            return self.macro_avg_recall_

    def macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        self.macro_avg_f1_ = np.mean(2 * (precisions * recalls) / (precisions + recalls))
        return self.macro_avg_f1_
    
    #========================================WEIGHTED MACRO AVERAGE========================================

    def weighted_macro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            self.weighted_macro_avg_precision_ = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_positives)))
            return self.weighted_macro_avg_precision_
        
    def weighted_macro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            self.weighted_macro_avg_recall_ = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_negatives)))
            return self.weighted_macro_avg_recall_
        
    def weighted_macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        self.weighted_macro_avg_f1_ = np.sum(self.weights * (2 * (precisions * recalls) / (precisions + recalls)))
        return self.weighted_macro_avg_f1_