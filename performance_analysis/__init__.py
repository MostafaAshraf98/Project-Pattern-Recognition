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
    def __init__(self, model, modelName, y_train, x_test, y_test):
        self.model = model
        self.modelName = modelName
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.predictions = []

        # Weights
        self.weights = np.zeros(6)
        for i in range(6):
            self.weights[i] = np.sum(self.y_train == i) / len(self.y_train)

        
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
            f.write('Model: ' + self.modelName + '\n')
            f.write('Accuracy: ' + str(np.round(self.accuracy() * 100, 2)) + '%\n\n')
            f.write('Micro Average Precision: ' + str(self.micro_avg_precision()) + '\n')
            f.write('Micro Average Recall: ' + str(self.micro_avg_recall()) + '\n')
            f.write('Micro Average F1: ' + str(self.micro_avg_f1()) + '\n\n')
            f.write('Macro Average Precision: ' + str(self.macro_avg_precision()) + '\n')
            f.write('Macro Average Recall: ' + str(self.macro_avg_recall()) + '\n')
            f.write('Macro Average F1: ' + str(self.macro_avg_f1()) + '\n\n')
            f.write('Weighted Macro Average Precision: ' + str(self.weighted_macro_avg_precision()) + '\n')
            f.write('Weighted Macro Average Recall: ' + str(self.weighted_macro_avg_recall()) + '\n')
            f.write('Weighted Macro Average F1: ' + str(self.weighted_macro_avg_f1()) + '\n')
            f.write('========================================\n\n')

        
  
    def confusion_matrix(self):
        self.predictions = self.model.predict(self.x_val)
        for i in range(6):
            self.true_positives[i] = np.sum((self.predictions == i) & (self.y_val == i))
            self.false_positives[i] = np.sum((self.predictions == i) & (self.y_val != i))
            self.true_negatives[i] = np.sum((self.predictions != i) & (self.y_val != i))
            self.false_negatives[i] = np.sum((self.predictions != i) & (self.y_val == i))
        self.total_TP = np.sum(self.true_positives)
        self.total_FP = np.sum(self.false_positives)
        self.total_TN = np.sum(self.true_negatives)
        self.total_FN = np.sum(self.false_negatives)
        self.avg_TP = np.mean(self.true_positives)
        self.avg_FP = np.mean(self.false_positives)
        self.avg_TN = np.mean(self.true_negatives)
        self.avg_FN = np.mean(self.false_negatives)
        self.confusion_matrix_computed = True



    def accuracy(self):
        return np.sum(self.predictions == self.y_test) / len(self.y_test)



    #========================================MICRO AVERAGE========================================

    def micro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            micro_avg_precision_ = self.total_TP / (self.total_TP + self.total_FP)
            return micro_avg_precision_

    def micro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            micro_avg_recall_ =  self.total_TP / (self.total_TP + self.total_FN)
            return micro_avg_recall_
    
    def micro_avg_f1(self):
        return 2 * (self.micro_avg_precision_ * self.micro_avg_recall_) / (self.micro_avg_precision_ + self.micro_avg_recall_)
        
    
    #========================================MACRO AVERAGE========================================

    def macro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            macro_avg_precision_ = np.mean(self.true_positives / (self.true_positives + self.false_positives))
            return macro_avg_precision_

    def macro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            macro_avg_recall_ = np.mean(self.true_positives / (self.true_positives + self.false_negatives))
            return macro_avg_recall_

    def macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        macro_avg_f1_ = np.mean(2 * (precisions * recalls) / (precisions + recalls))
        return macro_avg_f1_
    
    #========================================WEIGHTED MACRO AVERAGE========================================

    def weighted_macro_avg_precision(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            weighted_macro_avg_precision_ = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_positives)))
            return weighted_macro_avg_precision_
        
    def weighted_macro_avg_recall(self):
        if (self.confusion_matrix_computed == False):
            self.confusion_matrix()
        else:
            weighted_macro_avg_recall_ = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_negatives)))
            return weighted_macro_avg_recall_
        
    def weighted_macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        weighted_macro_avg_f1_ = np.sum(self.weights * (2 * (precisions * recalls) / (precisions + recalls)))
        return weighted_macro_avg_f1_