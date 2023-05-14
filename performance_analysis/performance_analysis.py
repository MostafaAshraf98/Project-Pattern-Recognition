import sys
sys.path.append('../')

from imports import *


class PerformanceAnalysis:
    def __init__(self, modelName, predictions, true_labels, validation=False):
        self.modelName = modelName
        self.predictions = predictions
        self.true_labels = true_labels
        self.validation = validation

        # Weights
        self.weights = np.zeros(6)
        for i in range(6):
            self.weights[i] = np.sum(self.true_labels == i) / len(self.true_labels)

        
        # Performance Metrics

        self.accuracy = 0
        
        ## Confusion Matrix Parameters
        self.true_positives = np.zeros(6)
        self.false_positives = np.zeros(6)
        self.true_negatives = np.zeros(6)
        self.false_negatives = np.zeros(6)
        self.confusion_matrix_computed = False
        
        ## Micro Average
        self.micro_avg_precision = 0
        self.micro_avg_recall = 0
        self.micro_avg_f1 = 0
        self.total_TP = 0
        self.total_FP = 0
        self.total_TN = 0
        self.total_FN = 0
        
        ## Macro Average
        self.macro_avg_precision = 0
        self.macro_avg_recall = 0
        self.macro_avg_f1 = 0
        self.avg_TP = 0
        self.avg_FP = 0
        self.avg_TN = 0
        self.avg_FN = 0
        
        ## Weighted Macro Average
        self.weighted_macro_avg_precision = 0
        self.weighted_macro_avg_recall = 0
        self.weighted_macro_avg_f1 = 0
        self.weighted_avg_TP = 0
        self.weighted_avg_FP = 0
        self.weighted_avg_TN = 0
        self.weighted_avg_FN = 0
        
    def calculate_performance_metrics(self):
        # Calculate the confusion matrix
        self.__confusion_matrix()
        
        # Write them to a file
        with open('performance_metrics.txt', 'w') as f:
            f.write('========================================\n')
            f.write(f'Timestamp: {str(datetime.datetime.now())}\n')
            f.write(f'Model: {self.modelName}\n')
            f.write(f'Accuracy: {np.round(self.accuracy * 100, 2)}%\n\n')
            f.write(f'Micro Average Precision: {str(self.micro_avg_precision)}\n')
            f.write(f'Micro Average Recall: {str(self.micro_avg_recall)}\n')
            f.write(f'Micro Average F1: {str(np.round(self.micro_avg_f1,2))}\n\n')
            f.write(f'Macro Average Precision: {str(self.macro_avg_precision)}\n')
            f.write(f'Macro Average Recall: {str(self.macro_avg_recall)}\n')
            f.write(f'Macro Average F1: {str(self.macro_avg_f1)}\n\n')
            f.write(f'Weighted Macro Average Precision: {str(self.weighted_macro_avg_precision)}\n')
            f.write(f'Weighted Macro Average Recall: {str(self.weighted_macro_avg_recall)}\n')
            f.write(f'Weighted Macro Average F1: {str(self.weighted_macro_avg_f1)}\n')
            f.write('========================================\n\n')
        
        # Print them to the console
        if (not self.validation):
            print('========================================')
            # print(f'Timestamp: {str(datetime.datetime.now())}')
            print(f'Model: {self.modelName}')
            print(f'Accuracy: {np.round(self.accuracy * 100, 2)}%')
            # print(f'Micro Average Precision: {str(self.micro_avg_precision)}')
            # print(f'Micro Average Recall: {str(self.micro_avg_recall)}')
            # print(f'Micro Average F1: {str(np.round(self.micro_avg_f1,2))}\n')
            # print(f'Macro Average Precision: {str(self.macro_avg_precision)}')
            # print(f'Macro Average Recall: {str(self.macro_avg_recall)}')
            # print(f'Macro Average F1: {str(self.macro_avg_f1)}\n')
            # print(f'Weighted Macro Average Precision: {str(self.weighted_macro_avg_precision)}')
            # print(f'Weighted Macro Average Recall: {str(self.weighted_macro_avg_recall)}')
            # print(f'Weighted Macro Average F1: {str(self.weighted_macro_avg_f1)}')
            print('========================================\n')

        else:
            print(Fore.RED)            
            print('*****************************************')
            # print(f'Timestamp: {str(datetime.datetime.now())}')
            print(f'Model: {self.modelName}')
            print(f'Accuracy: {np.round(self.accuracy * 100, 2)}%')
            # print(f'Micro Average Precision: {str(self.micro_avg_precision)}')
            # print(f'Micro Average Recall: {str(self.micro_avg_recall)}')
            # print(f'Micro Average F1: {str(np.round(self.micro_avg_f1,2))}\n')
            # print(f'Macro Average Precision: {str(self.macro_avg_precision)}')
            # print(f'Macro Average Recall: {str(self.macro_avg_recall)}')
            # print(f'Macro Average F1: {str(self.macro_avg_f1)}\n')
            # print(f'Weighted Macro Average Precision: {str(self.weighted_macro_avg_precision)}')
            # print(f'Weighted Macro Average Recall: {str(self.weighted_macro_avg_recall)}')
            # print(f'Weighted Macro Average F1: {str(self.weighted_macro_avg_f1)}')
            print('*****************************************\n')
            print(Style.RESET_ALL)

        
  
    def __confusion_matrix(self):
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
    
        self.__accuracy()
        self.__micro_avg_precision()
        self.__micro_avg_recall()
        self.__micro_avg_f1()
        self.__macro_avg_precision()
        self.__macro_avg_recall()
        self.__macro_avg_f1()
        self.__weighted_macro_avg_precision()
        self.__weighted_macro_avg_recall()
        self.__weighted_macro_avg_f1()
    




    def __accuracy(self):
        self.accuracy = np.sum(self.predictions == self.true_labels.flatten()) / len(self.true_labels.flatten())
        return self.accuracy



    #========================================MICRO AVERAGE========================================

    def __micro_avg_precision(self):
        self.micro_avg_precision = self.total_TP / (self.total_TP + self.total_FP)
        # print('Micro Average Precision: ' + str(self.micro_avg_precision_))
        return self.micro_avg_precision

    def __micro_avg_recall(self):
        self.micro_avg_recall =  self.total_TP / (self.total_TP + self.total_FN)
        # print('Micro Average Recall: ' + str(self.micro_avg_recall_))
        return self.micro_avg_recall
    
    def __micro_avg_f1(self):
        self.micro_avg_f1 = 2 * (self.micro_avg_precision * self.micro_avg_recall) / (self.micro_avg_precision + self.micro_avg_recall)
        return self.micro_avg_f1
        
    
    #========================================MACRO AVERAGE========================================

    def __macro_avg_precision(self):
        temp_arr = self.true_positives / (self.true_positives + self.false_positives)
        self.macro_avg_precision = np.mean(temp_arr)
        return self.macro_avg_precision

    def __macro_avg_recall(self):
        temp_arr = self.true_positives / (self.true_positives + self.false_negatives)
        self.macro_avg_recall = np.mean(temp_arr)
        return self.macro_avg_recall

    def __macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        self.macro_avg_f1 = np.mean(2 * (precisions * recalls) / (precisions + recalls))
        return self.macro_avg_f1
    
    #========================================WEIGHTED MACRO AVERAGE========================================

    def __weighted_macro_avg_precision(self):
        self.weighted_macro_avg_precision = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_positives)))
        return self.weighted_macro_avg_precision
        
    def __weighted_macro_avg_recall(self):
        self.weighted_macro_avg_recall = np.sum(self.weights * (self.true_positives / (self.true_positives + self.false_negatives)))
        return self.weighted_macro_avg_recall
        
    def __weighted_macro_avg_f1(self):
        precisions = self.true_positives / (self.true_positives + self.false_negatives)
        recalls = self.true_positives / (self.true_positives + self.false_positives)
        self.weighted_macro_avg_f1 = np.sum(self.weights * (2 * (precisions * recalls) / (precisions + recalls)))
        return self.weighted_macro_avg_f1