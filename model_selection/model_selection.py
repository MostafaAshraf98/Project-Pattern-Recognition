from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import hmmlearn.hmm as hmm
from sklearn.ensemble import AdaBoostClassifier

class ModelSelection:
    
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
    def KNN(self, n_neighbors=3):
        # create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        # train the model using the training data
        knn.fit(self.x_train, self.y_train)
        # predict the classes of the validation data
        y_pred = knn.predict(self.x_val)
        # calculate and return the accuracy score
        # accuracy_score(self.y_val, y_pred)
        # accuracy is done in performance_analysis.py (?)
        return knn, y_pred
    
    def ANN(self, input_dim, output_dim, hidden_layers=[100]):
        # create sequential model
        model = Sequential()
        # add input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        # add hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
        # add output layer
        model.add(Dense(output_dim, activation='softmax'))
        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # train the model using the training data
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, validation_data=(self.x_val, self.y_val))
        # evaluate the model using the validation data and return the accuracy score
        _, accuracy = model.evaluate(self.x_val, self.y_val)
        accuracy
        return model
    
    def SVM(self, kernel='linear', C=1.0):
        # create SVM classifier
        svm = SVC(kernel=kernel, C=C)
        # train the model using the training data
        svm.fit(self.x_train, self.y_train)
        # predict the classes of the validation data
        y_pred = svm.predict(self.x_val)
        # calculate and return the accuracy score
        accuracy_score(self.y_val, y_pred)
        return svm
    
    def HMM(self, n_components=2):
        # create HMM model
        model = hmm.GaussianHMM(n_components=n_components)
        # train the model using the training data
        model.fit(self.x_train)
        # predict the classes of the validation data
        log_likelihood, y_pred = model.decode(self.x_val)
        # calculate and return the accuracy score
        accuracy_score(self.y_val, y_pred)
        return model
        
    
    def Ensemble(self):
        # create Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # train the models using the training data
        rf.fit(self.x_train, self.y_train)
        # predict the classes of the validation data using each classifier
        y_pred_rf = rf.predict(self.x_val)
        return rf
        
    def AdaBoost(self):
        # Initialize the classifier
        clf = AdaBoostClassifier()
        
        # Fit the classifier on training data
        clf.fit(self.x_train, self.y_train)
        
        # Predict on validation data
        y_pred = clf.predict(self.x_val)
        
        return y_pred