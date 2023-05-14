import sys
sys.path.append('../')

from imports import *


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

        # predict the classes of the training data
        pred_train = knn.predict(self.x_train)

        # predict the classes of the validation data
        pred_val = knn.predict(self.x_val)

        self.save_model(knn, "knn.pkl")

        return knn, pred_train, pred_val

    def ANN(self, input_dim, output_dim, hidden_layers=[500, 400]):

        print(f'input_dim: {input_dim}')
        print(f'output_dim: {output_dim}')
        print(f'x_train: {self.x_train.shape}')
        print(f'y_train: {self.y_train.shape}')


        # create sequential model
        model = Sequential()
        
        # add input layer
        model.add(Dense(hidden_layers[0], activation="relu", input_dim=input_dim,
                        kernel_regularizer=regularizers.l2(0.01)))



        # add hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation="relu"))
            
        # add output layer
        model.add(Dense(output_dim, activation="softmax"))

        y_onehot_train = to_categorical(self.y_train, num_classes=6)
        y_onehot_val = to_categorical(self.y_val, num_classes=6)
        print(f'y_onehot: {y_onehot_train.shape}')
        
        # compile the model
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        
        # model.summary()
        # print(f'Output shape: {model.output_shape}')


        # train the model using the training data
        model.fit(
            self.x_train,
            y_onehot_train,
            epochs=10,
            batch_size=32,
            validation_data=(self.x_val, y_onehot_val),
        )

        # Predict the classes of the training data
        pred_train = model.predict(self.x_train)
        pred_train = pred_train.argmax(axis=1)

        # Predict the classes of the validation data
        pred_val = model.predict(self.x_val)
        pred_val = pred_val.argmax(axis=1)

        # Evaluate the model on the test data
        loss, test_accuracy = model.evaluate(self.x_val, y_onehot_val, verbose=0)
        print(f"Test accuracy: {test_accuracy}")
        print(f"Test loss: {loss}")
        
        model.save("ann.h5")

        return model, pred_train, pred_val

    def SVM(self, kernel="linear", C=1.0):
        # create SVM classifier
        svm = SVC(kernel=kernel, C=C)
        
        # train the model using the training data
        svm.fit(self.x_train, self.y_train)

        # predict the classes of the training data
        pred_train = svm.predict(self.x_train)

        # predict the classes of the validation data
        pred_val = svm.predict(self.x_val)

        self.save_model(svm, "svm.pkl")

        return svm, pred_train, pred_val

    def HMM(self, n_components=2):
        # create HMM model
        model = hmm.GaussianHMM(n_components=n_components)
        
        # train the model using the training data
        model.fit(self.x_train)

        # predict the classes of the training data
        log_likelihood, pred_train = model.decode(self.x_train)

        # predict the classes of the validation data
        log_likelihood, pred_val = model.decode(self.x_val)

        self.save_model(model, "hmm.pkl")

        return model, pred_train, pred_val

    def Ensemble(self):
        # create Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # train the models using the training data
        rf.fit(self.x_train, self.y_train)

        # predict the classes of the training data
        pred_train = rf.predict(self.x_train)

        # predict the classes of the validation data using each classifier
        pred_val = rf.predict(self.x_val)

        self.save_model(rf, "random_forest.pkl")

        return rf, pred_train, pred_val

    def AdaBoost(self):
        # Initialize the classifier
        clf = AdaBoostClassifier()

        # Fit the classifier on training data
        clf.fit(self.x_train, self.y_train)

        # Predict on training data
        pred_train = clf.predict(self.x_train)

        # Predict on validation data
        pred_val = clf.predict(self.x_val)

        self.save_model(clf, "adaboost.pkl")

        return clf, pred_train, pred_val

    def save_model(self, model, model_name):
        pickle.dump(model, open(model_name, "wb"))
