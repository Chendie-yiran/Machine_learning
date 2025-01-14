from medmnist import BreastMNIST
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# Load the dataset 
training_data = BreastMNIST(split='train', download=True,size=28)
validation_data = BreastMNIST(split='val', download=True,size=28)
test_data = BreastMNIST(split='test', download=True,size=28)

# Define a function to get the data from the dataset object
def get_data(data):
    X = []
    y = []

    for i in range(len(data)):
        x, label = data[i]
        x = np.array(x)
        X.append(x)
        y.append(label[0])

    # Convert X and y to numpy arrays before returning
    X = np.array(X)
    y = np.array(y)

    # Flatten X to be 2D
    X = X.reshape(X.shape[0], -1)

    return X, y

def get_result(pred,actual, name):
    accuracy = accuracy_score(actual, pred)
    confusion = confusion_matrix(actual, pred)
    
    print(name," Accuracy: ", accuracy)
    #disp = ConfusionMatrixDisplay(confusion)
    #disp.plot()
    #plt.show()
    print(name," Confusion Matrix: \n", confusion)
    

# Get the data from the dataset objects
X_train, y_train = get_data(training_data)
X_val, y_val = get_data(validation_data)
X_combined = np.vstack((X_train, X_val))
y_combined = np.concatenate((y_train, y_val))

X_test, y_test = get_data(test_data)

#logistic regression
"""
LR = LogisticRegression(C=0.001,max_iter=5000, random_state=42,solver='lbfgs')
scores = cross_val_score(LR, X_combined, y_combined, cv=10)
print("Logistic Regression Average cross-validation accuracy: ", np.mean(scores))
LR.fit(X_train, y_train)
threshold = 0.4
y_train_prob = LR.predict_proba(X_train)[:, 1]  # Probabilities for the positive class (class 1)
y_val_prob = LR.predict_proba(X_val)[:, 1]
y_test_prob = LR.predict_proba(X_test)[:, 1]

y_train_pred = (y_train_prob >= threshold).astype(int)
y_val_pred = (y_val_prob >= threshold).astype(int)
y_test_pred = (y_test_prob >= threshold).astype(int)

get_result(y_train_pred, y_train, "Training")
get_result(y_val_pred, y_val, "Validation")
get_result(y_test_pred, y_test, "Testing") 
"""
###SVM

"""
SVM = SVC(kernel='rbf', random_state=42, probability=True, max_iter=5000, C=10)

# Train the SVM model on the training data
SVM.fit(X_train, y_train)

threshold = 0.4

# Get the predicted probabilities for the positive class (class 1)
y_train_prob = SVM.predict_proba(X_train)[:, 1]
y_val_prob = SVM.predict_proba(X_val)[:, 1]
y_test_prob = SVM.predict_proba(X_test)[:, 1]

# Apply the threshold to convert probabilities into binary predictions
y_train_pred = (y_train_prob >= threshold).astype(int)
y_val_pred = (y_val_prob >= threshold).astype(int)
y_test_pred = (y_test_prob >= threshold).astype(int)

# Get and print the result for training and validation sets
get_result(y_train_pred, y_train, "Training")
get_result(y_val_pred, y_val, "Validation")
get_result(y_test_pred, y_test, "Testing")
"""

#neural network
MLP = MLPClassifier(hidden_layer_sizes=(500,100),  
    activation='logistic',            
    solver='adam',  
    learning_rate_init=0.01,              
    alpha=0.0001,                
    learning_rate='adaptive',     
    max_iter=5000,                
    random_state=42)	
scores = cross_val_score(MLP, X_combined, y_combined, cv=10)



print("Neural network Average cross-validation accuracy: ", np.mean(scores))

# Fit the MLP model on the training data
MLP.fit(X_train, y_train)

# Get predicted probabilities for training, validation, and testing sets
y_train_prob = MLP.predict_proba(X_train)[:, 1]  # Probabilities for the positive class
y_val_prob = MLP.predict_proba(X_val)[:, 1]
y_test_prob = MLP.predict_proba(X_test)[:, 1]

# Set threshold and convert probabilities to predicted labels
threshold = 0.4
y_train_pred = (y_train_prob >= threshold).astype(int)
y_val_pred = (y_val_prob >= threshold).astype(int)
y_test_pred = (y_test_prob >= threshold).astype(int)

# Get and print the result for training, validation, and testing sets
get_result(y_train_pred, y_train, "Training")
get_result(y_val_pred, y_val, "Validation")
get_result(y_test_pred, y_test, "Testing")


