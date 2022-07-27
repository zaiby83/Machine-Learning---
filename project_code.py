
# Importing Libraries
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("train.csv") # Importing the file

data.head() # To view details of the top entries

data.apply(lambda x: sum(x.isnull()),axis=0) # Checking missing values in each column of this dataset

data.Gender = data.Gender.fillna('Male') # Filling the NULL values in gender as 'Male'

data.Married = data.Married.fillna('Yes') # Filling the NULL values in Married as 'Yes'

data.Dependents = data.Dependents.fillna('0') # Filling the NULL values in Dependents as '0'

data.Self_Employed = data.Self_Employed.fillna('No') # Filling the NULL values in Self_Employed as 'No'

data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean()) # Filling the NULL values in LoanAmount as the mean of all Loan Amounts

data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0) # Filling the NULL values in Loan Amount Term as '360'

data.Credit_History = data.Credit_History.fillna(1.0) # Filling the NULL values in Credit History as '1.0'

data.apply(lambda x: sum(x.isnull()),axis=0) # To check if anything else is NULL

data.corr() # For Data Correlation

# Splitting training data
data.drop(['Loan_ID','Gender','Married'],axis=1)
X = data.iloc[:, 1: 12].values # For storing everything else then the Loan Status
Y = data.iloc[:, 12].values # For storing only the Loan Status

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 1/8, random_state = 0) # Spliting the values between Train and Test - the size of Test is 1/8 the size of Train

# Encoding categorical data for train data set

# Encoding the Independent Variable

LabelEncoderX = LabelEncoder()

for i in range(0, 5):
   Xtrain[:,i] = LabelEncoderX.fit_transform(Xtrain[:,i])

Xtrain[:,10] = LabelEncoderX.fit_transform(Xtrain[:,10])

# Encoding the Dependent Variable
LabelEncoderY = LabelEncoder()
Ytrain = LabelEncoderY.fit_transform(Ytrain)

# Encoding categorical data for test data set
# Encoding the Independent Variable

for i in range(0, 5):
    Xtest[:,i] =LabelEncoderX.fit_transform(Xtest[:,i])
Xtest[:,10] = LabelEncoderX.fit_transform(Xtest[:,10])

# Encoding the Dependent Variable
Ytest = LabelEncoderY.fit_transform(Ytest)

# Using Standard scaler to make sure the values are at lowest 0 and at max 1
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)

# FOR *** NAIVE BAYES ***
classifier = GaussianNB()
classifier.fit(Xtrain, Ytrain)

Ypred = classifier.predict(Xtest) # Preditcting the test set scores

# Measuring the accuracy

print("The Accuracy of Naive Bayes is:- ", metrics.accuracy_score(Ypred, Ytest))

# FOR *** KNN ***
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(Xtrain, Ytrain)

# Predicting the Test set results
Ypred = classifier.predict(Xtest)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of KNN is:- ', metrics.accuracy_score(Ypred, Ytest))

# Visualizing the Data

data.groupby('Loan_Status').size() 
sns.scatterplot(x=data['ApplicantIncome'],y=data['LoanAmount'], data=data,style=data['Loan_Status'], hue=data['Loan_Status'])   #displays the scatterplot of the given attributes
sns.jointplot(x="ApplicantIncome",y="CoapplicantIncome",data=data,hue="Loan_Status")   #displays the joint plot of the given attributes

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "blue", "red"])
cmap_bold = ["darkorange", "c", "darkblue"]

n_neighbors = 15 # Initial Value
weights = "uniform" # Initial Value

def plot_decision_boundaries(plt, X, y, x_name, y_name):

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

 
   

    # Put the result into a color plot
    Z=xx.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
       x=data['ApplicantIncome'],y=data['LoanAmount'], data=data,style=data['Loan_Status'], hue=data['Loan_Status']
    )
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
X = data.iloc[:, 1: 12].values # For storing everything else then the Loan Status
Y = data.iloc[:, 12].values # For storing only the Loan Status
for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Xtrain, Ytrain)

    plot_decision_boundaries(plt, Xtrain, Ytrain, data.ApplicantIncome[0], data.LoanAmount[1])
plt.show()






