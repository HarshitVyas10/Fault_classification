
### 1.1. Import library


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### 1.2. Generate the dataset"""

dataset = pd.read_csv('AllFeaturesCut.csv')
X = dataset.iloc[:, 1:40].values
Y = dataset.iloc[:, 0].values

X.shape

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

"""### 1.3. Let's examine the data dimension"""

X.shape

Y.shape

"""## 2. Data split (80/20 ratio)"""



"""### 2.1. Import library"""

from sklearn.model_selection import train_test_split

"""### 2.2. Data split"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

"""### 2.3. Let's examine the data dimension"""

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape

"""### 3.1. Import modules"""

import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

"""### 3.2. Defining learning classifiers"""

names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM",
          "Decision_Tree", "Random_Forest", "Neural_Net",
         "Naive_Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB()]

"""### 3.3. Build Model, Apply Model on Test Data & Record Accuracy Scores"""

scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    scores.append(score)

scores

"""## 4. Analysis of Model Performance

### 4.1. Import library
"""

import pandas as pd
import seaborn as sns

"""### 4.2. Create data frame of *model performance*"""

df = pd.DataFrame()
df['name'] = names
df['score'] = scores
df

"""### 4.3. Adding colors to the data frame"""

#https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html

cm = sns.light_palette("seagreen",n_colors=6, as_cmap=True)
s = df.style.background_gradient(cmap=cm)
s



"""### 4.4. Bar plot of model performance"""

import seaborn as sns
import matplotlib.pyplot as plt

# Set a custom color palette
custom_palette = sns.color_palette("viridis")  # Reverse of the "Blues" palette
sns.set_palette("husl")  # Change the color scheme to "husl"

# Set the style to "whitegrid"
sns.set(style="whitegrid")

# Sort the DataFrame by the "score" column in ascending order
df_sorted = df.sort_values(by="score", ascending=False)

df_sorted["score"] *= 100

# Create the bar plot
ax = sns.barplot(y="name", x="score", data=df_sorted)

# Set plot title and labels
ax.set_xlabel("Accuracy (%)")
ax.set_ylabel("")  # Remove y-axis label

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Define the data
x_values = [10, 20, 30, 40, 50, 60]
y_values = [0.794118, 0.852941, 0.885, 0.9411, 0.9137, 0.9117]

# Create a line plot
plt.plot(x_values, y_values, marker='o')

# Add labels and title
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
# Show the plot
plt.grid(True, linewidth=0.25)
plt.show()


