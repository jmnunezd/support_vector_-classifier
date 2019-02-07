#%%
from sklearn.model_selection import train_test_split
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 600)
pd.set_option('display.width', 1000)

data_set_metadata = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/'


# Functions that may come handy on future
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Import data
df = pd.read_csv('abalone.csv')
del df['Unnamed: 0']

# Creating the label of our desire variable adult, and preparing the format of new features
df['adult'] = df['sex'].apply(lambda sex: {'I': 0, 'M': 1, 'F': 1}.get(sex, ' '))
df['feature'] = df.apply(lambda row: [row['length'], row['whole_weight']], axis=1)

# Split the data in in sample and out sample
train, test = train_test_split(df, test_size=0.95)

# With this values we are going to train the models
x = list(train['feature'])
y = list(train['adult'])

# models we want to compare
models = (svm.SVC(kernel='linear', C=1),           # linear
          svm.SVC(kernel='rbf', gamma=1, C=1),     # expected to be underfitting
          svm.SVC(kernel='rbf', gamma=150, C=1),   # expected to be overfitting
          svm.SVC(kernel='rbf', gamma=20, C=5))    # expected to be just fine

models = (clf.fit(x, y) for clf in models)

# title for the plots
titles = ('Linear C=1',
          'rbf gamma=1 C=1',
          'rbf gamma=150 and C=1',
          'rbf gamma=20 and C=5')


# Set-up 2x2 grid for plotting
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Preparing plot
x0, x1 = train['length'], train['whole_weight']
xx, yy = make_meshgrid(x0, x1)


# Bokeh plot for inspection more lively the train set
p = figure(plot_width=500, plot_height=500)
p.circle(train[train['adult'] == 1]['length'], train[train['adult'] == 1]['whole_weight'], color='blue', alpha=0.3)
p.asterisk(train[train['adult'] == 0]['length'], train[train['adult'] == 0]['whole_weight'], color='red', alpha=0.3)

show(p)

# For each model create it's respective contour graph
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax.scatter(x0, x1, c=y, cmap=plt.cm.RdYlBu, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Abalon Length')
    ax.set_ylabel('Abalone Whole Weight')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    # Using the model predict if the new featuresa are adults of not, then, print the result
    results = clf.predict(list(test['feature']))
    rdf = test.copy()
    rdf['prediction'] = np.array(results)
    rdf['hit'] = rdf.apply(lambda row: 1 if row['prediction'] == row['adult'] else 0, axis=1)

    print(f'{title} has an accuracy of: ', np.round(rdf.describe().at['mean', 'hit'], 4))


plt.show()

