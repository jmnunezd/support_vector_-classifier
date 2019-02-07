# Support Vector Classifier 

This pretends to be an example of use for the sklearn.svm.SVC module using the famous abalone data set,
this data set contains 4177 rows, one per abalone with it's corresponding measures.

We'll recreate a common practice in the study of wild life... taking a sample of few individuals, make
measures on them and later, try on understand how the whole population in doing. For this reason and
trying to be as realistic as possible, we define a sample size of just 5% of the total abalones, and 
with the measure of length, whole_weight and sex (Male, Female, Infant) try to fit a model that predicts
knowing the length and whole_weight the adulthood of the abalone with high precision (This may come super handy
in real life, if for instance, determining the adulthood and sex of a abalone is much more complicated and 
expensive than measure other variables)

With this train data set we will train 4 models with distinct parameters in order to appreciate when a
model is apparently underfitted overfitted or just ok... the model are:
* a linear SVC with C=1            We are trying to get a simple standard model
* a rbf SVC with gamma=1 and C=1   We are trying to get a good model that do not use all its potencial 
* a rbf SVC with gamma=150 and C=1 We are trying to get a model overfitted
* a rbf SVC with gamma=20 and C=5  We are trying to get a balance model

As you can see we are playing and tweaking the parameters of the model in order to see in the plots how
this changes have repercussions in the frontier between the groups. Remember that here C represents the
measure of the tradeoff we are making between a large margin frontier and few miss classify points, and 
the gamma parameters is sort of a measure of the waviness of the curve we'll have.

If you run the code you'll see how the different models perform and how look their probabilities of 
success at the hour of classify a new prospect that do not come from the train set.