DecisionTreeExample class is implemented in decision_tree_example.py. This class
demonstrates the decision tree for classification problems by providing visualizations.
Jupyter widgets are used to enhance the interactive nature of the notebook.

A minimal Jupyter notebook:

```python
>>> %matplotlib nbagg
>>> from decision_tree_example import DecisionTreeExample
>>> dte = DecisionTreeExample()
>>> dte.widget_decision_tree()
```

This minimal notebook is contained in decision_tree_notebook.ipynb.
When this is run, it provides a framework to see different datasets and experiment with
the DecisionTreeClassifier and LogisticRegression classes from sklearn. The need for
reducing variance in decision trees is also motivated by splitting the dataset. This
lays the foundation for an understanding of bootstrap, bagging and random forests.
