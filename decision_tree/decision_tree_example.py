import ipywidgets
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_blobs, make_circles, make_moons


class DecisionTreeExample:
    """
    DecisionTreeExample demonstrates the decision tree for classification problems by
    providing visualizations for different cases. Jupyter widgets are used to enhance
    the interactive nature of the implementation.

    Attributes
    ----------
    datasets : dict
        A dictionary whose keys are strings and values are a dictionary containing the
        function to be run and its arguments.
    models_class : dict
        A dictionary whose keys are strings representing the classifier, and values are
        sklearn classifier objects.
    clf_class : DecisionTreeClassifier or LogisticRegression class from sklearn
    classifier_string : str
        The classifier that is currently used. Gets updated based on the choices made
        in the notebook.
    max_depth : int
        The depth of the tree.
    display : bool
        Decides whether to display the contour plots of the classifier.
    split_data : bool
        If True, displays two scatter plots of the dataset, with half the number of
        samples in each.

    Notes
    -----
    A minimal Jupyter notebook:

    >>> %matplotlib nbagg
    >>> from dt_v1 import DecisionTreeExample
    >>> dte = DecisionTreeExample()
    >>> dte.widget_decision_tree()

    This minimal notebook is contained in decision_tree_notebook.ipynb.
    When this is run, it provides a framework to see different datasets and experiment
    with the DecisionTreeClassifier and LogisticRegression classes from sklearn. The
    need for reducing variation in decision trees is also motivated by splitting the
    dataset. This lays the foundation for an understanding of bootstrap, bagging and
    random forests.
    """

    def __init__(self):
        self.datasets = {
            "blobs": {
                "widget": self.get_blobs,
                "args": {
                    "num_classes": ipywidgets.SelectionSlider(
                        options=[1, 2, 3, 4, 5], value=2
                    ),
                    "cluster_std": ipywidgets.SelectionSlider(
                        options=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], value=1.0
                    ),
                },
            },
            "circles": {
                "widget": self.get_circles,
                "args": {
                    "noise": ipywidgets.SelectionSlider(
                        options=np.arange(0, 1, 0.05), value=0.1
                    ),
                    "scale_factor": ipywidgets.SelectionSlider(
                        options=np.arange(0, 1, 0.1),
                        value=0.5,
                    ),
                },
            },
            "moons": {"widget": self.get_moons, "args": {},},
        }
        self.models_class = {
            "decision_tree": DecisionTreeClassifier,
            "logistic_regression": LogisticRegression,
        }
        self.scatter_count = 0
        self.clf_class = DecisionTreeClassifier
        self.classifier_string = "decision_tree"
        self.max_depth = 1
        self.display = False
        self.split_data = False

    def widget_decision_tree(self):
        """
        The main widget function to be called from the Jupyter notebook.
        """
        ipywidgets.interact(
            self.widget_dataset,
            dataset=ipywidgets.Dropdown(
                options=["blobs", "circles", "moons"], value=None
            ),
        )

    def widget_dataset(self, dataset=None):
        """
        Parameters
        ----------
        dataset : str
            One of the keys of the instance attribute "datasets" such as "blobs",
            "circles", or "moons".
        Notes
        -----
        Displays the chosen dataset and calls other relevant widget methods.
        """
        if not dataset:
            return
        self.display = False
        self.split_data = False
        plt.close("all")
        self.fig_1, self.axs_1 = plt.subplots(1, 2, figsize=(8, 4))
        self.axs_1[1].axis("off")
        chosen_dataset = self.datasets[dataset]  # get_blobs
        ipywidgets.interact(chosen_dataset["widget"], **chosen_dataset["args"])
        ipywidgets.interact(self.choose_classifier, classifier=ipywidgets.Dropdown(
            options=["decision_tree", "logistic_regression"],
            value="decision_tree"
        ))
        ipywidgets.interact(
            self.run_widget_dt,
            max_depth=ipywidgets.SelectionSlider(
                options=range(1, 20), value=1
            ),
        )

    def choose_classifier(self, classifier):
        """
        Parameters
        ----------
        classifier : str
            Either "logistic_regression" or "decision_tree"

        Notes
        -----
        Sets the instance attribute "cls_class" based on the classifier chosen.
        This method will be called from widget_dataset.
        """
        self.classifier_string = classifier
        self.clf_class = self.models_class[classifier]
        self.run_widget_dt(self.max_depth, self.display, self.split_data)

    def run_widget_dt(self, max_depth=3, display=False, split_data=False):
        """
        Parameters
        ----------
        max_depth : int
            The max_depth parameter for the DecisionTreeClassifier from sklearn.
        display : bool
            Decides whether to display the contour plots of the classifier.
        split_data : bool
            If True, displays two scatter plots of the dataset, with half the number of
            samples in each.

        Notes
        -----
        Calls the "display_decision_tree" based on the parameters passed.
        Sets the instance attributes "max_depth", "display", "split_data".
        This method will be called from widget_dataset or choose_classifier.
        """
        self.max_depth = max_depth
        self.display = display
        self.split_data = split_data
        X, y = self.X, self.y
        self.display_decision_tree(
            X, y, self.axs_1[0], self.axs_1[1], max_depth, display
        )
        if split_data:
            self.fig_2, self.axs_p1 = plt.subplots(1, 2, figsize=(8, 4))
            self.fig_3, self.axs_p2 = plt.subplots(1, 2, figsize=(8, 4))
            self.split_dataset_axis = [self.axs_p1, self.axs_p2]
            self.display_decision_tree(
                X[0::2], y[0::2], self.axs_p1[0], self.axs_p2[0], max_depth, display
            )
            self.display_decision_tree(
                X[1::2], y[1::2], self.axs_p1[1], self.axs_p2[1], max_depth, display
            )
        else:
            try:
                plt.close(self.fig_2)
                plt.close(self.fig_3)
            except:
                pass

    def get_blobs(self, num_classes, cluster_std):
        """
        Parameters
        ----------
        num_classes : int
            The number of classes
        cluster_std : float
            The standard deviation of each cluster
        Notes
        -----
        Gets the blobs dataset from make_blobs method of sklearn.
        Sets the instance attributes "num_classes", "X", "y" and calls show_scatter_plot.
        This method will be called from widget_dataset.
        """
        self.num_classes = num_classes
        self.X, self.y = make_blobs(
            n_samples=300, centers=num_classes, random_state=0, cluster_std=cluster_std
        )
        self.show_scatter_plot()

    def get_circles(self, noise=0.05, scale_factor=0.8):
        """
        Parameters
        ----------
        noise : float
            The noise standard deviation.
        scale_factor : float, between 0 and 1
            The scale factor between the circles.
        Notes
        -----
        Gets the circles dataset from make_circles method of sklearn.
        Sets the instance attributes "num_classes", "X", "y" and calls show_scatter_plot.
        This method will be called from widget_dataset.
        """
        self.X, self.y = X, y = make_circles(
            n_samples=300, noise=noise, factor=scale_factor
        )
        self.num_classes = 2
        self.show_scatter_plot()

    def get_moons(self):
        """
        Gets the moons dataset from make_moons method of sklearn.
        Sets the instance attributes "num_classes", "X", "y" and calls show_scatter_plot.
        This method will be called from widget_dataset.
        """
        self.X, self.y = make_moons(n_samples=300)
        self.num_classes = 2
        self.show_scatter_plot()

    def show_scatter_plot(self):
        """
        Displays the scatter plot of the datasets from instance attributes "X" and "y".
        """
        self.axs_1[0].clear()
        self.axs_1[1].clear()
        self.axs_1[0].scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=30)
        self.axs_1[0].axis("tight")
        self.axs_1[1].axis("off")
        self.choose_classifier(self.classifier_string)

    def display_decision_tree(self, X, y, ax1, ax2, max_depth=3, display=False):
        """
        Parameters
        ----------
        ax1 : matplotlib axes object
        ax2 : matplotlib axes object
        max_depth : int
        display : bool
        """
        if self.clf_class == DecisionTreeClassifier:
            clf_instance = self.clf_class(max_depth=max_depth)
        elif self.clf_class == LogisticRegression:
            clf_instance = self.clf_class()
        else:
            return
        clf_instance.fit(X, y)
        ax1.clear()
        ax2.clear()
        ax1.scatter(X[:, 0], X[:, 1], c=y, s=30)
        ax1.axis("tight")
        ax2.axis("off")
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()

        xx, yy = np.meshgrid(
            np.linspace(*xlim, num=200), np.linspace(*ylim, num=200)
        )
        a = np.c_[xx.ravel(), yy.ravel()]
        Z = clf_instance.predict(a).reshape(xx.shape)

        if display:
            contours = ax1.contourf(
                xx,
                yy,
                Z,
                alpha=0.3,
                levels=np.arange(self.num_classes + 1) - 0.5,
            )
            if self.clf_class == DecisionTreeClassifier:
                plot_tree(clf_instance, ax=ax2)


if __name__ == "__main__":
    dte = DecisionTreeExample()
