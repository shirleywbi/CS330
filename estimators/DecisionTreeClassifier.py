from sklearn.tree import DecisionTreeClassifier
from constants import X, y

import re
import graphviz
from sklearn.tree import export_graphviz

def display_tree(feature_names, tree):
    """ For binary classification only """
    dot = export_graphviz(tree, out_file=None, feature_names=feature_names, class_names=tree.classes_.astype(str), impurity=False)
    # adapted from https://stackoverflow.com/questions/44821349/python-graphviz-remove-legend-on-nodes-of-decisiontreeclassifier
    dot = re.sub('(\\\\nsamples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])(\\\\nclass = [A-Za-z0-9]+)', '', dot)
    dot = re.sub(     '(samples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])\\\\n', '', dot)
    return graphviz.Source(dot)

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X, y)
tree.score(X, y)

# Plot
plot_classifier(X_nodup, y_nodup, tree, ticks=True)
plt.xlabel("Meat consumption (% days)")
plt.ylabel("Expected grade (%)")