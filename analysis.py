from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, plot_confusion_matrix, plot_precision_recall_curve, precision_score, recall_score, f1_score

# Confusion Matrix
confusion_matrix(y_valid_fold, pipe_lr.predict(X_valid_fold)) # 2D array
plot_confusion_matrix(pipe_lr, X_train_fold, y_train_fold, display_labels=['Non fraud', 'Fraud']) # pretty plot


# Outputs precision, recall, f1-score, accuracy
print(classification_report(y_valid_fold, pipe_lr.predict(X_valid_fold), target_names=("Not Fraud", "Fraud"), digits=4))

precision_score(y_valid_fold, pipe_lr.predict(X_valid_fold))
recall_score(y_valid_fold, pipe_lr.predict(X_valid_fold))


# Precision-recall curve
from sklearn.metrics import plot_precision_recall_curve

plot_precision_recall_curve(pipe_lr, X_valid_fold, y_valid_fold)
plt.plot(recall_score(y_valid_fold, pipe_lr.predict(X_valid_fold)), precision_score(y_valid_fold, pipe_lr.predict(X_valid_fold)), '*r', markersize=15)

## AP score
from sklearn.metrics import average_precision_score

average_precision_score(y_valid_fold, pipe_lr.predict_proba(X_valid_fold)[:,1])


# ROC curves
from sklearn.metrics import plot_roc_curve

cm = confusion_matrix(y_valid_fold, pipe_lr.predict(X_valid_fold))
plot_roc_curve(pipe_lr, X_valid_fold, y_valid_fold)
plt.plot(cm[0,1]/(cm[0].sum()), cm[1,1]/(cm[1].sum()), '*r', markersize=15)


# Imbalanced datasets
## Thresholding
recall_score(y_valid_fold, pipe_lr.predict_proba(X_valid_fold)[:,1] > 0.5)
recall_score(y_valid_fold, pipe_lr.predict_proba(X_valid_fold)[:,1] > 0.8)

## Class Weight
### Specifying class weight
pipe_lr_1_10 = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight={1:10}))
])
pipe_lr_1_10.fit(X_train_fold, y_train_fold)

### Specifying balanced class weight
pipe_lr_balanced = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
pipe_lr_balanced.fit(X_train_fold, y_train_fold)


# Confidence Scores
from sklearn.metrics import log_loss

## What is the negative score if correct and 95% confident?
log_loss(y_true=np.array([0]), y_pred=np.array([[0.95, 0.05]]), labels=(0,1))