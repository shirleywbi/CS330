from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

# Some Classifiers
classifiers = {
    'logistic regression' : pipe_lr,
    'decision tree' : dt,
    'random forest' : rf,
    'XGBoost' : XGBClassifier(scale_pos_weight=500, random_state=999), 
    'LightGBM' : LGBMClassifier(class_weight='balanced', random_state=999),
    'CatBoost' : CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=999)
}

# GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=999) # no class_weight
gb_results = pd.DataFrame(cross_validate(gb, X_train, y_train, scoring=score_method, return_train_score=True)).mean()

summary = pd.DataFrame([lr_results, dt_results, rf_results, gb_results], index=["logistic regression", "decision tree", "random forest", "gradient boosting"])
summary.sort_values(by=["test_score"], ascending=False)

