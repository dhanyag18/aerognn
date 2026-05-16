import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from aerognn.data.groups import get_groups
from sklearn.preprocessing import StandardScaler
import os

def cv_xgb(df):
    
    BATCH_GROUPS = get_groups()
    groups_labels = df['id'].map(BATCH_GROUPS).values
    cv_strategy = GroupKFold(n_splits = 10)
    param_grid = {
        'max_depth': [2, 4, 5, 8],
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': [0.01, 0.05, 0.1]
    }


    feature_names = ['m', 'n', 'AR', 'helical_twist', 'bulge',
                 'taper', 'num_setbacks', 'setback_reduction', 'chamfer_dist']

    X = df[feature_names]
    y = df['score']

    def evaluate_model(model, X, y, cv, groups):
        results = cross_validate(model, X, y, cv=cv, groups=groups_labels, scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'})
        return -results['test_mae'], results['test_r2']

    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X, y, groups=groups_labels)

    xgb_best = grid_search.best_estimator_
    xgb_mae_scores, xgb_r2_scores = evaluate_model(xgb_best, X, y, cv_strategy, groups_labels)
    
    xgb_mae = np.mean(xgb_mae_scores)
    xgb_r2 = np.mean(xgb_r2_scores)

    print(f'XGBoost MAE: {xgb_mae}, XGBoost R^2: {xgb_r2}')
    
def cv_gp(df):
    
    BATCH_GROUPS = get_groups()
    groups_labels = df['id'].map(BATCH_GROUPS).values
    feature_names = ['m', 'n', 'AR', 'helical_twist', 'bulge', 'taper', 'num_setbacks', 'setback_reduction', 'chamfer_dist']

    X = df[feature_names]
    y = df['score']
    kernel_rbf = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    kernel_matern_1 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
    kernel_matern_2 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    param_grid = {
        'kernel': [kernel_matern_1, kernel_matern_2, kernel_rbf],
        'alpha': [1e-5, 0.01, 0.1]
    }

    gp_base = GaussianProcessRegressor(n_restarts_optimizer=20, random_state=42)
    cv_strategy = GroupKFold(n_splits=10)

    grid_search_gp = GridSearchCV(
        estimator=gp_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search_gp.fit(X_scaled, y, groups=groups_labels)

    gp_best = GaussianProcessRegressor(
        **grid_search_gp.best_params_, 
        n_restarts_optimizer=20, 
        random_state=42
    )

    gp_mae_scores = -cross_val_score(gp_best, X_scaled, y, cv=cv_strategy, groups=groups_labels, scoring='neg_mean_absolute_error')
    gp_r2_scores = cross_val_score(gp_best, X_scaled, y, cv=cv_strategy, groups=groups_labels, scoring='r2')
    print(f"GP MAE:  {np.mean(gp_mae_scores):.4f}")
    print(f"GP R2:   {np.mean(gp_r2_scores):.4f}")
    

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "..", "..")
    results_csv = os.path.join(base_path, "data", "raw", "simulations.csv")
    df = pd.read_csv(results_csv)    
    cv_xgb(df)
    cv_gp(df)



