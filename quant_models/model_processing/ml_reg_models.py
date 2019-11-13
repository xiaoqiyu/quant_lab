# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : rpyxqi@gmail.com
# @file      : ml_reg_models.py


from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from quant_models.utils.decorators import timeit
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.model_processing.models import Model
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNet

config = get_config()
logger = Logger(log_level='DEBUG', handler='ch').get_log()


class Ml_Reg_Model(Model):
    def __init__(self, model_name=None):
        super().__init__(model_name)
        # Model.__init__(self, model_name)

    def build_model(self, **kwargs):
        regs = {
            'linear': linear_model.LinearRegression(),
            'gbdt': GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, min_samples_split=2,
                                              min_samples_leaf=1, init=None, random_state=None,
                                              max_features=None, max_depth=None,
                                              alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False),
            # max_features: 划分时候考虑的最大特征数，默认是‘None’，考虑所有的特征数；浮点数表示特征的百分比，
            # 或者'log2''sqrt'/'auto'
            'enet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'ridge': Ridge(alpha=.5),  # 正则化的线性模型，可以解决过拟合情况，L2范数惩罚项
            'lasso': Lasso(alpha=.5),  # 正则化的线性模型，可以解决过拟合情况，L1范数惩罚项
            'linear_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'poly_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=0.5, epsilon=0.2, gamma='auto',
                            kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'rbf_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                           kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'sigmoid_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                               kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # kernel values: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
            'decision_tree': DecisionTreeRegressor(max_depth=5),
            'random_forest': RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42),
            'linear_nnls': linear_model.LinearRegression(),
        }
        self.features = kwargs.get('features')
        self.model = regs.get(self.model_name)
        return self.model

    @timeit
    def train_model(self, train_X=[], train_Y=[], **kwargs):
        param_grids = self.get_grid_search_params(self.model_name)
        if param_grids:
            logger.info('Grid Search Tunning the model:{0} with params:{1}'.format(self.model_name, param_grids))
            model_tuning = GridSearchCV(self.model, param_grid=param_grids, scoring=make_scorer(mean_squared_error))
            model_tuning.fit(train_X, train_Y)
            self.model = model_tuning
            self._best_estimate = model_tuning.best_params_
            logger.info('Complete grid esarch tunning with best params:{0}'.format(self._best_estimate))
        sample_weights = kwargs.get('sample_weights')
        if self.model_name == 'linear_nnls':
            self.fit_linear_nnls(train_X, train_Y, sample_weight=sample_weights)
            mse_scores = []
            r2_scores = []
        else:
            self.model.fit(train_X, train_Y)
            mse_scores = cross_val_score(self.model, train_X, train_Y, cv=int(config['ml_reg_model']['cv']), n_jobs=-1,
                                         scoring=make_scorer(mean_squared_error))
            logger.info("Mean squared error 67%%: %0.5f -  %0.5f" % (
                mse_scores.mean() - mse_scores.std() * 3, mse_scores.mean() + mse_scores.std() * 3))

            r2_scores = cross_val_score(self.model, train_X, train_Y, cv=int(config['ml_reg_model']['cv']),
                                        scoring=make_scorer(r2_score))
            logger.info("r2_scores 67%%: %0.5f -  %0.5f" % (
                r2_scores.mean() - r2_scores.std() * 3, r2_scores.mean() + r2_scores.std() * 3))
        return mse_scores, r2_scores

    def output_model(self, path=None):
        if self.model_name and 'linear' in self.model_name.lower():
            coef, intercept = self.model.coef_, self.model.intercept_
            return coef, intercept

    def best_estimate(self, train_X, train_Y):
        if self.model_name == 'gbdt':
            # TODO add logics for n_estimators and learning_rate
            param_grids = [
                {'n_estimators': range(98, 105),
                 'subsample': [0.5, 0.6, 0.7],
                 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                 }
            ]
            cv = int(config['ml_reg_model']['cv'])
            self.fine_grained(param_grids, cv, 'neg_mean_squared_error', train_X, train_Y)
