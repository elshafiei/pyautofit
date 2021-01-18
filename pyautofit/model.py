"""TODO:
1. Feature selection
    1. iv
    2. correlation
    3. gbm feature importance
    4. run another model for selected features

3. categorical feature
    1. 2 options
        1. one-hot permutation
        2. evidence
    2. lightgbm

4. segmentation
    1. segmentation col

"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier


from pyautofit.data import AutoFitDataset
from pyautofit.feature import AutoFitFeature
from pyautofit.utils import gini

logging.getLogger().setLevel(logging.INFO)

class AutoFitModel:
    def __init__(self, name, config, version):
        self.name = name
        self.binning_config = config.binning_config
        self.classifier_config = config.classifier_config
        self.version = version
        self.features = None
        self.gbm = None

    @classmethod
    def from_pickle(self, path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f'Model successfully loaded from {path}')
            return model
        except Exception as e:
            logging.error(e)


    def to_pickle(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Model successfully saved to {path}')
        except Exception as e:
            logging.error(e)

    def external_model_validation(self, af_ds):
        """Validation and character analysis an external model on given dataset
        Assume model only have features available, which assigned bins and points manually

        Args:
            af_ds: an AutoFitDataset object

        Returns:
            self
        """
        feats_df = af_ds.get_feat_df()
        flags = af_ds.df[af_ds.flag_col]
        instance_weight = None
        if af_ds.instance_weight_col:
            instance_weight = af_ds.df[af_ds.instance_weight_col]
        new_features = dict()
        for n, f in self.features.items():
            if n not in feats_df:
                logging.error(f'Feature not in AutoFitDataset: {n}')
                continue
            cutoffs = []
            for b in f.bins:
                cutoffs.append(b.left)
            cutoffs.append(float('inf'))
            tt_feat = AutoFitFeature(n).from_series(af_ds.df[n], flags, bin_method='manual', cutoffs=cutoffs, instance_weight=instance_weight)
            # handle missing bins in external model
            bp_map = {k:v for k,v in zip(f.bins, f.points)}
            tt_points = []
            for b in tt_feat.bins:
                if b=='missing':
                    tt_points.append(0)
                elif b in bp_map:
                    tt_points.append(bp_map[b])
                else:
                    tt_points.append(None)
            tt_feat.points = tt_points
            new_features[n] = tt_feat
        self.features = new_features


    def gbm_inference(self, af_ds):
        """Apply predict of trained xgboost model on an other dataset

        Args:
            af_ds: an AutoFitDataset object

        Returns:
            gini
        """
        if self.gbm is None:
            logging.error('Can not found a trained model')
            return

        df_val = self.replace_value(af_ds, method='order')
        feats = [f for f, _ in self.features.items()]
        X = df_val[feats]
        y = df_val[af_ds.flag_col]
        y_pred = self.gbm.predict_proba(X)[:,1]
        gini_val = gini(y.values, y_pred)
        logging.info(f'Gini on this sample is {gini_val}')
        return gini_val


    def auto_binning(self, af_ds, binning_config):
        """Auto binning features based on number of bads"""
        assert isinstance(af_ds, AutoFitDataset)

        # Numerical feature
        numerical_feat_df = af_ds.get_numerical_feat_df()

        method = binning_config.get('method')
        max_bins = int(binning_config.get('max_bins'))
        min_bads = int(binning_config.get('min_bads'))

        feat_ls = []
        if isinstance(numerical_feat_df, pd.DataFrame):
            for c_name in numerical_feat_df:
                try:
                    tt_feat = AutoFitFeature(c_name).from_series(numerical_feat_df[c_name], af_ds.df[af_ds.flag_col], method, bads_pct=1/max_bins, min_bads=min_bads)
                    feat_ls.append(tt_feat)
                except Exception as e:
                    logging.info(f'Error occured, numerical feature: {c_name}')
                    logging.error(e)


        # Categorical features
        categorical_feat_df = af_ds.get_categorical_feat_df()
        if isinstance(categorical_feat_df, pd.DataFrame):
            for c_name in categorical_feat_df:
                try:
                    tt_feat = AutoFitFeature(c_name).from_series(categorical_feat_df[c_name], af_ds.df[af_ds.flag_col], 'categorical')
                    feat_ls.append(tt_feat)
                except Exception as e:
                    logging.info(f'Error occured, categorical feature: {c_name}')
                    logging.error(e)

        feat_ls = [f for f in feat_ls if f]
        self.features = {f.name:f for f in feat_ls}

    def replace_value(self, af_ds, method='order'):
        """Replace raw values in autofit dataset with bined features or woe

        Args:
            method: one of ['woe', 'order', 'point']
            af_ds: input AutoFitDataset object

        Returns:
            a pandas dataset for modeling
        """
        assert method in ['woe', 'order', 'point']
        feat_ls = [f for f, v in self.features.items()]

        replaced_df = af_ds.df[feat_ls].copy()
        label_col = af_ds.df[af_ds.flag_col].copy()
        id_col = af_ds.df[af_ds.id_col].copy()
        cat_df = None
        if isinstance(af_ds.categorical_cols, list):
            cat_df = af_ds.df[af_ds.categorical_cols].copy()
        if af_ds.instance_weight_col:
            weights_col = af_ds.df[af_ds.instance_weight_col].copy()

        def _assign(col):
            feat = self.features[col.name]
            feat_map = {b: v for b, v in zip(feat.bins, feat.woe)}

            def _rules_woe(x):
                if x in feat.bins:
                    return feat_map[x]
                for b, v in feat_map.items():
                    if isinstance(b, pd.Interval):
                        if x in b:
                            return v
                return np.nan

            def _rules_order(x):
                if x in feat.bins:
                    return x

                for b in feat.bins:
                    if isinstance(b, pd.Interval):
                        if x in b:
                            return b.right
                return np.nan

            def _rules_point(x):
                if x in feat.bins:
                    return x

                for b,p in zip(feat.bins, feat.points):
                    if isinstance(b, pd.Interval):
                        if x in b:
                            return p
                return 0

            if method == 'woe':
                series = col.apply(_rules_woe)
                if 'missing' in feat_map:
                    series = series.fillna(feat_map['missing'])

            if method == 'order':
                series = col.apply(_rules_order)

            if method == 'point':
                series = col.apply(_rules_point)

            return series

        replaced_df = replaced_df.apply(_assign, axis=0)
        if isinstance(cat_df, pd.DataFrame):
            replaced_df = replaced_df.join(cat_df)
        replaced_df[af_ds.id_col] = id_col
        replaced_df[af_ds.flag_col] = label_col
        if af_ds.instance_weight_col:
            replaced_df[af_ds.instance_weight_col] = weights_col

        return replaced_df

    def _apply_monotone_constraints(self, correlation_threshold):
        # get monotonicity constraints
        def _constraint_fn(feat, correlation_threshold):
            x, y = [], []
            for b, w in zip(feat.bins, feat.woe):
                if isinstance(b, pd.Interval):
                    x.append(b.right)
                    y.append(w)
            try:
                r, _ = pearsonr(x, y)
                if r > correlation_threshold:
                    feat.is_monotonic = 1
                    return 1
                elif r < - correlation_threshold:
                    feat.is_monotonic = -1
                    return -1
                else:
                    # For features that non-monotonic, need further inspection
                    feat.is_monotonic = 0
                    return 0
            except:
                feat.is_monotonic = 0
                return 0
        if self.classifier_config.get('model') == 'xgboost':
            constraints = ''
            for k, v in self.features.items():
                constraints += f'{_constraint_fn(v, correlation_threshold)},'
            constraints = '(' + constraints + ')'
        elif self.classifier_config.get('model') == 'lightgbm':
            constraints = []
            for k, v in self.features.items():
                constraints.append(_constraint_fn(v, correlation_threshold))
        else:
            logging.error('Model not implemented')

        return constraints

    def modeling_xgboost(self, af_ds, apply_binning=True, correlation_threshold=.3, test_size=.2):
        """Apply xgboost tree on bined features

        Args:
            af_ds: an AutoFitDataset object
            binning: bool, apply auto_binning
            correlation_threshold(Optional): correlation threshold for applying monotone constraint
            test_size: test sample percentage

        Returns:
            xgboost model object
        """
        if apply_binning:
            self.auto_binning(af_ds, self.binning_config)

        df_bined = self.replace_value(af_ds, method='order')

        # assume intercept
        base_score = df_bined[af_ds.flag_col].mean()

        # monotone constraints
        constraints = self._apply_monotone_constraints(correlation_threshold)

        clf_params = self.classifier_config.get('model_params')
        clf_params['base_score'] = base_score
        clf_params['monotone_constraints'] = constraints
        fit_params = self.classifier_config['fit_params']

        gbm = XGBClassifier(**clf_params)

        features = [k for k,_ in self.features.items()]
        x_cols = features
        if af_ds.instance_weight_col:
            x_cols = features + [af_ds.instance_weight_col]

        X_train, X_test, y_train, y_test = train_test_split(
            df_bined[x_cols], df_bined[af_ds.flag_col], test_size=test_size, random_state=fit_params.get('random_state'))

        sample_weight = None
        sample_weight_eval_set = None

        if af_ds.instance_weight_col:
            sample_weight = X_train[af_ds.instance_weight_col].values
            sample_weight_eval_set = X_test[af_ds.instance_weight_col].values
            X_train = X_train.drop(columns = af_ds.instance_weight_col)
            X_test = X_test.drop(columns = af_ds.instance_weight_col)

        gbm.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_train[features], y_train), (X_test[features], y_test)], sample_weight_eval_set=[
                sample_weight, sample_weight_eval_set], **fit_params)

        self.gbm = gbm

        logging.info(f'Gini on test sample is: {2*gbm.best_score-1}')

        return gbm

    def modeling_lightgbm(self, af_ds, apply_binning=True, correlation_threshold=.3, test_size=.2):
        if apply_binning:
            self.auto_binning(af_ds, self.binning_config)
        df_bined = self.replace_value(af_ds, method='order')

        clf_params = self.classifier_config.model_params
        constraints = self._apply_monotone_constraints(correlation_threshold)
        clf_params['monotone_constraints'] = constraints
        fit_params = self.classifier_config.fit_params

        features = [k for k,_ in self.features.items()]
        x_cols = features
        if af_ds.instance_weight_col:
            x_cols = features + [af_ds.instance_weight_col]

        clf = LGBMClassifier(**clf_params)
        X_train, X_test, y_train, y_test = train_test_split(
            df_bined[x_cols], df_bined[af_ds.flag_col], test_size=test_size, random_state=fit_params.get('random_state'))

        sample_weight = None
        sample_weight_eval_set = None

        if af_ds.instance_weight_col:
            sample_weight = X_train[af_ds.instance_weight_col].values
            sample_weight_eval_set = X_test[af_ds.instance_weight_col].values
            X_train = X_train.drop(columns = af_ds.instance_weight_col)
            X_test = X_test.drop(columns = af_ds.instance_weight_col)

        clf.fit(X=X_train, y=y_train, sample_weight=sample_weight, eval_set=(X_test, y_test), eval_sample_weight=[sample_weight_eval_set], **fit_params)
        self.gbm = clf

        eval_gini = 2*clf.best_score_['valid_0']['auc']-1
        logging.info(f'Gini on test sample is: {eval_gini}')

        return clf


    def _update_feat_gbm(self):
        assert self.gbm != None
        trees_df = self.gbm.get_booster().trees_to_dataframe()
        for i in range(self.gbm.best_iteration):
            feat_df = trees_df[trees_df['Tree'] == i].set_index('Node')

            feat = self.features[feat_df.iloc[0]['Feature']]
            split = feat_df.iloc[0]['Split']
            if feat.points == None:
                feat.points = [0 for _ in range(len(feat.bins))]
            for j, b in enumerate(feat.bins):
                if b == 'missing':
                    feat.points[j] += feat_df[feat_df['ID'] == feat_df.iloc[0]['Missing']]['Gain'].values[0]

                elif isinstance(b, pd.Interval) and b.right < split:
                    feat.points[j] += feat_df.iloc[1]['Gain']

                else:
                    feat.points[j] += feat_df.iloc[2]['Gain']


    def features_to_csv(self, path=None):
        df = pd.DataFrame()
        for _, v in self.features.items():
            df = df.append(v.to_csv())
        df = df.reset_index(drop=True)
        if path:
            df.to_csv(path, index=False)
        return df

    def character_analysis(self, path=None):
        self._update_feat_gbm()
        new_feats = dict()
        for k, v in self.features.items():
            new_feats[k] = v.merge_bins_by_points()
        self.features = new_feats

        df = self.features_to_csv()
        df = df.dropna(subset=['points'])
        inter_row = []
        for col in df.columns:
            if col == 'points':
                inter_row.append(self.gbm.base_score)
            elif col == 'feature':
                inter_row.append('intercept')
            else:
                inter_row.append(np.nan)
        df.loc[-1] = inter_row
        df.index = df.index + 1
        df = df.sort_index()
        df = df.reset_index(drop=True)
        if path:
            df.to_csv(path, index=False)
        return df

    def plot_feature_importance(self):
        try:
            plot_importance(self.gbm)
        except Exception as e:
            logging.error(e)
            logging.error("No trained model found, please run modeling first!")


"""Test script
"""
# if __name__ == '__main__':
#     import pandas as pd
#     import json
#     from pyautofit.config import AutoFitConfig
#     config = AutoFitConfig.from_json('configs/autofit_config.json')
#     af_ds = AutoFitDataset.from_csv('demo.csv', config.dataset_config)
#     model = AutoFitModel('demo', config, 'demo')
#     model.auto_binning(af_ds, binning_config=config.binning_config)
#     replaced_df = model.replace_value(af_ds, method='order')
