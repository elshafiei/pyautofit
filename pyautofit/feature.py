"""
TODO:
edge_cases in binning
categorical features
29/05/20: removed num_bins in binning method
"""

import logging
import pandas as pd
import numpy as np
import copy
from pyautofit.utils import iv, woe

class AutoFitFeature:
    def __init__(self, name, bins=None, points=None, is_categorical=False):
        self.name = name
        self.set_bins_points(bins, points)
        self.is_categorical = is_categorical
        self.bads = None
        self.bads_pct = None
        self.goods = None
        self.goods_pct = None
        self.counts = None
        self.counts_pct = None
        self.bad_rate = None
        self.woe = None
        self.iv = None
        self.is_monotonic = None

    def set_bins(self, bins):
        if self._validation(bins, None):
            # When update bins only, set points of old bins to None
            if self.points:
                self.points = None
            self.bins = bins
            return True
        return False

    def set_points(self, points):
        if self._validation(self.bins, points):
            self.points = points
            return True
        return False

    def set_bins_points(self, bins, points):
        if self._validation(bins, points):
            self.bins = bins
            self.points = points
            return True
        else:
            logging.error('Number of Bins and Points are wrong, please make sure len(points) = len(bins) + 1')
            self.bins = None
            self.points = None
            return False

    def from_series(self,
        series,
        flags,
        bin_method,
        num_bins=None,
        bads_pct=None,
        min_bads=50,
        cutoffs=None,
        instance_weight=None,
        edge_cases=[]):
        """Construct feature object from pandas series

        Args:
            series: pandas series with feature raw data
            flags: pandas series with binary flag data, 1 for good and 0 for bad
            bin_method: one of ('num_bads', 'manual')
            num_bins: specify number of bins, required when bin_method == 'num_bins'
            bads_pct: specify bads percentage, required when bin_method == 'num_bads'
            min_bads: minimum number of bads in a bin, optional when bin_method == 'bads_pct'
            cutoffs: list of cutoff values, required when bin_method == 'manual'
            instance_weight: series with weight for each instance
            edge_cases(optional): list of special value who need an separate bin

        Returns:
            feat: a feature object
        """
        assert isinstance(series, pd.Series)
        assert isinstance(flags, pd.Series)
        assert bin_method in ['num_bads', 'manual', 'categorical']

        # Use weight column for bin count
        # if no weight column provided, then assign a all one
        if not isinstance(instance_weight, pd.Series):
            instance_weight = pd.Series([1 for _ in range(series.shape[0])], name='weights')


        # Create dfs:
        # 1. clean
        # 2. missing value
        # 3. one df per edge case
        full_df = pd.concat([series, flags, instance_weight], sort=False, axis=1)
        aux_dfs = []

        # Add a missing bin
        nan_df = full_df[full_df[series.name].isnull()].copy()
        df = full_df[full_df[series.name].isnull()].copy()
        nan_df['category'] = 'missing'
        aux_dfs.append(nan_df)

        # Define clean df
        df = full_df[full_df[series.name].notnull()].copy()

        # Add special dfs
        for ec in edge_cases:
            temp_df = full_df[full_df[series.name] == ec]
            temp_df['category'] = ec
            aux_dfs.append(temp_df)
            df = df[df[series.name] != ec]

        if bin_method == 'num_bads':
            if bads_pct == None or min_bads == None:
                logging.error('Minimum number of bads need to be specified')
                return False

            total_bads = df[df[flags.name] == 0][instance_weight.name].values.sum()
            if total_bads < min_bads:
                logging.error(
                    f'Not enough bads in {series.name}, total bads: {total_bads} \n Try to reduce min_bads or use other method.')
                return False

            vc_temp = df[df[flags.name] == 0].groupby(series.name).agg({instance_weight.name:'sum'})
            value_counts = vc_temp[instance_weight.name].sort_index()

            bin_bads = round(total_bads * bads_pct, 0)
            if bin_bads < min_bads:
                bin_bads = min_bads
            cutoffs = []
            acc_bads = 0
            for i, v in value_counts.iteritems():
                if cutoffs == []:
                    cutoffs.append(-float('inf'))
                acc_bads += v
                if acc_bads >= bin_bads:
                    cutoffs.append(i)
                    acc_bads = 0

            if acc_bads >= bin_bads or acc_bads >= min_bads:
                cutoffs.append(i)
            else:
                cutoffs[-1] = i

            if len(cutoffs) == 2:
                logging.error(f'Feature: {self.name} dropped because only 1 valid bin found!')
                return

            res = pd.cut(df[series.name], cutoffs, duplicates='drop')

        elif bin_method == 'manual':
            if not isinstance(cutoffs, list):
                logging.error('Manual bin need a cutoff list.')
                return False
            res = pd.cut(df[series.name], cutoffs, duplicates='drop')
        elif bin_method == 'categorical':
            self.is_categorical = True
            res = df[series.name]
        else:
            logging.error('Method Not Implemented')
            return False

        df['category'] = res

        for aux in aux_dfs:
            df = df.append(aux, sort=False, ignore_index=True)

        # Get stats by groupby
        df['weighted_good'] = df.apply(lambda x: x[flags.name] * x[instance_weight.name], axis=1)
        stats = df.groupby('category').agg({instance_weight.name: 'sum', 'weighted_good': 'sum'})
        total_count = sum(stats[instance_weight.name].values)
        total_goods = sum(stats['weighted_good'].values)
        total_bads = total_count - total_goods
        if self.set_bins(bins=list(stats.index)):
            self.counts = stats[instance_weight.name].values
            self.counts_pct = self.counts / total_count
            self.goods = stats['weighted_good'].values
            self.goods_pct = self.goods/total_goods
            self.bads = self.counts - self.goods
            self.bads_pct = self.bads/total_bads
            self.bad_rate = self.bads/self.counts
            self.woe = woe(self.goods/total_goods, self.bads/total_bads)
            self.iv = iv(self.goods/total_goods, self.bads/total_bads)
        else:
            logging.error('Could not set bins for this feature!')
            return
        return self

    def merge_bins_by_points(self):
        """Merge bins with same points
        """
        if self.points == None:
            return self

        self._validation(self.bins, self.points)
        feat_df = self.to_csv()
        missing = feat_df[feat_df['bins'] == 'missing']
        feat_gp = feat_df[feat_df['bins'] != 'missing'].groupby('points')

        bins_ls = []
        points_ls = []
        bads_ls = []
        goods_ls = []

        for i in feat_gp.groups:
            bin_df = feat_gp.get_group(i).sort_values(by='bins')

            left = None
            for _, row in bin_df.iterrows():
                if left is None:
                    left = row['bins'].left
                    right = row['bins'].right
                    goods = row['goods']
                    bads = row['bads']
                    continue

                if right == row['bins'].left:
                    right = row['bins'].right
                    goods += row['goods']
                    bads += row['bads']

                else:
                    bins_ls.append(pd.Interval(left, right, closed='right'))
                    points_ls.append(row['points'])
                    goods_ls.append(goods)
                    bads_ls.append(bads)

                    left = row['bins'].left
                    right = row['bins'].right
                    goods = row['goods']
                    bads = row['bads']

            bins_ls.append(pd.Interval(left, right, closed='right'))
            points_ls.append(row['points'])
            goods_ls.append(goods)
            bads_ls.append(bads)

        # add missing bin
        if missing.shape[0] > 0:
            bins_ls.append('missing')
            points_ls.append(missing['points'].values[0])
            goods_ls.append(missing['goods'].values[0])
            bads_ls.append(missing['bads'].values[0])

        assert len(bins_ls) == len(goods_ls)
        assert len(bins_ls) == len(bads_ls)
        assert len(bins_ls) == len(points_ls)

        if self.set_bins_points(bins_ls, points_ls):
            goods_ls = np.array(goods_ls)
            bads_ls = np.array(bads_ls)
            self.goods = goods_ls
            self.bads = bads_ls
            self.counts = self.goods + self.bads
            total_count = self.counts.sum()
            total_goods = self.goods.sum()
            total_bads = self.bads.sum()
            self.goods_pct = self.goods/total_goods
            self.bads_pct = self.bads/total_bads
            self.counts_pct = self.counts/total_count
            self.bad_rate = self.bads/self.counts
            self.woe = woe(self.goods/total_goods, self.bads/total_bads)
            self.iv = iv(self.goods/total_goods, self.bads/total_bads)
            return self
        else:
            logging.error('Could not set bins for this feature!')
            return False

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_csv(self, path=None):
        df = pd.DataFrame()
        feat_dict = self.to_dict()
        for k, v in feat_dict.items():
            if k != 'name':
                df[k] = v
        df['feature'] = feat_dict['name']
        if path:
            df.to_csv(path, index=False)
        return df

    def _validation(self, bins=None, points=None):
        bins = np.array(bins)
        points = np.array(points)
        if len(bins.shape) == 0 and len(points.shape) == 0:
            return True

        if len(bins.shape) != 0 and len(points.shape) == 0:
            return True

        if bins.shape[0] == points.shape[0]:
            return True

        return False


"""Test script
could be migrate to pytest module
"""
# if __name__ == '__main__':
#     test_df = pd.read_csv('feat_final.csv')
#     cat_feat = AutoFitFeature('tt').from_series(test_df['payment_type_NEG_D_CC_txn_month_ratio'], flags=test_df['bad_flag'], bin_method='num_bads', bads_pct=0.1, min_bads=20)
#     cat_dict = cat_feat.to_dict()
#     print('!')