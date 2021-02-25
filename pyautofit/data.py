"""
1.
"""
import logging
import pandas as pd

class AutoFitDataset:
    """Constructs dataset for autofit"""
    def __init__(
            self,
            df,
            id_col='idx',
            flag_col='gbf',
            instance_weight_col=None,
            categorical_cols=None,
            exclusions=None
    ):
        """Constructs AutoFit Dataset
        Args:
            df: a pandas DataFrame
            id_col: id column name
            flag_col: flag column name, value should only have 1 and 0, where 1 for good and 0 for bad
            instance_weight_col(Optional): weight column for each record
            categorical_cols(Optional): categorical feature columns
            exclusions(Optional): a list contains columns not used in model

        Return:
            an AutoFitDataset instance
        """
        assert isinstance(df, pd.DataFrame)
        self.df = df
        self.id_col = id_col
        self.flag_col = flag_col
        self.instance_weight_col = instance_weight_col
        self.categorical_cols = categorical_cols
        self.exclusions = exclusions
        # validation
        self._val_df()

    @classmethod
    def from_dataframe(df, config):
        """Constructs AutoFitDataset from dataframe and dataset config dictionary

        Args:
            df: input dataframe
            config: dataset config, mandatory fields: id_col, flag_col

        Returns:
            an AutoFitDataset instance
        """
        af_ds = AutoFitDataset(
            df,
            config.get('id_col'),
            config.get('flag_col'),
            config.get('instance_weight_col'),
            config.get('categorical_cols'),
            config.get('exclusions')
        )
        return af_ds
    
    @classmethod
    def from_csv(ds, file_path, config):
        """Constructs AutoFitDataset from csv file and dataset config dictionary

        Args:
            file_path: csv file path
            config: dataset config, mandatory fields: id_col, flag_col

        Returns:
            an AutoFitDataset instance
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(e)
            return f'Couldn\'t load csv from {file_path}'
        af_ds = AutoFitDataset(
            df,
            config.get('id_col'),
            config.get('flag_col'),
            config.get('instance_weight_col'),
            config.get('categorical_cols'),
            config.get('exclusions')
        )
        return af_ds

    def _get_feat_cols(self):
        """Return feature columns in df"""
        exc_cols = [self.id_col, self.flag_col]
        if isinstance(self.exclusions, list):
            exc_cols += self.exclusions
        if self.instance_weight_col:
            exc_cols.append(self.instance_weight_col)
        feat_cols = list(set(self.df.columns) - set(exc_cols))
        return feat_cols

    def _val_df(self):
        """validate essential columns are in our dataframe"""
        assert self.id_col
        assert self.flag_col
        val_cols = [self.id_col, self.flag_col]
        if self.instance_weight_col:
            val_cols.append(self.instance_weight_col)
        if isinstance(self.categorical_cols, list):
            val_cols += self.categorical_cols
        if isinstance(self.exclusions, list):
            val_cols += self.exclusions
        for col in val_cols:
            assert col in self.df.columns

    def get_feat_df(self):
        return self.df[self._get_feat_cols()]

    def get_numerical_feat_df(self):
        if isinstance(self.categorical_cols, list):
            num_cols = list(set(self._get_feat_cols()) - set(self.categorical_cols))
            return self.df[num_cols]
        else:
            return self.df[self._get_feat_cols()]

    def get_categorical_feat_df(self):
        if isinstance(self.categorical_cols, list):
            return self.df[self.categorical_cols]
        else:
            return


"""Test script
"""
# if __name__ == '__main__':
#     ds_config = dict(
#         id_col = 'index',
#         flag_col = 'gbf',
#         instance_weight_col = 'weight',
#         categorical_cols = ['feat_cat']
#     )
#     afds = AutoFitDataset.from_csv('demo.csv', config=ds_config)
#     print('!')
