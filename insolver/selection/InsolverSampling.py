import pandas as pd
import numpy as np


class Sampling:
    """ 
    Sampling class.
    It includes several different techniques: simple sampling, systematic sampling, cluster sampling, 
    stratified sampling.
    
    Parameters:
        n (int): This parameter is used in chosen sampling method:
            for a `simple` sampling `n` is the number of values to keep;
            for a `systematic` sampling `n` is the number of step size;
            for a `cluster` sampling `n` is the number of clusters to keep;
            for a `stratified` sampling `n` is the number of values to keep in each cluster.
        n_clusters (int), default = 10: Number of clusters for the `cluster` and `stratified` sampling.
        cluster_column (str): Column name of the data frame used as clusters.
        method (str): Sampling method, supported methods: `simple`, `systematic`, `cluster`, `stratified`. 
    """
    def __init__(self, n, cluster_column=None, n_clusters=10, method='simple'):
        self.method = method
        self.n = n
        self.n_clusters = n_clusters
        self.cluster_column = cluster_column
        
    def sample_dataset(self, df):
        """
        A method for performing sampling with the dataset using selected method.
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Raises:
            NotImplementedError: If self.method is not supported.
            
        Returns:
            New dataset with selected rows.
        """
        sampling_dict = {
            'simple': lambda d: self._simple_sampling(d),
            'systematic': lambda d: self._systematic_sampling(d),
            'cluster': lambda d: self._cluster_sampling(d),
            'stratified': lambda d: self._stratified_sampling(d)
        }
        
        if self.method not in list(sampling_dict.keys()):
            raise NotImplementedError(f'{self.method} method is not supported.')
            
        new_df = sampling_dict[self.method](df)
        return new_df
        
    def _simple_sampling(self, df):
        """
        Simple sampling. 
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
            
        Returns:
            New dataset with selected rows.
        """
        simple_random_sample = df.sample(n=self.n)
        return simple_random_sample
    
    def _systematic_sampling(self, df):
        """
        Systematic sampling. 
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Returns:
            New dataset with selected rows.
        """
        indexes = np.arange(0, len(df), step=self.n)
        systematic_sample = df.iloc[indexes]
        return systematic_sample
    
    def _cluster_sampling(self, df):
        """
        Cluster sampling.
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Returns:
            New dataset with selected rows.
        """
        cluster_df = self._create_clusters(df)
        clusters_count = cluster_df['cluster_id'].unique().sum()
            
        cluster_sample = pd.DataFrame()
        
        if self.n > clusters_count:
            raise Exception(f'{self.n} cannot be bigger then number of clusters.')
            
        elif self.n == clusters_count:
            return df
        
        else:
            clusters_to_keep = np.random.choice(cluster_df['cluster_id'].unique(), self.n)
            for cluster in clusters_to_keep:
                cluster_sample = pd.concat([cluster_sample, cluster_df[cluster_df['cluster_id'] == cluster]])
                
        return cluster_sample
    
    def _stratified_sampling(self, df):
        """
        Stratified sampling. 
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Returns:
            New dataset with selected rows.
        """
        cluster_df = self._create_clusters(df)
        stratified_sample = pd.DataFrame()
        
        for cluster in cluster_df['cluster_id'].unique():
            sample_cluster = cluster_df[cluster_df['cluster_id'] == cluster].sample(n=self.n)
            stratified_sample = pd.concat([stratified_sample, sample_cluster])
            
        return stratified_sample
    
    def _create_clusters(self, df):
        """
        Creating dataframe with clusters. 
        If self.cluster_column is defined, the clusters column is created using the dataframe column.
        Otherwise the clusters are formed according to the existing order.
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Raises:
            ValueError: Values in the column must be not null.
            
        Returns:
            New dataset with cluster column.
        """
        cluster_size = round(len(df)/self.n_clusters)
        new_df = df.copy()
        
        if self.cluster_column:
            if df[self.cluster_column].isnull().sum() > 0:
                raise ValueError('All values in the column must be not null.')

            new_df = df.copy()
            new_df['cluster_id'] = df[self.cluster_column]
            
        else:
            try:
                new_df['cluster_id'] = np.repeat([range(1, self.n_clusters + 1)], cluster_size)

            except ValueError:
                indexes = np.repeat([range(1, self.n_clusters+1)], cluster_size)
                diff = len(indexes) - len(df)
                if diff > 0:
                    new_df['cluster_id'] = np.delete(indexes, len(indexes)-1)

                if diff < 0:
                    new_df['cluster_id'] = np.append(indexes, self.n_clusters)
        
        return new_df
