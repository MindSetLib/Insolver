import pandas as pd 
import numpy as np

class Sampling:
    """ 
    A class for performing sampling with the dataset. 
    It includes several different techniques: simple sampling,systematic sampling, cluster sampling, 
    stratified sampling.
    
    Parameters:
        n (int): This parameter is used in chosen sampling method:
            - for a `simple` sampling `n` is the number of values to keep;
            - for a `systematic` sampling `n` is the number of step size;
            - for a `cluster` sampling `n` is the number of clusters to keep;
            - for a `stratified` sampling `n` is the number of values to keep in each cluster.
        n_clusters (int), default = 10: Number of clusters for the `cluster` and `stratified` sampling.
        method (str): Sampling method, supported methods: `simple`, `systematic`, `cluster`, `stratified`. 
    """
    def __init__(self, n, n_clusters=10, method='simple'):
        self.method = method
        self.n = n
        self.n_clusters = n_clusters
        
    def sample_dataset(self, df):
        """
        A method for performing sampling with the dataset using selected method.
        
        Parameters:
            df (pd.DataFrame): Dataframe to sample.
        
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
        A method for simple sampling. 
        
        Returns:
            New dataset with selected rows.
        """
        simple_random_sample = df.sample(n = self.n)
        return simple_random_sample
    
    def _systematic_sampling(self, df):
        """
        A method for systematic sampling. 
        
        Returns:
            New dataset with selected rows.
        """
        indexes = np.arange(0, len(df), step = self.n)
        systematic_sample = df.iloc[indexes]
        return systematic_sample
    
    def _cluster_sampling(self, df):
        """
        A method for cluster sampling. 
        
        Returns:
            New dataset with selected rows.
        """
        cluster_df = self._create_clusters(df)
        cluster_sample = pd.DataFrame()
        
        if self.n > self.n_clusters:
            raise Exception (f'{self.n} cannot be bigger then {self.n_clusters}.')
            
        elif self.n == self.n_clusters:
            return df
        
        else:
            clusters_to_keep = np.random.randint(1, self.n_clusters, self.n)
            for cluster in clusters_to_keep:
                cluster_sample = pd.concat([cluster_sample, cluster_df[cluster_df['cluster_id']==cluster]])
                
        return cluster_sample
    
    def _stratified_sampling(self, df):
        """
        A method for stratified sampling. 
        
        Returns:
            New dataset with selected rows.
        """
        cluster_df = self._create_clusters(df)
        stratified_sample = pd.DataFrame()
        
        for cluster in range(1, self.n_clusters+1):
            sample_cluster = cluster_df[cluster_df['cluster_id'] == cluster].sample(n=self.n)
            stratified_sample = pd.concat([stratified_sample, sample_cluster])
            
        return stratified_sample
    
    def _create_clusters(self, df):
        """
        A method for creating dataframe with clusters. 
        
        Returns:
            New dataset with cluster column.
        """
        cluster_size = round(len(df)/self.n_clusters)
        new_df = df.copy()
        
        try:
            new_df['cluster_id'] = np.repeat([range(1, self.n_clusters + 1)], cluster_size)
            
        except(ValueError):
            indexes = np.repeat([range(1, self.n_clusters+1)], cluster_size)
            diff = len(indexes) - len(df)
            if diff > 0:
                new_df['cluster_id'] = np.delete(indexes, len(indexes)-1)
                
            if diff < 0:
                new_df['cluster_id'] = np.append(indexes, self.n_clusters)
        
        return new_df
