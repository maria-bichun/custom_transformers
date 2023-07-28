import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupbyImputer(BaseEstimator, TransformerMixin):
  def __init__(self, column_to_fill, groupby_column, num_groups=None, aggfunc='median'):
    self.column_to_fill = column_to_fill
    self.groupby_column = groupby_column
    self.aggfunc = aggfunc
    if num_groups:
      self.num_groups = num_groups
    else: self.num_groups = 4

  
  def fit(self, X, y=None):
    """fit method creates groups and group medians attributes within class instance.
    If dtype is numeric and there're more unique values than number of groups required,
    the range is cut into given number of groups with equal intervals (default number of groups is 4).
    If groupby column is categorical or there are not many numeric values in it, every unique value makes up a group.
    Median value in the column to fill for every group is calculated
    """
    # for numeric
    if (pd.api.types.is_numeric_dtype(X[self.groupby_column])) and (X[self.groupby_column].nunique() > self.num_groups):      
      X['groups'] = pd.cut(X[self.groupby_column], self.num_groups)
      self.groups = X['groups'].unique()   
      self.medians = []
      for group in self.groups:
        self.medians.append(X[X['groups'] == group][self.column_to_fill].agg(self.aggfunc))   
    # for categorial
    else:
      self.groups = X[self.groupby_column].unique()
      self.medians = []
      for group in self.groups:
        self.medians.append(X[X[self.groupby_column] == group][self.column_to_fill].agg(self.aggfunc))
    return self

  def transform(self, X, y=None):
    """transform method iterates over groups defined within fit method.
    if the row belongs to a given group and the value in column to fill is missing, it's filled with the pre-calculated median.
    afterwards all groups are concatenated and the result reindxed to match the original dataframe
    """
    new_dfs = []
    for i in range(len(self.groups)):
      # for numeric
      if (pd.api.types.is_numeric_dtype(X[self.groupby_column])) and (X[self.groupby_column].nunique() > self.num_groups):      
        X['groups'] = pd.cut(X[self.groupby_column], self.num_groups)
        group_df = X[X['groups'] == self.groups[i]].copy()
      # for categorial  
      else:
        group_df = X[X[self.groupby_column] == self.groups[i]].copy()

      group_df[self.column_to_fill] = group_df[self.column_to_fill].fillna(self.medians[i])
      new_dfs.append(group_df)
    new_df = pd.concat(new_dfs).reindex(X.index)  
    return new_df

  def agg(aggfunc):
    """choice of aggregation function from user input.
    median by default
    """
    if aggfunc == 'mean':
      return pd.DataFrame.mean
    elif aggfunc == 'min':
      return pd.DataFrame.min
    elif aggfunc == 'max':
      return pd.DataFrame.max
    else: return pd.DataFrame.median

class FfillImputer(BaseEstimator, TransformerMixin):
  def __init__(self, column_to_fill, orderby_column=None, strategy='ffill'):
    self.column_to_fill = column_to_fill        
    self.orderby_column = orderby_column
    self.strategy = strategy

  def fit(self, X, y=None):
    """fit method tries to find a datetime column to sort the dataframe by values in that column.
    if there's none, index is used to order by.
    returns self with attribute of sorted dataframe added
    """
    if not self.orderby_column:
      # try to find col with dates
      try:
        self.orderby_column = X.select_dtypes(include='datetime').iloc[:,0].name
      except:
        pass
    
    if not self.orderby_column:
      # sort by index
      self.sorted_df = X.sort_index()
    else:
      self.sorted_df = X.sort_values(by=self.orderby_column)
    return self


  def transform(self, X, y=None):
    """transform method simply implements pandas.ffill or pandas.bfill or both consequently on the previously sorted dataframe.
    afterwards the result is reindxed to match the original dataframe
    """
    new_df = self.sorted_df.copy()
    if self.strategy == 'ffill':
      new_df[self.column_to_fill] = new_df[self.column_to_fill].ffill()
    elif self.strategy == 'bfill':
      new_df[self.column_to_fill] = new_df[self.column_to_fill].bfill()
    elif self.strategy == 'both_ways':
      new_df[self.column_to_fill] = new_df[self.column_to_fill].ffill()
      new_df[self.column_to_fill] = new_df[self.column_to_fill].bfill()
    new_df = new_df.reindex(X.index)
    return new_df
