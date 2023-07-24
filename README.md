### Groupby imputer

Dealing with missing values in our earlier machine learning projects my groupmates and I every now and then thought of filling them with values different for some distinct groups of objects. Sklearn doesn't offer a built-in filling with the median by groups, and writing a custom transformer seemed too complicated then.

So, here is a custom imputer, Sklearn compatible, which takes the name of a column to fill missing values and the name of the column to group by and returns the same dataset with NaN's in the desired column filled, by default, with the median values for groups.

If the group_by_column is categorical groups are its unique values. If it's numeric the desired number of groups can be specified: the range will be split into this number of groups with equal intervals. Default num_groups is 4.

Default aggfunc is median, but mean, min amd max can also be used.

Next I'm planning to add grouping by more than one column, filling NaN's in more than one column and an option to split the numeric value range by percentiles not real intervals - so that the groups were equal. Any other ideas (actually any feedback) welcomed


### Ffill imputer

Then I thought of another project where all of us agreed that the best strategy of dealing with missing values would be filling then with closest present value, either previous or next. Ffil was a suitable pandas method, though a good practice would be doing preprocessing in a pipeline. One cannot just put ffill into sklearn pipeline, sklearn seemed to have no analogues, and my code reviewer said that making a custom transformer was the only chance. So we just filled them with the median I suppose. But now - meet a sklearn compatible ffill imputer

It takes the name of a column to fill. The column by which the dataset should be ordered could alse be specified. If not the imputer will look for any datetime columns and take the first. If there are no dates it'll order the dataset by index. In the output the dataset is reindexed again - to correspond the original X.

The strategy of imputation can be chosen from ffill, bfill or both (first ffill, then bfill to fill the very first row in case it has NaN in it)