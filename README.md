Dealing with missing values in our earlier machine learning projects my groupmates and I every now and then thought of filling them with values different for some distinct groups of objects. Sklearn doesn't offer a built-in filling with the median by groups, and writing a custom transformer seemed too complicated then.

So, here is a custom imputer, Sklearn compatible, which takes the name of a column to fill missing values and the name of the column to group by and returns the same dataset with NaN's in the desired column filled, by default, with the median values for groups.

If the group_by_column is categorical groups are its unique values. If it's numeric the desired number of groups can be specified: the range will be split into this number of groups with equal intervals. Default num_groups is 4.

Default aggfunc is median, but mean, min amd max can also be used.

Next I'm planning to add grouping by more than one column, filling NaN's in more than one column and an option to split the numeric value range by percentiles not real intervals - so that the groups were equal. Any other ideas (actually any feedback) welcomed