A chi-square test is used in statistics to test the independence of two events. Given the data of two variables, we can get observed count O and expected count E. Chi-Square measures how expected count E and observed count O deviates each other.
Danger!: Can lead to errors when a feature is low-frequent

![[Pasted image 20210912152810.png]]

 In feature selection, we aim to select the features which are highly dependent on the response.

When two features are independent, the observed count is close to the expected count, thus we will have smaller Chi-Square value. So high Chi-Square value indicates that the hypothesis of independence is incorrect. In simple words, higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training.

Steps to perform the Chi-Square Test:
1.  Define Hypothesis.
2.  Build a Contingency table.
3.  Find the expected values.
4.  Calculate the Chi-Square statistic.
5.  Accept or Reject the Null Hypothesis.

![[Pasted image 20210912153511.png|300]]
![[Pasted image 20210912153540.png|400]]
![[Pasted image 20210912153600.png|400]]

Link to the explanation and the Example:
[Chi-Square Test for Feature Selection in Machine learning | by sampath kumar gajawada | Towards Data Science](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223)