Geht ähnlich vor wie [[Lineare Modelle]], lässt aber manche Features schrumpfen um die Varianz zu verringern

**least absolute shrinkage and selection operator**
[Introduction to Lasso Regression - Statology](https://www.statology.org/lasso-regression/#:~:text=%20The%20following%20steps%20can%20be%20used%20to,to%20ridge%20regression%20and%20ordinary%20least...%20More%20)

Linear Modell:

The values for β0, β1, B2, … , βp are chosen using **the least square method**, which minimizes the sum of squared residuals (RSS)
- **Y = β0 + β1X1 + β2X2 + … + βpXp + ε**

where:

-   **Y**: The response variable
-   **Xj**: The jth predictor variable
-   **βj**: The average effect on Y of a one unit increase in Xj, holding all other predictors fixed
-   **ε**: The error term