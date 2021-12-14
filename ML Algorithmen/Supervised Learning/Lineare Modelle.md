Man legt verschiedene Geraden in die Punktwolke und versucht, die Summe der Distanzen (Residuen) zwischen Regressionsgerade und Punkten zu minimieren

Daten müssen sollten Richtung [[Normalverteilung]] gehen

Features have to be scaled ([[Scaling]])

### What are the drawbacks of the linear model?

-   The assumption of linearity of the errors, deshalb Residuen checken
-   It can't be used for count outcomes or binary outcomes
-   There are overfitting problems that it can't solve

[[Lasso]] kann die Varianz in den Features etwas verringern und dann ähnlich vorgehen

Formeln:

The values for β0, β1, B2, … , βp are chosen using **the least square method**, which minimizes the sum of squared residuals (RSS)
- **Y = β0 + β1X1 + β2X2 + … + βpXp + ε**

where:

-   **Y**: The response variable
-   **Xj**: The jth predictor variable
-   **βj**: The average effect on Y of a one unit increase in Xj, holding all other predictors fixed
-   **ε**: The error term
