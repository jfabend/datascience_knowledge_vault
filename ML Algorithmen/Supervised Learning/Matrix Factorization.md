Wenn es eine Matrix A gibt, die in den Spalten Filme, in den Zeilen Menschen und in den Zellen Präferenzen enthält, dann kann man mit Matrix Factorization für jede Dimension eine Faktormatrix abschätzen. Wenn man dann die Faktormatrizen miteinander multipliziert, erhält man die Präferenzwerte.
Leere Zellenwerte lassen sich so predikten.

Mit dem [[Gradient Descent]] kann man sich den Faktormatrizen annähern, in dem man anfangs random-Faktorwerte einsetzt und dann den Error aller Werte von den tatsächlichen Präferenzwerten berechnet