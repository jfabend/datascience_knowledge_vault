Mit Momentum kann man auf der Suche nach dem globalen Minimum lokale Minima überspringen.

Die Idee ist hier, beim [[Gradient Descent]] oder anderen Optimizern (?) nicht nur auf Basis des aktuellen Wertes zu entscheiden, in welche Richtung man weitergeht, sondern auf Basis des Durchschnittes der letzten n steps.

So können kleinere Hügel auf dem Weg zum globalen Minima übersprungen werden, wenn die absteigenden Gradient Descent Werte der letzten Steps ausreichend "Schwung" mit sich bringen.

Man kann die GD-Werte der letzten Steps auch gewichten, so dass der vorherige Step mehr gewichtet wird als z.B. der vor-vor-vor-letzte Step

