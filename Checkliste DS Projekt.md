|  <h4>Scope</h4>  |
| ---- |
| &nbsp &nbsp &nbsp Wer zahlt? Welche Einsparung verspricht er sich? |
| &nbsp &nbsp &nbsp Welches Budget ergibt sich daraus? |

|  <h4>Fragestellungen / Hypothesen</h4>  |
| --- |
| &nbsp &nbsp &nbsp Gibt es bereits Vorarbeiten? |

|  <h4>Betriebsführung</h4>  |
| --- |   
| &nbsp &nbsp &nbsp Wie soll das Ergebnis genutzt werden? API? Web-Anwendung? Report? |

|  <h4>Wie soll der Erfolg des Projekts gemessen werden?</h4>  |
| --- |   
|    Welche Kennzahl soll verbessert werden? |

|  <h4>Datenverfügbarkeit</h4>  |
| --- |  
| |

|  <h4>Umgebung für gemeinsames Arbeiten einrichten</h4>  |
| --- |  
|  &nbsp &nbsp &nbsp S3 / Datenkbank |
|  &nbsp &nbsp &nbsp    Gitlab |
 |  &nbsp &nbsp &nbsp    Entwicklungsumgebung (Server) |
 |  &nbsp &nbsp &nbsp Docker Container |

|  <h4>Daten (-qualität) und Cleansing</h4>  |
| --- |
|  &nbsp &nbsp &nbsp   Gibt es eine Beschreibung der Daten? Spaltenbeschreibung? |
|  &nbsp &nbsp &nbsp   Grundsätzliche Checks |
 |  &nbsp &nbsp &nbsp &nbsp &nbsp - Passen die Datentypen der Spalten zu den Inhalten? |
|  &nbsp &nbsp &nbsp &nbsp &nbsp - Sind kategorische Features numerisch? |
|  &nbsp &nbsp &nbsp &nbsp &nbsp - Passen Spaltenname und Spalteninhalt zusammen? |
|  &nbsp &nbsp &nbsp [[Outlier- Anomaly Detection]]  |
|  &nbsp &nbsp &nbsp &nbsp &nbsp - mit unsupervised Verfahren finden |
| &nbsp &nbsp &nbsp &nbsp &nbsp - dann [[Outliers]] löschen, anpassen, winsorizing |
|  &nbsp &nbsp &nbsp Datenfehler |
|  &nbsp &nbsp &nbsp &nbsp &nbsp Datenfehler |
| &nbsp &nbsp &nbsp &nbsp &nbsp - NA-Werte |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Dupletten |
| &nbsp &nbsp &nbsp &nbsp &nbsp - unkonsistente Definitionen in versch. Reihen |
   
|  <h4>Data Understanding</h4>  |
| --- |
|  &nbsp &nbsp &nbsp How: |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Numerical: Histograms & Scatter Plots |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Categorical: Bar Plots |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Both: [[Boxplot]], violin Plots, colored histograms |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Date/Time: Line Plots |
|  &nbsp &nbsp &nbsp Why: |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Outliers |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Different factors contributing to larger  phenomena |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Correlations |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Does the data change over time? |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Constrasting values |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Are there hierarchies in the data? |
	

|  <h4>[[Sampling]]</h4>  |
| --- |
|  &nbsp &nbsp &nbsp Von verschiedenen Ausprägungsgruppen sollten ähnlich viele Daten vorhanden sein. Es gibt verschiedene Verfahren, mit denen man Verzerrungen "auffangen" kann |
|  &nbsp &nbsp &nbsp Nicht balanciertes Datenset bei Klassifikation? => [[SMOTE]]. Hat Auswirkungen auf die Auswahl von Performance-Kennzahlen |


|  <h4>[[Feature Engineering]]</h4>  |
| --- |
|  &nbsp &nbsp &nbspGibt es wichtige Features, die vielleicht gar nicht in den Daten vorhanden sind? Kann zu Verzerrungen führen |
|  &nbsp &nbsp &nbsp Wie wird mit fehlenden Werten umgegangen => Impute ? |
|  &nbsp &nbsp &nbsp [[Scaling]]  |
|  &nbsp &nbsp &nbsp Dimension Reduction - viele Features zu wenigen relevanten Features zusammenfassen |
 |  &nbsp &nbsp &nbsp &nbsp &nbsp - PCA |
 |  &nbsp &nbsp &nbsp &nbsp &nbsp - Embedding |
  |   &nbsp &nbsp &nbsp  [[Feature Creation]] |
 |   &nbsp &nbsp &nbsp [[Feature Selection]] |
  |   &nbsp &nbsp &nbsp [[Text Feature Engineering]] |
 

|  <h4>Training</h4>  |
| --- |
|   &nbsp &nbsp &nbsp Split in Training und Test Set |
|  &nbsp &nbsp &nbsp &nbsp &nbsp - zwischen 80 und 90 Prozent |
| &nbsp &nbsp &nbsp  [[Cross Validation]] |
| &nbsp &nbsp &nbsp Modellauswahl |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Predicting numbers - Regression]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Predicting categories - Classfication]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Categorical Features]]? |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[NA Values]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Explainable]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - Trainings- / Vorhersagege- [[Speed]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[NLP or Text]] |

|  <h4>[[Feature Selection]]</h4>  |
| --- |
    
|  <h4>[[Hyperparameter Tuning]]</h4>  |
| --- |

|  <h4>Evaluation</h4>  |
| --- |
| &nbsp &nbsp &nbsp Auswahl geeigneter Performance Kennzahlen |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Numerical]] |
| &nbsp &nbsp &nbsp &nbsp &nbsp - [[Categorical]] |
| &nbsp &nbsp &nbsp [[Overfitting]] prüfen |








