# dwd
DWD-Skript zum Kurs Web and Social Media Analytics (WS 18/19)

Der DWD (Deutscher Wetterdienst) stellt über seinen FTP (https://www.dwd.de/DE/leistungen/cdcftpmesswerte/cdcftpmesswerte.html?nn=17626) eine Menge Wetterdaten zu DWD-eigenen oder Partnerstationen zur freien Verfügung.

Ich habe mir die täglichen Mittelwerte der DWD-eigenen und Partnerstationen vorgenommen.

Dazu habe ich ein Skript geschrieben, mit dem ich

* die Stationsdaten der DWD-eigenen Stationen, 
* die Stationsdaten der Partnerstationen,
* die darauf aufbauenden Wetterdaten

vom FTP herunterlade.

Die unterschiedlichen Datenformate wandle ich dann maschinenlesbar um und führe alle Daten zusammen.

Alle Funktionen sind der Übersicht halber in der 0_funktionen.py ausgelagert.

# Eingesetzte Module

## Im Hauptskript

* *import* datetime *as dt*
* *import* itertools
* *import* pandas *as pd*
* *import* numpy *as np*
* *import* pandas_profiling *as pp*
* *import* matplotlib.pyplot *as plt*
* *import* seaborn *as sns*

* *import* statsmodels.api *as sm*
* *import* statsmodels.formula.api *as smf*
* *import* statsmodels.tsa.api *as smt*
* *from statsmodels.graphics.regressionplots import* plot_leverage_resid2

* *import* funktionen_0 as f0

## In der funktionen_0
(über die bereits oben aufgeführten hinaus)

* import* os, shutil
* import* zipfile *as zip*
*import* re
*import* random
*import* warnings

* *from bs4 import* BeautifulSoup
* import* requests
* from ftplib import* FTP
