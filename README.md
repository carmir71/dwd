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

Ein paar Data Preperation-Ansätze habe ich noch mit reingepackt.

Update am 20.01.2020: Anpassung der FTP-Verbindungsdaten des DWD
