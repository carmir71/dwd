# Fallstudie Webcrawler-gestütztes (Preis-)Abschriftenmanagement im stationären Einzelhandel
# Carsten Mirgeler, Erim Kansoy und Petermax Fricke
# Hochschule der Medien Stuttgart
# Seminar: Web and Social Media Analytics

# Benötigte Module laden
## Übliche Module
import datetime as dt
import itertools
import pandas as pd
import numpy as np
import pandas_profiling as pp
import os, shutil
import zipfile as zip
import re
import random
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

## Statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

## Crawler im weiteren Sinne
from bs4 import BeautifulSoup
import requests
from ftplib import FTP

# Angabe eines headers, da der Server sonst den request ablehnen könnte
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


# Definition der Tage für die Wettervorhersage
d0 = dt.date.today()
d1 = d0 + dt.timedelta(days=1)
d2 = d0 + dt.timedelta(days=2)
d3 = d0 + dt.timedelta(days=3)


# Daten als CSV wegschreiben
def save_csv(quelle, name):
    dataframe = quelle
    dataframe.to_csv(name + ".csv", index=0)
    dataframe = pd.read_csv(name + ".csv")
    print('Der Datensatz wurde erfolgreich gespeichert')
    print("Er umfasst: ",dataframe.shape[0],"Zeilen und ",dataframe.shape[1],"Spalten")
    return dataframe.head()


def Beschreibung_Stationen1():
    file = 'KL_Tageswerte_Beschreibung_Stationen.txt'
    
    ftp = FTP('ftp-cdc.dwd.de')
    ftp.login()
    ftp.cwd('pub/CDC/observations_germany/climate/daily/kl/recent/')
    
    list1 = ftp.nlst()
    
    if file in list1:
        ftp.retrbinary('RETR ' + file, open('rückblick/' + file, 'wb').write)
        print('Beschreibung der Stationen vom FTP heruntergeladen und abgelegt')
    else:
        print('Beschreibung der Stationen vom FTP nicht verfügbar')
    
    ftp.quit()


def Beschreibung_Stationen2():
    file = 'KL_Standardformate_Beschreibung_Stationen.txt'
    
    ftp = FTP('ftp-cdc.dwd.de')
    ftp.login()
    ftp.cwd('pub/CDC/observations_germany/climate/subdaily/standard_format/')
    
    list1 = ftp.nlst()
    
    if file in list1:
        ftp.retrbinary('RETR ' + file, open('rückblick/' + file, 'wb').write)
        print('Beschreibung der Stationen vom FTP heruntergeladen und abgelegt')
    else:
        print('Beschreibung der Stationen vom FTP nicht verfügbar')
    
    ftp.quit()
    

def Stationen_importieren1(von_datum):
    list1 = []
    
    file = 'KL_Tageswerte_Beschreibung_Stationen.txt'
    
    with open('rückblick/' + file, 'r', encoding='iso-8859-1') as f:
        imp = f.read()
        
    imp = str(imp).split('\n')
        
    for x in imp:
        y = re.split('\s+', x)
        list1 += [y]
    
    df = pd.DataFrame.from_dict(list1)
    # Erste Zeile enthält die Spaltenbezeichnungen
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    
    # Spalten-Bereinigung beim Import
    # Stationsnamen und Bundesländer wurden in mehrere Spalten getrennt
    #df = df.replace([None], [' '], regex=True)
    df = df.replace(np.NaN, ' ', regex=True)
    
    df = df.rename(columns={df.columns.get_values()[8]: 'a1'})
    cols=pd.Series(df.columns)
    for dup in df.columns.get_duplicates(): 
        cols[df.columns.get_loc(dup)] = [dup+'.'+str(d_idx) if d_idx!=0 else dup for d_idx in range(df.columns.get_loc(dup).sum())]
    df.columns=cols
    df['ColumnA'] = df[df.columns[6:]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df = df.drop(df.columns[6:12], axis=1)
    
    # Zeilen-Bereinigung beim Import
    df = df.drop([0]) # Erste Zeile (Trennzeichen) löschen
    df = df.drop([1180]) # Letzte Zeile (Leerzeile) löschen
    
    # Datum umwandeln
    df['von_datum2'] = pd.to_datetime(df['von_datum'], format='%Y%m%d')
    df['bis_datum2'] = pd.to_datetime(df['bis_datum'], format='%Y%m%d')
    df = df.drop(df.columns[1:3], axis=1)
    df['von_datum'] = df['von_datum2']
    df['bis_datum'] = df['bis_datum2']
    df = df.drop(df.columns[5:7], axis=1)
    
    # Stationsname und Bundesland trennen (Reste aus der Spalten-Bereinigung)
    df[['Stationsname','Bundesland']] = df['ColumnA'].str.rsplit(expand=True, n=1)
    df = df.drop(['ColumnA'], axis=1)
    
    # Nur aktuelle Stationen beibehalten
    df = df.loc[df['bis_datum'] >= von_datum]
    
    print('Beschreibung der Stationen vom FTP importiert')

    return df 


def Stationen_importieren2(von_datum):
    list1 = []
    
    file = 'KL_Standardformate_Beschreibung_Stationen.txt'
    
    with open('rückblick/' + file, 'r', encoding='iso-8859-1') as f:
        imp = f.read()
        
    imp = str(imp).split('\n')
        
    for x in imp:
        y = re.split('\s+', x)
        list1 += [y]
    
    df = pd.DataFrame.from_dict(list1)
    # Erste Zeile enthält die Spaltenbezeichnungen
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    
    df = df.drop(df.index[81:], axis=0)
    df = df.drop(df.index[0], axis=0)
    
    #df = df.replace([None], [' '], regex=True)
    df = df.replace(np.NaN, ' ', regex=True)
    
    df = df.rename(columns={df.columns.get_values()[9]: 'a1'})
    df = df.rename(columns={df.columns.get_values()[10]: 'a1'})
    df = df.rename(columns={df.columns.get_values()[11]: 'a1'})
    cols=pd.Series(df.columns)
    for dup in df.columns.get_duplicates(): 
        cols[df.columns.get_loc(dup)] = [dup+'.'+str(d_idx) if d_idx!=0 else dup for d_idx in range(df.columns.get_loc(dup).sum())]
    df.columns=cols
    df['ColumnA'] = df[df.columns[7:]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df = df.drop(df.columns[7:12], axis=1)
    
    df['von_datum'] = pd.to_datetime(df['von'], format='%Y%m%d')
    df['bis_datum'] = pd.to_datetime(df['bis'], format='%Y%m%d')
    df = df.drop(df.columns[2:4], axis=1)
    
    # Stationsname und Bundesland trennen (Reste aus der Spalten-Bereinigung)
    df[['Stationsname','Bundesland']] = df['ColumnA'].str.rsplit(expand=True, n=1)
    df = df.drop(['ColumnA'], axis=1)
    
    print('Beschreibung der Stationen vom FTP importiert')

    return df


def Tageswerte_downloaden(stations_ids): 
    list1, list2, list3, list4 = [], [], [], []
    
    d0 = dt.date.today()
    pfad = 'rückblick/' + str(d0)
    os.makedirs(pfad, exist_ok=True)
    
    # Verbindung aufbauen
    
    ftp = FTP('ftp-cdc.dwd.de')
    ftp.login()
    ftp.cwd('pub/CDC/observations_germany/climate/daily/kl/recent/')
    
    # Verzeichnis auslesen
    list1 = ftp.nlst()

    for x in stations_ids:
        file1 = 'tageswerte_KL_' + x + '_akt.zip'

        if file1 in list1:
            ftp.retrbinary('RETR ' + file1, open(pfad + '/' + file1, 'wb').write)
        else:
            list3 += [x]
    
    ftp.quit()
    
    # Verbindung trennen und neu aufbauen
    
    ftp = FTP('ftp-cdc.dwd.de')
    ftp.login()
    ftp.cwd('pub/CDC/observations_germany/climate/subdaily/standard_format/')
    
    #Verzeichnis auslesen
    list2 = ftp.nlst()
    
    for x in list3:
        file2 = 'kl_' + x + '_00_akt.txt'
    
        if file2 in list2:
            ftp.retrbinary('RETR ' + file2, open(pfad + '/' + file2, 'wb').write)
        else:
            list4 += [x]
    
    ftp.quit()
    
    # Verbindung trennen
    for x in list4:
        print(x + ' wurde auf dem FTP nicht gefunden')
    
    print('Alle anderen Tageswerte heruntergeladen und abgelegt')


def Tageswerte_entpacken(datum):    
    pfad = 'rückblick/' + datum
    pfad_entpackt = 'rückblick/' + datum + '/entpackt'

    if os.path.isdir(pfad_entpackt) == False:
        os.mkdir(pfad_entpackt)

    for file in os.listdir(pfad):
        if file.endswith('.zip'):
            with zip.ZipFile(pfad + '/' + file, 'r') as zip_ref:
                list1 = zip_ref.namelist()
                for file in list1:
                    if file.startswith('produkt_klima_tag_'):
                        zip_ref.extract(member=file, path=pfad)
                    else:
                        pass
                zip_ref.close()
        else:
            pass

        if file.endswith('.txt'):
            shutil.move(pfad + '/' + file, pfad_entpackt + '/' + file)
        else:
            pass
    
    print('Alle Tageswerte zum Datum ' + datum + ' im gleichnamigen Verzeichnis entpackt oder verschoben')


def Tageswerte_importieren1(datum, ab_datum, ausgabe):  
    filelist = []
    
    dataframe_temp = pd.DataFrame()
    dataframe = pd.DataFrame()
    
    pfad_entpackt = 'rückblick/' + datum + '/entpackt'
    
    for file in os.listdir(pfad_entpackt):
        if file.startswith('produkt_klima_tag_') and file.endswith('.txt'):
            filelist += [file]
    
    for file in filelist:
        with open(pfad_entpackt + '/' + file, 'r', encoding='iso-8859-1') as f:
            imp = f.read()
        imp = str(imp).split('\n')
            
        dataframe_temp = pd.DataFrame.from_dict(imp)
        dataframe_temp = dataframe_temp.replace(' ', '', regex=True)
        dataframe_temp = dataframe_temp[0].str.split(';', expand=True)
            
        headers = dataframe_temp.iloc[0]
        dataframe_temp = pd.DataFrame(dataframe_temp.values[1:], columns=headers)
            
        dataframe_temp = dataframe_temp.dropna()
        dataframe_temp['MESS_DATUM2'] = pd.to_datetime(dataframe_temp['MESS_DATUM'], format='%Y%m%d')
        dataframe_temp = dataframe_temp.drop(['MESS_DATUM', 'QN_3', 'QN_4', 'eor'], axis=1)
        dataframe_temp = dataframe_temp.rename({'MESS_DATUM2': 'MESS_DATUM'}, axis=1)
        dataframe_temp = dataframe_temp.loc[dataframe_temp['MESS_DATUM'] >= ab_datum]
        dataframe_temp['MESS_DATUM'] = dataframe_temp['MESS_DATUM'].astype('datetime64[D]')
        dataframe_temp['STATIONS_ID'] = dataframe_temp['STATIONS_ID'].astype('category')
        dataframe_temp['RSKF'] = dataframe_temp['RSKF'].astype('category')
        dataframe_temp['NM'] = dataframe_temp['NM'].astype('category')
        
        if ausgabe == 1:
            print(file, len(dataframe_temp))
        
        dataframe = dataframe.append(dataframe_temp)
        dataframe_temp = pd.DataFrame()

    print('Tageswerte ab dem ' + ab_datum + ' in das DataFrame importiert')
        
    return dataframe


def Tageswerte_importieren2(datum, ab_datum, ausgabe):
    # Anhand des KX-Formats der Datensatzbeschreibung ergibt sich aus dem importierten String mit 288 Zeichen eine 
    # Reihenfolge, nach der ab einer bestimmten Zeichenzahl ein Split durchgeführt werden muss
    split = [2, 7, 11, 13, 15, 19, 24, 25, 30, 31, 36, 37, 42, 43, 47, 48, 52, 53, 56, 57, 61, 62, 63, 67, 68, 72, 
             73, 77, 78, 82, 83, 87, 88, 89, 93, 94, 95, 99, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 
             121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 147, 149, 150, 152, 154, 155, 157, 159, 
             160, 163, 164, 166, 167, 169, 170, 172, 173, 175, 176, 178, 179, 181, 182, 184, 185, 187, 188, 190, 191, 
             194, 195, 198, 199, 200, 202, 203, 205, 206, 208, 209, 211, 212, 214, 215, 217, 218, 220, 221, 223, 224, 
             226, 227, 231, 232, 233, 237, 238, 239, 243, 244, 245, 249, 250, 251, 254, 255, 256, 259, 260, 261, 264, 
             265, 269, 270, 275, 276, 281, 282]
    list2, list3, list4 = [], [], []
    cols = ['KE','ST','JA','MO','TA','--','P1','Q','P2','Q','P3','Q','PM','Q','TXK','Q','TNK','Q','TRK','Q','TGK','S',
            'Q','T1','Q','T2','Q','T3','Q','TMK','Q','TF1','ETF1','Q','TF2','ETF2','Q','TF3','ETF3','Q','VP1','Q','VP2',
            'Q','VP3','Q','VPM','Q','UP1','Q','UP2','Q','UP3','Q','UPM','Q','UR1','Q','UR2','Q','UR3','Q','D1','FK1','Q',
            'D2','FK2','Q','D3','FK3','Q','FMK','Q','N1','Q','C1','Q','W1','Q','N2','Q','C2','Q','W2','Q','N3','Q','C3',
            'Q','W3','Q','NM','Q','SDK','SDJ','Q','V1','Q','V2','Q','V3','Q','E1','Q','E2','Q','E3','Q','VAK','Q','VBK',
            'Q','VCK','Q','R1','RF1','Q','R2','RF2','Q','R3','RF3','Q','RSK','RSKF','Q','SHK','SA','Q','NSH','NSHJ','Q',
            'FXK','Q','ASH','Q','WAAS','Q','WASH','Q']
    filelist = [] #['kl_10147_00_akt.txt']
    
    dataframe_temp = pd.DataFrame()
    dataframe = pd.DataFrame()
    
    pfad_entpackt = 'rückblick/' + datum + '/entpackt'
    
    for file in os.listdir(pfad_entpackt):
        if file.startswith('kl_') and file.endswith('_00_akt.txt'):
            filelist += [file]
    
    for file in filelist:
        with open(pfad_entpackt + '/' + file, 'r', encoding='iso-8859-1') as f:
            imp = f.read()
        imp = imp.split('\n')

        for y in imp:
            j = 0
            list2 = []
            for x in split:
                list2 = y[j:x]
                j = x
        
                list3 += [list2]
        
            list4 += [list3]
            list3 = []
            
        dataframe_temp = pd.DataFrame.from_dict(list4)
        list4 = []
        dataframe_temp = dataframe_temp.append(pd.Series(cols, index=dataframe_temp.columns ), ignore_index=True)
        headers = dataframe_temp.iloc[-1]
        dataframe_temp = pd.DataFrame(dataframe_temp.values[0:], columns=headers)
        dataframe_temp = dataframe_temp.drop(dataframe_temp.index[-1], axis=0)
        dataframe_temp = dataframe_temp.drop(['Q'], axis=1)
        stations_id = str(file).split('_')[2]
        dataframe_temp['ST'] = file.replace('kl_', '').replace('_00_akt.txt', '')
        dataframe_temp['MESS_DATUM'] = dataframe_temp['JA'] + '-' + dataframe_temp['MO'] + '-' + dataframe_temp['TA']
        dataframe_temp = dataframe_temp.loc[dataframe_temp['MESS_DATUM'] >= ab_datum]
        
        dataframe = dataframe.append(dataframe_temp)
        dataframe.drop_duplicates(inplace=True)
        if ausgabe == 1:
            print(file, len(dataframe_temp))
        dataframe_temp = pd.DataFrame()
        
    print('Tageswerte ab dem ' + ab_datum + ' in das DataFrame importiert')
        
    return dataframe


def Tageswerte2_transformieren(dataframe):
    # subdaily-DataFrame anpassen
    # auf die Spalten in subdaily_columns einschränken
    subdaily_columns = ['ST', 'FXK', 'FMK', 'RSK', 'RSKF', 'SDK', 'SHK', 'NM', 
                        'VPM', 'PM', 'TMK', 'UPM', 'TXK', 'TNK', 'TGK', 'MESS_DATUM']
    daily_columns = ['STATIONS_ID', 'FX', 'FM', 'RSK', 'RSKF', 'SDK', 'SHK_TAG', 'NM', 
                     'VPM', 'PM', 'TMK', 'UPM', 'TXK', 'TNK', 'TGK', 'MESS_DATUM']
        
    dataframe = dataframe[subdaily_columns]
    # Spaltenüberschriften aus den daily_columns übernehmen
    dataframe.columns = daily_columns
        
    to_change = [col for col in dataframe.columns if col not in ['MESS_DATUM', 'STATIONS_ID']]
    
    for col in to_change:
        dataframe[col] = dataframe[col].astype('float64')
    dataframe['MESS_DATUM'] = dataframe['MESS_DATUM'].astype('datetime64[D]')
    dataframe['STATIONS_ID'] = dataframe['STATIONS_ID'].astype('category')
    dataframe['RSKF'] = dataframe['RSKF'].astype('category')

    dataframe = dataframe.reset_index().drop('index', axis=1)
    
    dataframe.loc[dataframe['FX'] != -99, 'FX'] = dataframe['FX'] * 0.1
    dataframe.loc[dataframe['FM'] != -99, 'FM'] = (0.836 * dataframe['FM'] * dataframe['FM']**0.5) * 0.1
    dataframe.loc[dataframe['RSK'] != -999, 'RSK'] = dataframe['RSK'] * 0.1
    dataframe.loc[dataframe['SDK'] != -99, 'SDK'] = dataframe['SDK'] *0.1
    dataframe.loc[dataframe['NM'] != -99, 'NM'] = dataframe['NM'] * 0.1
    dataframe.loc[dataframe['VPM'] != -99, 'VPM'] = dataframe['VPM'] * 0.1
    dataframe.loc[dataframe['PM'] != -9999, 'PM'] = dataframe['PM'] * 0.1
    dataframe.loc[dataframe['TMK'] != -999, 'TMK'] = dataframe['TMK'] * 0.1
    dataframe.loc[dataframe['TXK'] != -999, 'TXK'] = dataframe['TXK'] * 0.1
    dataframe.loc[dataframe['TNK'] != -999, 'TNK'] = dataframe['TNK'] * 0.1
    dataframe.loc[dataframe['TGK'] != -999, 'TGK'] = dataframe['TGK'] * 0.1
        
    return dataframe


def Tageswerte_zusammenlegen(subdaily, daily):
    # daily-DataFrame anpassen
    to_change = [col for col in daily.columns if col not in ['MESS_DATUM', 'STATIONS_ID']]
    
    for col in to_change:
        daily[col] = daily[col].astype('float64')
    daily['MESS_DATUM'] = daily['MESS_DATUM'].astype('datetime64[D]')
    daily['STATIONS_ID'] = daily['STATIONS_ID'].astype('category')
    daily['RSKF'] = daily['RSKF'].astype('category')
    daily['NM'] = daily['NM'].astype('category')
    
    # DataFrames zusammenführen
    dataframe = subdaily.append(daily)
    dataframe['STATIONS_ID'] = dataframe['STATIONS_ID'].astype('category')
    dataframe = dataframe.reset_index().drop('index', axis=1)
    dataframe.drop_duplicates(inplace=True)
    
    return dataframe


def Stationsdaten_loeschen(dataframe, days):
    filelist = []
    pfad = 'rückblick/'
    
    last_days = dataframe['bis_datum'].max() - dt.timedelta(days)

    filelist = [f for f in os.listdir(pfad) if os.path.isdir(os.path.join(pfad, f))]
    
    j = 0
    for i in filelist:
        if i < str(last_days):
            shutil.rmtree(pfad + '/' + i)
            print('Das Verzeichnis ', i, ' wurde gelöscht.')
            j = j + 1
    
    if j == 0:
        print('Es wurde kein Verzeichnis gelöscht.')
            
    return


def best_formula(dataframe, response):
    remaining = set(dataframe.columns)
    remaining.remove(response)
    selected, results = [], []
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            formula = '{} ~ {}'.format(response, ' + '.join(selected + [candidate]))
            lm = smf.gls(formula, sm.add_constant(dataframe)).fit()
            score = lm.rsquared_adj
            scores_with_candidates.append((score, candidate, [formula, lm.rsquared_adj, lm.ssr]))
        scores_with_candidates.sort()
        best_score, best_candidate, best_metrics = scores_with_candidates.pop()
        results.append(best_metrics)
        remaining.remove(best_candidate)
        selected.append(best_candidate)
    dataframe = pd.DataFrame(results)
    dataframe.columns = ['formula', 'adjr2', 'ssr']
    dataframe = dataframe.sort_values('adjr2', axis=0, ascending=False)[:1].reset_index()
    formula = str(dataframe['formula'].values).replace('[', '').replace(']', '').replace("'", '')
    
    return formula, dataframe


def df_infos(dateframe):    
    result=pd.DataFrame()
    result["Columns"]=[x for x in dateframe.columns]
    result["type"]=[x for x in dateframe.dtypes]
    result["Unique values count"]=[dateframe.groupby(x)[x].count().shape[0] for x in dateframe.columns]
    result["Zeros"]=[sum(dateframe[x] == 0) for x in dateframe.columns]
    result["NaNs"]=[sum(dateframe[x] == np.NaN) for x in dateframe.columns]
    result['Sample']=[dateframe[x][10] for x in dateframe.columns]
    
    return result


def rnd_stations(dataframe, anzahl):
    dataframe_temp1 = pd.DataFrame()
    dataframe_temp2 = pd.DataFrame()
    
    list1 = dataframe['STATIONS_ID'].unique()
    random_items = random.choices(population=list1, k=anzahl)
    
    for i in random_items:
        dataframe_temp2 = dataframe.loc[dataframe['STATIONS_ID'] == i]
        dataframe_temp1 = dataframe_temp1.append(dataframe_temp2)
        
    return dataframe_temp1


def zeitraeume_ergaenzen(dataframe):
    df_rest = dataframe.copy()

    list1 = list(set(dataframe['STATIONS_ID']))
    list2 = []
    df_temp = pd.DataFrame()
    df_asdf = pd.DataFrame()
    madate = max(df_rest['MESS_DATUM'])
    midate = min(df_rest['MESS_DATUM'])
    dedate = int(str(madate - midate).replace(' days 00:00:00', '')) + 1
    maxdate = df_rest.copy()
    maxdate.index = pd.DatetimeIndex(df_rest['MESS_DATUM']).floor('D')

    dates = pd.date_range(midate, madate)

    for s in list1:
        df_temp = dataframe.loc[dataframe['STATIONS_ID'] == s]
        if df_temp.shape[0] < dedate:
            list2 += [s]

    df_temp = pd.DataFrame()
    for t in list2:
        #print('start', df_rest.shape)
        df_drop = dataframe.loc[dataframe['STATIONS_ID'] == t]
        df_rest = df_rest.drop(df_drop.index, axis=0)
        #print('after drop', df_rest.shape)
        df_temp = dataframe.loc[dataframe['STATIONS_ID'] == t]
        a = df_temp.shape
        #print('len station original', t, a)
        df_temp.index = pd.DatetimeIndex(df_temp['MESS_DATUM']).floor('D')
        #print(df_asdf.index)
        all_days = pd.date_range(df_temp.index.min(), maxdate.index.max(), freq='D')
        #print(all_days)
        df_temp = df_temp.loc[all_days]
        df_temp['STATIONS_ID'] = t
        df_temp['MESS_DATUM'] = df_temp.index
        df_temp = df_temp.reset_index().drop('index', axis=1)
        b = df_temp.shape
        #print('len station new', t, b)
        #print('delta', t, b[0] - a[0])
        df_asdf = df_asdf.append(df_temp)
        #print('sum end', df_asdf.shape)
        #print('----------')

    df = df_rest.append(df_asdf)

    madate = max(df['MESS_DATUM'])
    midate = min(df['MESS_DATUM'])
    dedate = int(str(madate - midate).replace(' days 00:00:00', '')) + 1
    anzahl_stat = len(df['STATIONS_ID'].unique())
    anzahl_dsatz = len(df)
    anzahl_dsoll = anzahl_stat * dedate
    anzahl_dfehl = anzahl_dsoll - anzahl_dsatz

    if anzahl_dfehl == 0:
        print('Alle fehlenden Werte (Zeitäume) wurden ergänzt')
    else:
        print('Nicht alle fehlenden Werte (Zeitäume) wurden ergänzt')
        
    return df


def fehlwerte_ermitteln(dataframe):
    df_temp = dataframe.copy()
    
    df_temp = df_temp.replace(-9999, np.NaN).replace(-999, np.NaN).replace(-99, np.NaN)
    count = df_temp.isnull().values.sum()

    print('Gesamtzahl aller Fehlwerte: ', count)

    return df_temp


def einzelne_fehlwerte_ersetzen(dataframe):
    df_temp = dataframe.copy()
    list1 = list(set(dataframe['STATIONS_ID']))
    list2 = list(set(dataframe['MESS_DATUM']))
    dft1 = pd.DataFrame()
    dft2 = pd.DataFrame()
    dft3 = pd.DataFrame()
    
    df_temp = df_temp.replace(-9999, np.NaN).replace(-999, np.NaN).replace(-99, np.NaN)
    df_temp = df_temp.sort_values(by=['STATIONS_ID', 'MESS_DATUM'])
    
    for s in list1:
        dft1 = df_temp.loc[df_temp['STATIONS_ID'] == s]
        dft1 = dft1.fillna(method='pad')
        dft1 = dft1.fillna(method='backfill')
        dft2 = dft2.append(dft1)
    
    dft2 = dft2.sort_values(by=['MESS_DATUM', 'STATIONS_ID'])
    dft1 = pd.DataFrame()
        
    for t in list2:
        dft1 = dft2.loc[dft2['MESS_DATUM'] == t]
        dft1 = dft1.fillna(method='pad')
        dft1 = dft1.fillna(method='backfill')
        dft3 = dft3.append(dft1)
            
    dft3 = dft3.sort_values(by=['STATIONS_ID', 'MESS_DATUM'])
    
    dft3.drop_duplicates(inplace=True)
    dft3['STATIONS_ID'] = dft3['STATIONS_ID'].astype('int64')
    dft3['NM'] = dft3['NM'].astype('category')
    
    print('Einzele Fehlwerte wurden ersetzt')
    
    return dft3


def rest_fehlwerte_ersetzen(dataframe):
    df_temp = dataframe.copy()
    
    df_temp = df_temp.replace(np.NaN, 0)
    
    return df_temp

