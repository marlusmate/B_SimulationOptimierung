#Bibs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("Bibliothken erfolgreich Importiert\n")



#Kontext, Labels
drop_col = ['exp', 'Slugflow', 'AirFlow', 'WaterFlow']
#Sensordata
col = ['P4-B14', 'P5-B13','P6-B12','P7-B11','P8-B10','P13-B20','P14-B08','P15-B09','P17-B05']
colx = ['P4-B14', 'P5-B13','P6-B12','P7-B11','P8-B10','P13-B20','P14-B08','P15-B09','P17-B05','Slugflow', 'id', 'freq']


# FFt
def fft(f_in, n):
    Fs = 5000  # Sampling Rate, Abtastrate 5kHz [1/s]
    tstep = 1 / Fs  # Zeit zwischen Abtastung [s]
    f0 = []  # ? Signal Frequenz, unbekannt?
    # n = 1 #Betrachtungsdauer [s]
    t = np.arange(0, n, tstep)  # Messzeitpunkte

    ns = len(t)  # Anzahl Samples [-]
    L = np.arange(1, np.floor(ns / 2), dtype='int')

    fhat = np.fft.fft(f_in)
    fhat = np.hstack((np.zeros(1), fhat[L]))
    # FastFourierTransf., ergibt n=len(L) komplexe Fourierkoeffizienten [-],
    # Magnitude(Gewichtigkeit/betrag) & Phase(sin/cos Verhältnis)  (?)
    # die (jeweils) aufsummiert werden müssten um das vorgegebene Signal darzustellen
    # (Deswegen brauchen wir auch mindestens n Frequenzen  (rekonstruktion))

    psd = fhat * np.conj(fhat) / ns
    psd = np.hstack((np.zeros(1), psd[L]))
    # Berechnung Power Spectrum (Leistungsdichte über dem Spektrum der Freq)
    # "fhat*np.conj(fhat)" ergibt Quadrat des Betrages des img Vektors (jeder Zeile/Frequenz)
    # Gesamte Leistung des Signal über die Bandbreite der n Frequenzen verteilt ("/ns")
    # (Das Integral dieses Graphen ergibt demnach wieder die Leistung des Signal)
    # Komplex konjug. des Vektor ergibt Stärke der jeweiligen Freq. [W?]

    mag = np.abs(fhat) / ns
    mag = 2 * mag[0:int(ns / 2)]
    mag[0] = mag[0] / 2

    freq = (1 / (tstep * ns)) * np.arange(ns)
    freq = np.hstack((np.zeros(1), freq[L]))
    # Berechnung des zum psd-vektor korrespondierenden Frequenz-vektor

    # post process
    f_out = np.array([fhat, psd, mag, freq])
    f_out = pd.DataFrame(f_out.transpose(), columns=['fhat', 'psd', 'mag', 'freq'])

    return f_out


def shorten(c_in, ind_in, exp_in, rows, method):
    Fs = 1 / 5000
    j = 0
    k = 0
    l = len(c_in) % rows
    c_in = c_in[:len(c_in) - l]
    ind_in = ind_in[:len(c_in) - l]
    exp_in = exp_in[:len(c_in) - l]

    # d_out = pd.DataFrame([], columns= ['SensorD','Slugflow', 'id'] )
    if not "fft" in method:
        c_f = np.ones(int(len(c_in) / rows))
        ind_f = np.ones(int(len(ind_in) / rows))
        id_f = np.ones(int(len(exp_in) / rows))

        for i in range(0, len(c_in), rows):

            if len(set(exp_in.iloc[i:(i + rows)])) == 1:
                if method == "mean":
                    c_f[j] = np.mean(c_in[i:(i + rows)])
                if method == "median":
                    c_f[j] = np.median(c_in[i:(i + rows)])
            else:
                print(k, 'te Überschneidung zw. 2 exp übergangen')
                k = k + 1
            ind_f[j] = ind_in[i]
            id_f[j] = j
            j = j + 1
        d_temp = np.array([c_f, ind_f, id_f])
        d_out = pd.DataFrame(d_temp.transpose(), columns=['value', 'Slugflow', 'id'])

    if "fft" in method:
        k = 0
        ue = 1
        rh = rows // 2
        samplingrate = 1 / 5000
        t = samplingrate * rows
        # np.arange(0,n,tstep)
        ar_temp = np.empty((len(c_in)//rows, rh))
        sl_ind = np.ones((len(c_in)//rows, 1))


        for i in range(0, len(c_in), rows):
            if len(set(exp_in.iloc[i:(i + rows)])) == 1:
                if "mag" in method:
                    temp = fft(c_in[i:i + rows], t)
                    ar_temp[j] = temp['mag'].transpose()
                    sl_ind[j] = ind_in[i]

                elif "psd" in method:
                    temp = fft(c_in[i:i + rows], t)
                    ar_temp[j][0] = temp['psd'].transpose()
                    sl_ind[j] = ind_in[i]*sl_ind[j]

                j = j + 1
                k = k + rh

            else:
                print(ue, 'te Überschneidung zw. 2 exp übergangen')
                ue = ue + 1
                continue

        d_out = ar_temp[:j]
        label = sl_ind[:j]

    return d_out, label

def shorten_df(df_in, ind_in, exp_in, rows, method):
    dtest, ltest = shorten(df_in[col[0]], ind_in, exp_in, rows, method)
    ftest = dtest.shape
    df_f = np.empty((len(ltest), len(col), ftest[1]))
    for i in range(0,len(col)):
        ctemp = df_in[col[i]]
        xr, xl = shorten(ctemp, ind_in, exp_in, rows, method) #tupel?
        df_f[:,i,:] = xr

    return df_f, ltest
