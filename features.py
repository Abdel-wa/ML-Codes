# features.py
# (prétraitement EDA, segmentation 5s, features, XGBoost SA_Detection.json)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.signal import butter, find_peaks
from scipy.stats import linregress


try:
    # Le notebook utilise cvxEDA (fichier cvxEDA.py)
    from cvxEDA import cvxEDA
except Exception as e:  # pragma: no cover
    cvxEDA = None
    _cvxeda_import_error = e


# ----------------------------
# Helpers (du notebook)
# ----------------------------
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    # Normalization of the cutoff signal
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter_filtfilt(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def decomposition(eda, Fs=4) -> dict:
    y = np.array((eda))
    yn = (y - y.mean()) / y.std()
    if cvxEDA is None:
        raise ImportError(f"cvxEDA is required but could not be imported: {_cvxeda_import_error}")
    [r, p, t, l, d, e, obj] = cvxEDA(yn, 1./Fs)

    return {
        "phasic components": np.array(p).ravel(),
        "tonic component": np.array(t).ravel(),
        "driver": np.array(r).ravel(),
        "lameda": np.array(l).ravel(),
        "disturbances": np.array(d).ravel(),
        "error": np.array(e).ravel(),
        "objective": np.array(obj).ravel()
    }


# Constante du notebook
SAMPLE_RATE = 4

def findPeaks(data, offset, start_WT, end_WT, thres=0, sampleRate=SAMPLE_RATE):
    '''
        This function finds the peaks of an EDA signal and returns basic properties.
        Also, peak_end is assumed to be no later than the start of the next peak.
        
        ********* INPUTS **********
        data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
        offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak
        start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
        end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
        thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
        sampleRate:  number of samples per second, default=8
        
        ********* OUTPUTS **********
        peaks:               list of binary, 1 if apex of SCR
        peak_start:          list of binary, 1 if start of SCR
        peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
        peak_end:            list of binary, 1 if rec.t/2 of SCR
        peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
        amplitude:           list of floats,  value of EDA at apex - value of EDA at start
        max_deriv:           list of floats, max derivative within 1 second of apex of SCR
    '''
    
    EDA_deriv = data['EDA_Phasic'][1:].values - data['EDA_Phasic'][:-1].values
    peaks = np.zeros(len(EDA_deriv))
    peak_sign = np.sign(EDA_deriv)
    for i in range(int(offset), int(len(EDA_deriv) - offset)):
        if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
            peaks[i] = 1
            for j in range(1, int(offset)):
                if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                    peaks[i] = 0
                    break

    # Finding start of peaks
    peak_start = np.zeros(len(EDA_deriv))
    peak_start_times = [''] * len(data)
    max_deriv = np.zeros(len(data))
    rise_time = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            temp_start = max(0, i - sampleRate)
            max_deriv[i] = max(EDA_deriv[temp_start:i])
            start_deriv = .01 * max_deriv[i]

            found = False
            find_start = i
            # has to peak within start_WT seconds
            while found == False and find_start > (i - start_WT * sampleRate):
                if EDA_deriv[find_start] < start_deriv:
                    found = True
                    peak_start[find_start] = 1
                    peak_start_times[i] = data.index[find_start]
                    rise_time[i] = get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))

                find_start = find_start - 1

        # If we didn't find a start
            if found == False:
                peak_start[i - start_WT * sampleRate] = 1
                peak_start_times[i] = data.index[i - start_WT * sampleRate]
                rise_time[i] = start_WT

            # Check if amplitude is too small
            if thres > 0 and (data['EDA_Phasic'].iloc[i] - data['EDA_Phasic'][peak_start_times[i]]) < thres:
                peaks[i] = 0
                peak_start[i] = 0
                peak_start_times[i] = ''
                max_deriv[i] = 0
                rise_time[i] = 0

    # Finding the end of the peak, amplitude of peak
    peak_end = np.zeros(len(data))
    peak_end_times = [''] * len(data)
    amplitude = np.zeros(len(data))
    decay_time = np.zeros(len(data))
    half_rise = [''] * len(data)
    SCR_width = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            peak_amp = data['EDA_Phasic'].iloc[i]
            start_amp = data['EDA_Phasic'][peak_start_times[i]]
            amplitude[i] = peak_amp - start_amp

            half_amp = amplitude[i] * .5 + start_amp

            found = False
            find_end = i
            # has to decay within end_WT seconds
            while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                if data['EDA_Phasic'].iloc[find_end] < half_amp:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                    decay_time[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])

                    # Find width
                    find_rise = i
                    found_rise = False
                    while found_rise == False:
                        if data['EDA_Phasic'].iloc[find_rise] < half_amp:
                            found_rise = True
                            half_rise[i] = data.index[find_rise]
                            SCR_width[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                        find_rise = find_rise - 1

                elif peak_start[find_end] == 1:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                find_end = find_end + 1

            # If we didn't find an end
            if found == False:
                min_index = np.argmin(data['EDA_Phasic'].iloc[i:(i + end_WT * sampleRate)].tolist())
                peak_end[i + min_index] = 1
                peak_end_times[i] = data.index[i + min_index]

    peaks = np.concatenate((peaks, np.array([0])))
    peak_start = np.concatenate((peak_start, np.array([0])))

    max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)

    return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise

def detect_raw_peaks(df, time_col="Time", signal_col="filtered_BVP", fs=64, min_distance_sec=0.3):
    signal_data = df[signal_col].values
    peaks_idx, _ = find_peaks(signal_data, distance=int(min_distance_sec * fs))

    peaks_times = pd.Series(df[time_col].values[peaks_idx])
    peaks_values = pd.Series(signal_data[peaks_idx])

    return peaks_times, peaks_values

def filter_peaks(peaks_times, peaks_values, min_rr=0.3, max_rr=1.3):
    intervals = peaks_times.diff().dt.total_seconds()
    valid_mask = (intervals >= min_rr) & (intervals <= max_rr)

    valid_mask.iloc[0] = True  # garder le premier pic

    peaks_times = peaks_times[valid_mask].reset_index(drop=True)
    peaks_values = peaks_values[valid_mask].reset_index(drop=True)
    return peaks_times, peaks_values



# ----------------------------
# Fonctions principales
# ----------------------------
def load_eda_data(data, tz="Europe/Paris"):
    """
    Charge des données EDA depuis :
    - un fichier CSV (str ou Path)
    - un fichier Parquet (str ou Path)
    - un DataFrame déjà en mémoire

    Transformations :
    - renomme "value" -> "EDA" si nécessaire
    - ajoute "Time_sec" (secondes relatives depuis le début)
    - ajoute "Time" (datetime absolu au format YYYY-mm-dd HH:MM:SS.milliseconds, sans timezone)

    Retour :
    - DataFrame EDA prêt à l’emploi
    """

    # --- Chargement selon le type ---
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError("⚠️ Format non supporté (uniquement .csv ou .parquet)")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("⚠️ data doit être un chemin vers un fichier CSV/Parquet ou un DataFrame pandas")

    # --- Filtrer uniquement EDA si plusieurs signaux sont présents ---
    if "signal_type" in df.columns:
        df = df[df["signal_type"] == "EDA"].copy()

    # --- Renommer la colonne "value" en "EDA" si elle existe ---
    if "value" in df.columns:
        df = df.rename(columns={"value": "EDA"})

    # --- Temps relatif en secondes (si Time_us présent) ---
    if "Time_us" in df.columns:
        df["Time_sec"] = (df["Time_us"] - df["Time_us"].iloc[0]) / 1e6

        # Temps absolu en datetime, sans timezone
        df["Time"] = pd.to_datetime(df["Time_us"], unit="us", utc=True)
        df["Time"] = df["Time"].dt.tz_localize(None)  # retire le fuseau

        # Reformater proprement en string (millisecondes)
        df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

    return df

def preprocess_eda_signals(df):
    '''
    This function preprocesses the EDA data. 
    
    INPUT:
        df: requires a dataframe with a column named 'EDA' 
        
    OUTPUT:
        DF: returns the same dataframe with 6 new columnns 
        'EDA_Filtered', 'EDA_Phasic', 'EDA_Tonic', 'Wavelet1', 'Wavelet2', 'Wavelet3'
    '''
    
    
    df['EDA_Filtered'] = butter_lowpass_filter_filtfilt(data=df['EDA'], cutoff=0.6, fs=4, order=3)
    decomposition_results = decomposition(df['EDA_Filtered'], 4) 
    df['EDA_Phasic'] = decomposition_results['phasic components']
    df['EDA_Tonic'] = decomposition_results['tonic component']

    _, dtw3, dtw2, dtw1 = pywt.wavedec(np.asarray(df['EDA_Filtered'], dtype=float).copy(), 'Haar', level=3)
    dtw3_duplicates = list(np.repeat(dtw3, 8))
    dtw2_duplicates = list(np.repeat(dtw2, 4))
    dtw1_duplicates = list(np.repeat(dtw1, 2))

    df['Wavelet3'] = dtw3_duplicates[:len(df)]
    df['Wavelet2'] = dtw2_duplicates[:len(df)]
    df['Wavelet1'] = dtw1_duplicates[:len(df)]
    
    print("1. EDA data preprocessing completed")
    
    return df

def segment_eda_data(df):
    '''
    This function segments the EDA signal into 5 second non-overlapping segments. 
    The sampling frequency of our sensor was 4HZ, thereby, the window size for segmenting the data is 20 samples 
    5 seconds*4Hz = 20 data samples.  
    
    INPUT:
        df: requires a dataframe with columns named: 
        'EDA_Filtered', 'EDA_Phasic', 'Wavelet1', 'Wavelet2', 'Wavelet3', 'Time'
        
    OUTPUT:
        df: returns a new dataframe with 6 new columnns 
        'EDA_Filtered', 'EDA_Phasic', 'Time', 'Wavelet1', 'Wavelet2', 'Wavelet3'
        Each cell in the dataframe contains an array of 20 values. 
    '''
    
    window_size = 20
    eda = df.EDA_Filtered.values.reshape(-1, window_size)
    eda_phasic = df.EDA_Phasic.values.reshape(-1, window_size)
    time = df.Time.values.reshape(-1, window_size)
    wavelet3 = df.Wavelet3.values.reshape(-1, window_size)
    wavelet2 = df.Wavelet2.values.reshape(-1, window_size)
    wavelet1 = df.Wavelet1.values.reshape(-1, window_size)

    df = pd.DataFrame(columns=['EDA', 'EDA_Phasic', 'Time', 'Wavelet3','Wavelet2','Wavelet1'])
    df.EDA = pd.Series(list(eda))
    df.EDA_Phasic = pd.Series(list(eda_phasic))
    df.Time = pd.Series(list(time))
    df.Wavelet3 = pd.Series(list(wavelet3))   
    df.Wavelet2 = pd.Series(list(wavelet2))   
    df.Wavelet1 = pd.Series(list(wavelet1))  

    eda_times = chain.from_iterable(zip(*df.Time.values))
    eda_times = list(eda_times)
    eda_times = eda_times[:len(df)]

    df['Time_array'] = pd.Series(list(time))   
    df['Time'] = eda_times
    
    print("2. EDA data segmentation completed")
        
    return df

def compute_statistical_wavelet(df):
    '''
    This function computes the statistical and wavelets features for each preprocessed component of the EDA signal.
    The features are calculated over the 5 second non-overlapping segments. 
    
    INPUT:
        df: requires a dataframe with columns named: 
        'EDA', 'EDA_Phasic', 'Time', 'Wavelet1', 'Wavelet2', 'Wavelet3'
        
    OUTPUT:
        df: returns the same dataframe with the following features calculated for each column defined in columns variable:
            median:        median value of the 5 second windo
            mean:          average value
            std:           standard deviation
            var:           variance of the signal
            slope:         slope of the signal
            min:           minimum value
            max:           maximum value 
            fdmean:        mean of the first derivative
            fdstd:         standard deviation of the first derivative
            sdmean:        mean of the second derivative
            sdstd:         standard deviation of the second derivative
            drange:        dynamic range
    '''
    
    columns = ['EDA', 'EDA_Phasic', 'Wavelet1', 'Wavelet2', 'Wavelet3']

    for col in columns:
        data = df[col]
        name = col
        time = range(1, 21)

        medians, means, stds, variances, mins, maxs, fdmeans, sdmeans, fdstds, sdstds, dranges, slopes,  = [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(0, len(data)):
            eda = data[i]
            fd = np.gradient(eda)
            fdmeans.append(np.mean(fd))
            sd = np.gradient(fd)
            sdmeans.append(np.mean(sd))
            fdstds.append(np.std(fd))
            sdstds.append(np.std(sd))
            dranges.append(np.max(eda) - np.min(eda))
            medians.append(np.median(eda))
            means.append(np.mean(eda))
            stds.append(np.std(eda))
            variances.append(np.var(eda))
            mins.append(np.min(eda))
            maxs.append(np.max(eda)) 
            slope, intercept, r_value, p_value, std_err = linregress(time,data[i])  
            slopes.append(slope)


        df[name+'_median'] = medians
        df[name+'_mean'] = means
        df[name+'_std'] = stds
        df[name+'_var'] = variances
        df[name+'_slope'] = slopes
        df[name+'_min'] = mins
        df[name+'_max'] = maxs
        df[name+'_fdmean'] = fdmeans
        df[name+'_fdstd'] = fdstds
        df[name+'_sdmean'] = sdmeans
        df[name+'_sdstd'] = sdstds
        df[name+'_drange'] = dranges
        
    print("3. Statistical and wavelets feature extraction completed")
    
    return df

def compute_peaks_features(df):
    '''
    This function computes the peaks features for each 5 second window in the EDA signal.

    INPUT:
        df: DataFrame with columns 'EDA' and 'Time' (tz-aware)

    OUTPUT:
        df: same DataFrame with new columns:
            peaks_p, rise_time_p, max_deriv_p, amp_p,
            decay_time_p, SCR_width_p, auc_p
    '''

    thresh = 0.01
    offset = 1
    start_WT = 3
    end_WT = 10

    peaks, rise_times, max_derivs, amps, decay_times, SCR_widths, aucs = [], [], [], [], [], [], []

    for i in range(len(df)):
        # Préparer le DataFrame pour la fenêtre actuelle
        data_df = pd.DataFrame({
            "EDA_Phasic": [df.EDA.iloc[i]],
            "Time": [df.Time.iloc[i]]
        })
        data_df.set_index('Time', inplace=True)

        # Appel à la fonction findPeaks
        returnedPeakData = findPeaks(
            data_df, offset*SAMPLE_RATE, start_WT, end_WT, thresh, SAMPLE_RATE
        )

        # Construire un DataFrame des résultats
        result_df = pd.DataFrame({
            "peaks": returnedPeakData[0],
            "amp": returnedPeakData[5],
            "max_deriv": returnedPeakData[6],
            "rise_time": returnedPeakData[7],
            "decay_time": returnedPeakData[8],
            "SCR_width": returnedPeakData[9]
        })

        featureData = result_df[result_df.peaks == 1].copy()
        featureData[['SCR_width','decay_time']] = featureData[['SCR_width','decay_time']].replace(0, np.nan)
        featureData['AUC'] = featureData['amp'] * featureData['SCR_width']

        # Ajouter les valeurs pour cette fenêtre
        peaks.append(len(featureData))
        amps.append(result_df[result_df.peaks != 0].amp.mean())
        max_derivs.append(result_df[result_df.peaks != 0].max_deriv.mean())
        rise_times.append(result_df[result_df.peaks != 0].rise_time.mean())
        decay_times.append(featureData.decay_time.mean())
        SCR_widths.append(featureData.SCR_width.mean())
        aucs.append(featureData.AUC.mean())

    # Ajouter les nouvelles colonnes au DataFrame
    df['peaks_p'] = peaks

    df['rise_time_p'] = rise_times

    df['max_deriv_p'] = max_derivs
    df['amp_p'] = amps
    df['decay_time_p'] = decay_times
    df['SCR_width_p'] = SCR_widths
    df['auc_p'] = aucs

    print("4. Peaks feature extraction completed")
    return df

def remove_flat_responses(df):
    '''
    This function computes the peaks features for each 5 second window in the EDA signal.
    
    INPUT:
        df:        requires a dataframe with the calculated EDA slope feature
        
    OUTPUT:
        df:        returns a dataframe with only the 5-second windows that are not flat responses
    '''
    
    eda_flats = df.EDA_slope.between(-0.002, 0.002)
    df['Flat'] = eda_flats.values
    df['Flat'] = df.Flat.astype(int).values
    df_wo_flat = df[df.Flat == 0]
    
    print("5. Flat responses removed")
        
    return df_wo_flat


@dataclass
class ArtifactModelConfig:
    model_path: str = "SA_Detection.json"


def predict_shape_artifacts(features, df, config: ArtifactModelConfig | None = None):
    """Version notebook : XGBoost .json + predict() -> Artifact (0/1)

    - Ne change pas la méthode : même modèle, mêmes features, même prédiction.
    - Seule amélioration : le chemin du modèle est paramétrable (par défaut SA_Detection.json).
    """
    import xgboost

    if config is None:
        config = ArtifactModelConfig()

    df = df.fillna(-1)

    df_subselect = df[features]
    test_data = df_subselect.values

    model = xgboost.XGBClassifier()
    model.load_model(str(config.model_path))

    results = model.predict(test_data)

    df['Artifact'] = list(results)

    print("6. EDA artifacts identification completed")
    return df


def label_artifacts(database_wo_flats_artifacts, database):
    '''
    This function adds the "Artifacts" column to the initial dataframe provided. 
    
    INPUT:
        database_wo_flats_artifacts:        requires as input the dataframe without flat responses 
        database:                           requires as input the initial dataframe with EDA and Time colmns
        
    OUTPUT:
        database:                           returns the database dataframe with a new column "Artifact"s
    '''

    database_wo_flats_artifacts = database_wo_flats_artifacts.explode('Time_array')

    # Add a new column that will contain the labeled artifacts (0 refers to clean, 1 refers to artifact)
    database['Artifact'] = 0

    # Add the identified artifacts from dataframe without flat responses to the initial dataframe with EDA and Time columns
    database.loc[database.Time.isin(database_wo_flats_artifacts.Time_array), 'Artifact'] = database_wo_flats_artifacts.loc[database_wo_flats_artifacts.Time_array.isin(database.Time), 'Artifact'].values

    # Label EDA < 0.05 as artifact
    database.loc[database.EDA < 0.05, 'Artifact'] = 1
        
    print("7. Preparing final database with labeled artifacts completed")
    
    return database 

