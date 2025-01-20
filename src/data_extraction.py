import wfdb
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import preprocessing
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
import warnings


warnings.filterwarnings("ignore", message="nperseg = .* is greater than input length  = .*")

class DataCleaner:
    def __init__(self, full_path):
        self.full_path = full_path
        self.patient_record = wfdb.rdsamp(full_path)
        self.record = wfdb.rdrecord(full_path)
        self.anno = wfdb.rdann(full_path, 'atr')
        self.segmented_r_peaks = []
        self.hrv_features_list = []
        self.labels = []
        
    def visualise(self, samples):
        plot_record = wfdb.rdrecord(self.full_path, sampto=samples)
        wfdb.plot_wfdb(plot_record)
        
    def show_annotations(self):
        for i in range(len(self.anno.symbol)):
            sample = self.anno.sample[i]
            symbol = self.anno.symbol[i]
            if symbol != 'N':
                print(f"Sample: {sample}, Symbol: {symbol}")
                
    def calculate_hrv_features(self, rr_intervals):
        time_domain_features = get_time_domain_features(rr_intervals)
        frequency_domain_features = get_frequency_domain_features(rr_intervals)
        
        return {
            "time_domain_features": time_domain_features,
            "frequency_domain_features": frequency_domain_features
        }
    
    def segment_peaks(self, window_size, overlap):
        signals = preprocessing.scale(np.nan_to_num(self.record.p_signal[:,0])).tolist()
        segmented_r_peaks = []
    
        peaks, _ = find_peaks(signals, distance=150)
        
        print('Peaks: ', peaks)

        for i in range(0, len(peaks) - window_size, overlap):
            segment = peaks[i:i + window_size]
            segmented_r_peaks.append(segment)
            
        return segmented_r_peaks     

    def get_hrv_features(self):
        hrv_features_list = []
    
        for segment in self.segmented_r_peaks:
            rr_intervals = np.diff(segment) 
            hrv_features = self.calculate_hrv_features(rr_intervals)            
            hrv_features_list.append(hrv_features)

        return hrv_features_list
            
    def get_labels(self, input_size):
        segment_labels = []
        
        for segment in self.segmented_r_peaks:
            for peak in segment:
                start, end = peak - input_size // 2, peak + input_size // 2
                if start < 0:
                    start = 0
                ann = wfdb.rdann(self.full_path, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
                window_label = 'N' if all(label == 'N' or not label for label in ann.symbol) else 'A'
                if window_label != 'N':
                    break
            segment_labels.append(window_label)
        
        return segment_labels
    
    def get_data(self, window_size, overlap, input_size):
        self.segmented_r_peaks = self.segment_peaks(window_size=window_size, overlap=overlap)
        self.hrv_features_list = self.get_hrv_features()
        self.labels = self.get_labels(input_size=input_size)
        patient_number = self.full_path[-3:]
        
        df = pd.DataFrame([
            {
                'patient': patient_number,
                **feature['time_domain_features'],
                **feature['frequency_domain_features'],
                'label': label,
            }
            for feature, label in zip(self.hrv_features_list, self.labels)
        ])
        
        print(f'Data created for patient number: {patient_number}')
        return df
        # csv_path = f'C:/Users/91916/OneDrive/Desktop/College work/LY/LY Project/csv_data/patient_{patient_number}.csv'
        # df.to_csv(csv_path, index=False)
        
        
if __name__ == '__main__':
    print('Testing')
    dataset_path = "dataset/mit-bih/"
    patient_number = "100"
    full_path = dataset_path+patient_number
    print(f'Data and patient number: {full_path}')
    
    obj = DataCleaner(full_path=full_path)
    obj.get_data(window_size=100, overlap=50, input_size=256)