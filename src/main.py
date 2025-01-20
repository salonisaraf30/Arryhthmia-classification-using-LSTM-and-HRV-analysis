from data_extraction import DataCleaner
import pandas as pd

# root_dir = "E:\Sem 7\LY Project\src\patient_records.txt"
# file1 = open('patient_records.txt', 'r')

file_path = r'E:\Sem 7\LY Project\src\patient_records.txt'
file1 = open(file_path, 'r')
Lines = file1.readlines()
root_dir = "E:\\Sem 7\\LY Project\\dataset\\mit-bih"
data = []

for line in Lines:
    full_path = root_dir+line.strip()
    data_cleaner = DataCleaner(full_path=full_path)
    df = data_cleaner.get_data(window_size=100, overlap=50, input_size=256)
    data.append(df)
    
result_df = pd.concat(data, ignore_index=True)
result_df.to_csv('final_patient_data.csv', index=False)