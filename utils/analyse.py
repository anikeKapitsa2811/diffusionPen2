
import pandas as pd
import os

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'acc:' in line:
                parts = line.split()
                acc_idx = parts.index('acc:') + 1
                acc_str = parts[acc_idx]
                acc = float(acc_str[1:].split(',')[0]) if acc_str.startswith('(') else float(acc_str)
                data.append(acc)
    print(f"Extracted {len(data)} lines from {os.path.basename(file_path)}")
    return data

def create_csv_from_logs(folder_path, output_csv):
    all_data = {}
    max_rows = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            file_data = parse_log_file(file_path)
            all_data[file_name] = file_data
            max_rows = max(max_rows, len(file_data))
    
    data_dict = {file_name: data + [""] * (max_rows - len(data)) for file_name, data in all_data.items()}
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False)
    print(f"Saved DataFrame to {output_csv}")

if __name__ == "__main__":
    folder_path = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/analyse/"
    output_csv = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/log_summary1.csv"
    create_csv_from_logs(folder_path, output_csv)
