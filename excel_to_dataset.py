import json
import argparse
from pathlib import Path
import pandas as pd
import docx
import os
import zipfile
import csv

def is_valid_file(file_path):
    return file_path.suffix.lower() in ['.xlsx', '.csv', '.docx', '.txt']

def process_file(file_path):
    print(f"處理文件: {file_path}")
    dataset = []

    try:
        if file_path.suffix.lower() == '.xlsx':
            try:
                df = pd.read_excel(file_path)
                for index, row in df.iterrows():
                    for column in df.columns:
                        if pd.notna(row[column]):
                            entry = create_entry(file_path, column, index, str(row[column]))
                            dataset.append(entry)
            except zipfile.BadZipFile:
                print(f"警告: 無法讀取Excel文件 {file_path}。嘗試作為CSV文件讀取。")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for index, row in enumerate(reader):
                        for column_index, value in enumerate(row):
                            if value.strip():
                                entry = create_entry(file_path, f"列{column_index+1}", index, value)
                                dataset.append(entry)
        
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
            for index, row in df.iterrows():
                for column in df.columns:
                    if pd.notna(row[column]):
                        entry = create_entry(file_path, column, index, str(row[column]))
                        dataset.append(entry)
        
        elif file_path.suffix.lower() == '.docx':
            doc = docx.Document(file_path)
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    entry = create_entry(file_path, "段落", i, paragraph.text)
                    dataset.append(entry)
        
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip():
                        entry = create_entry(file_path, "行", i, line.strip())
                        dataset.append(entry)
    
    except Exception as e:
        print(f"處理文件 {file_path} 時出錯: {str(e)}")
    
    return dataset

def create_entry(file_path, section, index, content):
    return {
        "instruction": f"在文件'{file_path.name}'的'{section}'中，第{index+1}項的內容是什麼？",
        "input": "",
        "output": content
    }

def convert_files_to_dataset(input_dir, output_file):
    input_path = Path(input_dir)
    all_data = []
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and is_valid_file(file_path):
            all_data.extend(process_file(file_path))
    
    # 將數據集寫入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將目錄中的所有Excel、CSV、Word和TXT文件轉換為Alpaca格式的數據集")
    parser.add_argument("input_dir", type=str, help="包含文件的輸入目錄路徑", default=".")
    parser.add_argument("output_file", type=str, help="輸出JSON文件的路徑")
    
    args = parser.parse_args()
    
    convert_files_to_dataset(args.input_dir, args.output_file)
    print(f"轉換完成。Alpaca格式的數據集已保存到 {args.output_file}")