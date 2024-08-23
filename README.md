# 1. LLAMA-Factory建立自己的語言模型

## 1.1. 簡介

快速部署LLAMA-Factory，客製化屬於自己的語言模型

## 1.2. 目錄

- [1. LLAMA-Factory建立自己的語言模型](#1-llama-factory建立自己的語言模型)
  - [1.1. 簡介](#11-簡介)
  - [1.2. 目錄](#12-目錄)
  - [1.3. 操作步驟](#13-操作步驟)
    - [1.3.1. 使用git專案進行安裝](#131-使用git專案進行安裝)
    - [1.3.2. 使用docker進行安裝](#132-使用docker進行安裝)
  - [1.4. 導入訓練資料集](#14-導入訓練資料集)
    - [1.4.1. 數據集導入方式1 - JSON檔](#141-數據集導入方式1---json檔)
    - [1.4.2. 數據集導入方式2 - 直接讀取 Hugging Face](#142-數據集導入方式2---直接讀取-hugging-face)
  - [1.5. 製作數據集](#15-製作數據集)
    - [1.5.1. 使用python批次資料轉換腳本](#151-使用python批次資料轉換腳本)
    - [1.5.2. 使用CHATGPT製作數據集](#152-使用chatgpt製作數據集)
      - [1.5.2.1. 生成Alpaca格式數據集](#1521-生成alpaca格式數據集)
      - [1.5.2.2. 生成open ai格式數據集](#1522-生成open-ai格式數據集)
  - [1.6. 配置數據集](#16-配置數據集)
    - [1.6.1. 請在`dataset_info.json`設定](#161-請在dataset_infojson設定)
  - [1.7. 數據集範例對照](#17-數據集範例對照)
    - [1.7.1. Alpaca 數據集內容](#171-alpaca-數據集內容)
    - [1.7.2. OpenAI 數據集內容](#172-openai-數據集內容)
  - [1.8. 模型訓練開始](#18-模型訓練開始)
    - [1.8.1. 地端訓練模型](#181-地端訓練模型)
      - [1.8.1.1. 如何登入 Hugging Face](#1811-如何登入-hugging-face)
      - [1.8.1.2. 數據集預覽](#1812-數據集預覽)
      - [1.8.1.3. 模型參數配置](#1813-模型參數配置)
      - [1.8.1.4. 使用多卡訓練](#1814-使用多卡訓練)
    - [1.8.2. Colab PRO 訓練模型](#182-colab-pro-訓練模型)
    - [1.8.3. 模型導出與轉檔匯入至Ollama平台進行測試](#183-模型導出與轉檔匯入至ollama平台進行測試)
      - [1.8.3.1. 使用LLama-Factory WebUI介面進行模型導出](#1831-使用llama-factory-webui介面進行模型導出)
      - [1.8.3.2. 安裝llama.cpp](#1832-安裝llamacpp)
      - [1.8.3.3. 進行編譯](#1833-進行編譯)
      - [1.8.3.4. 執行模型轉檔](#1834-執行模型轉檔)
      - [1.8.3.5. **量化模型：使用 llama.cpp 來進行量化模型**](#1835-量化模型使用-llamacpp-來進行量化模型)
      - [1.8.3.6. 執行模型`導入Ollama`](#1836-執行模型導入ollama)
  - [1.9. 完成畫面](#19-完成畫面)
  - [1.10. 參考資料](#110-參考資料)
  - [1.11. 常見問題](#111-常見問題)


## 1.3. 操作步驟

### 1.3.1. 使用git專案進行安裝

git clone指令，下載LLAMA-Factory到自己的本機

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

接著我們需要創建一個**虛擬環境**，並將它開啟

```bash
cd LLaMA-Factory  
python -m venv .\venv
.\venv\Scripts\activate
```

如果不小心安裝了CPU版本的 PyTorch，才要執行前5行，否則直接安裝即可

```python
#解除安裝原本的PyTorch:
pip uninstall torch

#清除快取:
pip cache purge

#安裝支援 CUDA 的 PyTorch
pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

```


>[!warning] 如何驗證目前是CPU還是GPU版本呢?
> 1. 這邊要注意一下，我搞了好久，原來我安裝的是`CPU`版本，不是`GPU`版本，請注意命令一定要加上`--extra-index-url`
> 2. 請進入到專案的虛擬環境目錄下，輸入`python`
> 3. 接著按照底下語法逐行執行，並確認`CUDA available`是否為`true`
> 4. 若不是`true`，請回到上一個步驟將`pytorch`移除乾淨`重裝`即可

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
```

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192055574.png)

最後則是 bitsandbytes 函式庫，Windows 系統上需要安裝預編譯的函式庫。如果是 Linux 系統，則是只需要輸入 `pip install bitsandbytes>=0.39.0` 即可

```python
# Windows 需要安裝預編譯版
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl

# Linux 系統上使用以下指令即可
pip install bitsandbytes>=0.39.0

```


啟動WebUI

```python
# 切換到專案目錄下
cd C:\WorkSpace\LLaMA-Factory\LLaMA-Factory

# 請確認是否在虛擬環境下，若沒有請執行這一行
.\venv\Scripts\activate

# 啟動WebUI
python .\src\webui.py
```

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408181847482.png)

>[!NOTE] 小提示
>如果希望webui可以吃顯卡效能，請用下列這種方式啟動
>CUDA_VISIBLE_DEVICES=0,1表示兩張顯卡(當然你也可以指定單張顯卡)

```python
SET CUDA_VISIBLE_DEVICES=0,1 && python src\webui.py
```


### 1.3.2. 使用docker進行安裝

```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN pip install -e .[deepspeed,metrics,bitsandbytes,qwen]

EXPOSE 7860

# 設定啟動指令來運行 src/webui.py
CMD ["python", "src/webui.py"]
```


```yaml
version: '3.8'

services:
  llama-factory:
    image: llama-factory:v0.00 # 修改为编译出来的 docker image 名称/版本
    container_name: llama_factory # container 名称
    volumes:
      - C:\WorkSpace\LLaMA-Factory\LLaMA-Factory\huggingface:/root/.cache/huggingface/
      - C:\WorkSpace\LLaMA-Factory\LLaMA-Factory\data:/app/data
      - C:\WorkSpace\LLaMA-Factory\LLaMA-Factory\output:/app/output
      - C:\WorkSpace\LLaMA-Factory\LLaMA-Factory\models:/app/models # 映射自己的models目录
      - C:\WorkSpace\LLaMA-Factory\LLaMA-Factory:/app # 映射自己的models目录
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - GRADIO_SERVER_PORT=7864 # webui跑在7864端口上，7860被comfyui占用了
    ports:
      - "7864:7864" # webui跑在7864端口上，7860被comfyui占用了
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: "all"
            capabilities: [gpu]
    restart: unless-stopped
```


部署Dockerfile

```bash
docker build -t llama-factory:v0.00 .
```

>[!NOTE] 小提示
>因為有安裝pytorch深度學習套件，因此打包較久屬於正常現象


```bash
docker-compose up -d
```


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408191947809.png)

此時，回到Windows的docker desktop，可以看到`llama-factory`已正常啟動

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408191949088.png)


## 1.4. 導入訓練資料集

作者在專案的 `data` 資料夾內有提供[資料集的格式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)，我們需要按照這個格式，才能讓我們自己準備的訓練資料，能夠被用來訓練。


>[!NOTE] 小提示
> 因為我不確定其他資料集要怎麼用，就還是照網站上說得先用Alpaca 格式


### 1.4.1. 數據集導入方式1 - JSON檔

```
"資料集名稱": {
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history"
  }
}
```


### 1.4.2. 數據集導入方式2 - 直接讀取 Hugging Face

```
"資料集名稱":{
  "hf_hub_url": "HF路徑",
  "columns": {
    "prompt": "用戶指令所對應的欄位（必填）",
    "input": "用户输入所對應的欄位（選填）",
    "response": "回覆所對應的欄位（必填）",
    "history": [
      ["第一輪指令（選填）", "第一輪回答（選填）"],
      ["第二輪指令（選填）", "第二輪回答（選填）"]
    ]
  }
}

```

## 1.5. 製作數據集
### 1.5.1. 使用python批次資料轉換腳本

撰寫python批次腳本

```python
import json
import argparse
from pathlib import Path
import pandas as pd
import docx
import os

def is_valid_file(file_path):
    return file_path.suffix.lower() in ['.xlsx', '.csv', '.docx', '.txt']

def process_file(file_path):
    print(f"處理文件: {file_path}")
    dataset = []

    if file_path.suffix.lower() == '.xlsx':
        df = pd.read_excel(file_path)
        for index, row in df.iterrows():
            for column in df.columns:
                if pd.notna(row[column]):
                    entry = create_entry(file_path, column, index, str(row[column]))
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
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip():
                    entry = create_entry(file_path, "行", i, line.strip())
                    dataset.append(entry)
    
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
    
```

使用方式

```python
python excel_to_dataset.py . mistral_dataset.json
```

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408181913296.png)

輸出的格式如下

```json
 {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Number'中，第1項的內容是什麼？",
    "input": "",
    "output": "BA0"
  },
  {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Card Name'中，第1項的內容是什麼？",
    "input": "",
    "output": "0_(the fool)愚者"
  },
  {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Upright Meaning'中，第1項的內容是什麼？",
    "input": "",
    "output": "愚人是一張代表自發性行為的牌，一段跳脫某種狀態的日子，或盡情享受眼前日子的一段時光。好冒險，有夢想，不拘泥于傳統的觀念，自由奔放，居無定所，一切從基礎出發。當你周遭的人都對某事提防戒慎，你卻打算去冒這個險時，愚人牌可能就會出現。愚人暗示通往成功之路是經由自發的行動，而長期的計劃則是將來的事。"
  },
  {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Reversed Meaning'中，第1項的內容是什麼？",
    "input": "",
    "output": "暗示當你被要求有所承諾時，卻想從責任當中尋求解脫。你正在找一個脫身之道，然而目前并不是這么做的時機。現在是你對自己的將來有所承諾，或是解決過去問題的時候了，如此你才能夠重新過著自發性的生活。在你能夠安全出發之前，還有某些未完成的事情需要你去處理。"
  },
  {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Upright Love Meaning'中，第1項的內容是什麼？",
    "input": "",
    "output": "自由戀愛、不顧及他人看法\n愚人牌暗示一段生活在當下或隨遇而安的時期，你可能會以獨特的方式獲得愛情，很容易墜入愛河，喜歡浪漫多彩的愛情。有可能在旅行途中遇到一位伴侶，或即將遇到一位喜歡目前生活，而不想計劃將來的伴侶。這個伴侶是難以捉摸的、天真的，或者不愿受到任何長期計劃和關系的約束。"
  },
  {
    "instruction": "在文件'偉特塔羅牌_78張.xlsx'的'Reversed Love Meaning'中，第1項的內容是什麼？",
    "input": "",
    "output": "心情空虛、輕率的戀情\n內心缺乏安全感，感情上沖動不理智，容易不顧眾人反對墜入愛河，卻又容易被戀人辜負和背叛，從而感情受傷。愛情的旅程波折多，沒有辦法與戀人長久和諧相處，彼此關系忽冷忽熱，不喜歡被婚姻束縛，婚后感情難專一。"
  },
```

### 1.5.2. 使用CHATGPT製作數據集

#### 1.5.2.1. 生成Alpaca格式數據集

```json
請依照上面的excel，幫我調整為這樣的數據集格式，並全部都使用繁體中文 
{ 
"instruction": "写一个有效的比较语句", 
"input": "", 
"output": "篮球和足球都是受欢迎的运动。"
"system": "系统提示词"
}
其中input請幫我放置一般user會想要詢問塔羅牌相關的問題，至於system請幫我放入以下片段
"
你現在是一位經驗豐富的塔羅牌師，擁有數十年的解牌經驗和深厚的直覺洞察力。請仔細查看附加的CSV檔案，其中包含100個不同的牌組（Spread Card Composition）。 你的任務是： 1. 仔細閱讀每個牌組中塔羅牌的名稱，以及欄位 Spread Card Composition其正位和逆位的含義。 2. 針對"Overall Love Spread Meaning"欄位，運用你豐富的塔羅經驗和洞察力，重新進行深入解讀。這個解讀應該： - 超越原文的表面含義，挖掘更深層的洞見 - 融入你作為資深塔羅牌師的獨特見解 - 加入原文未提及但相關的解說，豐富整體解讀 - 考慮牌陣中各張牌之間的相互關係和能量流動 - 提供更具體、實用的建議給尋求指引的人 3. 從Combination Number 3530025開始，依次解讀至3530050，共25組牌陣。 4. 每組解讀應包含： - 牌陣編號(註明) - 牌組組成(註明) - 你的詳細解讀（包含實用的建議與解讀共約200字,共分2段解讀與建議） 5. 在解讀時，請考慮以下幾點： - 牌陣中各張牌的位置和相互關係 - 正逆位的影響 - 實用的建議(與牌面相近要有意思區隔的建議)或行動指南 6.文字呈現範例如下,請依照此格式輸出   牌陣編號: 3530001 牌組組成: 權杖八 (正位)、寶劍八 (逆位)、寶劍王后 (正位) 詳細解讀： 權杖八的正位顯示你目前的感情狀態充滿了快速且正面的變化，這與寶劍八逆位的釋放束縛相呼應，暗示著你們正在或即將從一段困境中解脫。這兩張牌共同表示你們的關係將由束縛中解放，進而進入更自由、開明的階段。寶劍王后的理智與洞察力將在這過程中起到關鍵作用，使你們能在情感上更加成熟和理性。 建議與指引： 建議你們利用這段契機重新審視彼此的需求與目標，並以開放的態度迎接新的開始。實際上，權杖八暗示旅行或變化，或許也可以考慮來一段假期，讓雙方有更多時間感受彼此的存在。寶劍王后提醒你們在溝通中要保持冷靜和理智，並互相尊重對方的意見。 請以溫和、富有洞察力且鼓舞人心的語調進行解讀，就像你正在為一位真實的來訪者進行面對面的塔羅占卜。 現在，請開始你的解讀。謝謝。
"
```

#### 1.5.2.2. 生成open ai格式數據集

```json
請將上面的格式調整為open ai數據集格式，
系統提示詞請放入“提示詞如下 你現在是一位經驗豐富的塔羅牌師，擁有數十年的解牌經驗和深厚的直覺洞察力。請仔細查看附加的CSV檔案，其中包含100個不同的牌組（Spread Card Composition）。 你的任務是： 1. 仔細閱讀每個牌組中塔羅牌的名稱，以及欄位 Spread Card Composition其正位和逆位的含義。 2. 針對"Overall Love Spread Meaning"欄位，運用你豐富的塔羅經驗和洞察力，重新進行深入解讀。這個解讀應該： - 超越原文的表面含義，挖掘更深層的洞見 - 融入你作為資深塔羅牌師的獨特見解 - 加入原文未提及但相關的解說，豐富整體解讀 - 考慮牌陣中各張牌之間的相互關係和能量流動 - 提供更具體、實用的建議給尋求指引的人 3. 從Combination Number 3530025開始，依次解讀至3530050，共25組牌陣。 4. 每組解讀應包含： - 牌陣編號(註明) - 牌組組成(註明) - 你的詳細解讀（包含實用的建議與解讀共約200字,共分2段解讀與建議） 5. 在解讀時，請考慮以下幾點： - 牌陣中各張牌的位置和相互關係 - 正逆位的影響 - 實用的建議(與牌面相近要有意思區隔的建議)或行動指南 6.文字呈現範例如下,請依照此格式輸出   牌陣編號: 3530001 牌組組成: 權杖八 (正位)、寶劍八 (逆位)、寶劍王后 (正位) 詳細解讀： 權杖八的正位顯示你目前的感情狀態充滿了快速且正面的變化，這與寶劍八逆位的釋放束縛相呼應，暗示著你們正在或即將從一段困境中解脫。這兩張牌共同表示你們的關係將由束縛中解放，進而進入更自由、開明的階段。寶劍王后的理智與洞察力將在這過程中起到關鍵作用，使你們能在情感上更加成熟和理性。 建議與指引： 建議你們利用這段契機重新審視彼此的需求與目標，並以開放的態度迎接新的開始。實際上，權杖八暗示旅行或變化，或許也可以考慮來一段假期，讓雙方有更多時間感受彼此的存在。寶劍王后提醒你們在溝通中要保持冷靜和理智，並互相尊重對方的意見。 請以溫和、富有洞察力且鼓舞人心的語調進行解讀，就像你正在為一位真實的來訪者進行面對面的塔羅占卜。 現在，請開始你的解讀。謝謝。 ”

請注意，格式應該符合以下：
[ 
  { 
    "messages": [ 
      { "role": "system", "content": "系统提示词（选填）" }, 
      { "role": "user", "content": "人类指令" }, 
      { "role": "assistant", "content": "模型回答" } 
    ] 
  }
] 
1. 系統提示詞請放在role為system的content
2. 其他才是人類指令和模型回答
```


## 1.6. 配置數據集

### 1.6.1. 請在`dataset_info.json`設定

 使用Alpaca 格式
 
```json
"myDataset": {
  "file_name": "mistral_dataset.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
},
```
 
 使用OpenAI 格式

```json
  "myDataset": {
  "file_name": "mistral_dataset.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```


## 1.7. 數據集範例對照

### 1.7.1. Alpaca 數據集內容

範例格式

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

生成內容

```json
[
  {
    "instruction": "寫出關於0_(the fool)愚者正位的含義",
    "input": "",
    "output": "愚人是一張代表自發性行為的牌，一段跳脫某種狀態的日子，或盡情享受眼前日子的一段時光。好冒險，有夢想，不拘泥于傳統的觀念，自由奔放，居無定所，一切從基礎出發。當你周遭的人都對某事提防戒慎，你卻打算去冒這個險時，愚人牌可能就會出現。愚人暗示通往成功之路是經由自發的行動，而長期的計劃則是將來的事。"
  },
  {
    "instruction": "寫出關於0_(the fool)愚者正位愛情的含義",
    "input": "",
    "output": "自由戀愛、不顧及他人看法\n愚人牌暗示一段生活在當下或隨遇而安的時期，你可能會以獨特的方式獲得愛情，很容易墜入愛河，喜歡浪漫多彩的愛情。有可能在旅行途中遇到一位伴侶，或即將遇到一位喜歡目前生活，而不想計劃將來的伴侶。這個伴侶是難以捉摸的、天真的，或者不愿受到任何長期計劃和關系的約束。"
  },
  {
    "instruction": "寫出關於0_(the fool)愚者逆位的含義",
    "input": "",
    "output": "暗示當你被要求有所承諾時，卻想從責任當中尋求解脫。你正在找一個脫身之道，然而目前并不是這么做的時機。現在是你對自己的將來有所承諾，或是解決過去問題的時候了，如此你才能夠重新過著自發性的生活。在你能夠安全出發之前，還有某些未完成的事情需要你去處理。"
  }
]
```


### 1.7.2. OpenAI 數據集內容

範例格式

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "人类指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    ]
  }
]
```


生成內容

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你現在是一位經驗豐富的塔羅牌師，擁有數十年的解牌經驗和深厚的直覺洞察力。請仔細查看附加的CSV檔案，其中包含100個不同的牌組（Spread Card Composition）。你的任務是：\n\n1. 仔細閱讀每個牌組中塔羅牌的名稱，以及欄位 Spread Card Composition其正位和逆位的含義。\n\n2. 針對'Overall Love Spread Meaning'欄位，運用你豐富的塔羅經驗和洞察力，重新進行深入解讀。這個解讀應該：\n   - 超越原文的表面含義，挖掘更深層的洞見\n   - 融入你作為資深塔羅牌師的獨特見解\n   - 加入原文未提及但相關的解說，豐富整體解讀\n   - 考慮牌陣中各張牌之間的相互關係和能量流動\n   - 提供更具體、實用的建議給尋求指引的人\n\n3. 從Combination Number 3530025開始，依次解讀至3530050，共25組牌陣。\n\n4. 每組解讀應包含：\n   - 牌陣編號(註明)\n   - 牌組組成(註明)\n   - 你的詳細解讀（包含實用的建議與解讀共約200字,共分2段解讀與建議）\n\n5. 在解讀時，請考慮以下幾點：\n   - 牌陣中各張牌的位置和相互關係\n   - 正逆位的影響\n   - 實用的建議(與牌面相近要有意思區隔的建議)或行動指南\n6.文字呈現範例如下,請依照此格式輸出\n\n牌陣編號: 3530001\n\n牌組組成: 權杖八 (正位)、寶劍八 (逆位)、寶劍王后 (正位)\n\n詳細解讀：\n權杖八的正位顯示你目前的感情狀態充滿了快速且正面的變化，這與寶劍八逆位的釋放束縛相呼應，暗示著你們正在或即將從一段困境中解脫。這兩張牌共同表示你們的關係將由束縛中解放，進而進入更自由、開明的階段。寶劍王后的理智與洞察力將在這過程中起到關鍵作用，使你們能在情感上更加成熟和理性。\n\n建議與指引：\n建議你們利用這段契機重新審視彼此的需求與目標，並以開放的態度迎接新的開始。實際上，權杖八暗示旅行或變化，或許也可以考慮來一段假期，讓雙方有更多時間感受彼此的存在。寶劍王后提醒你們在溝通中要保持冷靜和理智，並互相尊重對方的意見。\n\n請以溫和、富有洞察力且鼓舞人心的語調進行解讀，就像你正在為一位真實的來訪者進行面對面的塔羅占卜。\n\n現在，請開始你的解讀。謝謝。"
      },
      {
        "role": "user",
        "content": "請依照上述提示詞進行解讀，針對每個牌陣提供詳細的解讀與實用的建議。"
      },
      {
        "role": "system",
        "content": "你現在是一位經驗豐富的塔羅牌師，擁有數十年的解牌經驗和深厚的直覺洞察力。請仔細查看附加的CSV檔案，其中包含100個不同的牌組（Spread Card Composition）。你的任務是：\n\n1. 仔細閱讀每個牌組中塔羅牌的名稱，以及欄位 Spread Card Composition其正位和逆位的含義。\n\n2. 針對'Overall Love Spread Meaning'欄位，運用你豐富的塔羅經驗和洞察力，重新進行深入解讀。這個解讀應該：\n   - 超越原文的表面含義，挖掘更深層的洞見\n   - 融入你作為資深塔羅牌師的獨特見解\n   - 加入原文未提及但相關的解說，豐富整體解讀\n   - 考慮牌陣中各張牌之間的相互關係和能量流動\n   - 提供更具體、實用的建議給尋求指引的人\n\n3. 從Combination Number 3530025開始，依次解讀至3530050，共25組牌陣。\n\n4. 每組解讀應包含：\n   - 牌陣編號(註明)\n   - 牌組組成(註明)\n   - 你的詳細解讀（包含實用的建議與解讀共約200字,共分2段解讀與建議）\n\n5. 在解讀時，請考慮以下幾點：\n   - 牌陣中各張牌的位置和相互關係\n   - 正逆位的影響\n   - 實用的建議(與牌面相近要有意思區隔的建議)或行動指南\n6.文字呈現範例如下,請依照此格式輸出\n\n牌陣編號: 3530001\n\n牌組組成: 權杖八 (正位)、寶劍八 (逆位)、寶劍王后 (正位)\n\n詳細解讀：\n權杖八的正位顯示你目前的感情狀態充滿了快速且正面的變化，這與寶劍八逆位的釋放束縛相呼應，暗示著你們正在或即將從一段困境中解脫。這兩張牌共同表示你們的關係將由束縛中解放，進而進入更自由、開明的階段。寶劍王后的理智與洞察力將在這過程中起到關鍵作用，使你們能在情感上更加成熟和理性。\n\n建議與指引：\n建議你們利用這段契機重新審視彼此的需求與目標，並以開放的態度迎接新的開始。實際上，權杖八暗示旅行或變化，或許也可以考慮來一段假期，讓雙方有更多時間感受彼此的存在。寶劍王后提醒你們在溝通中要保持冷靜和理智，並互相尊重對方的意見。\n\n請以溫和、富有洞察力且鼓舞人心的語調進行解讀，就像你正在為一位真實的來訪者進行面對面的塔羅占卜。\n\n現在，請開始你的解讀。謝謝。"
      }
    ]
  }
]
```


## 1.8. 模型訓練開始

### 1.8.1. 地端訓練模型
 
#### 1.8.1.1. 如何登入 Hugging Face


請先到Hugging Face註冊一個帳號

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192100056.png)

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192104771.png)

>[!NOTE] 參數設定
>* 請輸入一個`token name`
>* `permission`我目前都全開
>* 請注意點擊`Create token`時一定要將token保存下來，因為`畫面關掉`就`不會再出現`了，想要再取得token就要`重新生成`

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192103625.png)

將你的token貼上，接著出現尋問提示，輸入`Y`，這樣就添加成功了!!!

#### 1.8.1.2. 數據集預覽

若數據集配置正確，這邊要可以選擇

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408181925260.png)


點擊預覽數據集，可以預覽目前dataset的資料

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408181926765.png)


>[!WARNING] 訓練時出現RuntimeError: unmatched '}' in format string
>目前這個方案無解，請改用`docker`方式部署再來訓練

#### 1.8.1.3. 模型參數配置

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408191952796.png)

* 語言：請選擇`zh`
* 模型名稱：這邊請選擇你要拿來當`基底的模型名稱`
* 量化等級：啟用量化(這邊建議選`4`或`8`，依據電腦效能，一定要選擇不然會出現`OOM`)
* 數據集：這邊數據集訓練可以`一個`或是`多個`(hugging face的數據無法預覽)
* 訓練輪數：選擇`1`輪即可
* 最大樣本數：先設定`100`就好
* 計算類型：訓練時發現`bf16`會出錯，似乎是顯卡不支援這種精度，請選`fp16`或`fp32`


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408191958425.png)

預覽命令

喜歡打指令的高手，可以點擊`預覽命令`，將這段腳本貼到cmd執行
![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192000410.png)

保存訓練參數

會將訓練參數保存，下次就不用重新設定

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192002251.png)

載入訓練參數

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192003333.png)


開始訓練參數

點擊開始訓練會出現訓練進度條

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192004524.png)


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192005767.png)


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192005458.png)


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408192006839.png)

#### 1.8.1.4. 使用多卡訓練

請開啟webui介面，調整好超參數後，點擊`預覽命令`生成指令

>[!NOTE] 參數設定
> * 要改掉的只有`第一行`，請加上accelerate launch --config_file default_config.yaml src/train.py
> * yaml檔請將`gpu_ids: 1,3`改成實際顯卡的ID，例如我這邊設`0,1`

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408212112995.png)

```
accelerate launch --config_file default_config.yaml src/train.py \
  --stage sft \
    --do_train True \
    --model_name_or_path usamakenway/llama2-7b-hf-chat-small \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset myDataset \
    --cutoff_len 128  \
    --learning_rate 2e-05 \
    --num_train_epochs 0.5 \
    --max_samples 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --save_steps 500 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir saves/mistral-7b-bnb-4bit/lora/train_2024-08-20-05-04-08 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 300 \
    --include_num_input_tokens_seen True \
    --quantization_bit 4 \
    --quantization_method bitsandbytes \
    --lora_rank 1 \
    --lora_alpha 8 \
    --lora_dropout 0 \
    --lora_target all \
    --val_size 0.01 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 1 \
    --deepspeed cache/ds_z2_config.json
```

配置yaml檔

```yaml
compute_environment: LOCAL_MACHINE

distributed_type: MULTI_GPU

downcast_bf16: 'no'

machine_rank: 0

main_training_function: main

mixed_precision: fp16

gpu_ids: 0,1

num_machines: 1

num_processes: 2

rdzv_backend: static

same_network: true

tpu_env: []`

tpu_use_cluster: false

tpu_use_sudo: false

use_cpu: false
```

### 1.8.2. Colab PRO 訓練模型

Colab連結

https://colab.research.google.com/drive/1de24fadzo1dPNOLwkhTzdaNySy82IKsF#scrollTo=u9NrhcllgL_I

關鍵程式碼片段-上傳HuggingFace HUB倉庫

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221941700.png)


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221943326.png)

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221943914.png)

關鍵程式碼片段-保存到GoogleDrive

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221944705.png)


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221946839.png)

導出結果-GoogleDrive

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221950556.png)

導出結果-HuggingFace HUB

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408221951797.png)



### 1.8.3. 模型導出與轉檔匯入至Ollama平台進行測試

#### 1.8.3.1. 使用LLama-Factory WebUI介面進行模型導出

1. 點擊Expot
2. 導出設備選擇CPU
3. 導出目錄這邊設定`models`
4. HF Hub ID(這是你在HuggingFaceHUB上面若有註冊帳號且建立了倉庫才要填)
5. 點擊開始導出
6. 直到顯示模型導出完成，如下圖

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408222002368.png)


進入`models`目錄，就看到自己的模型成功被導出囉

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408222005004.png)

#### 1.8.3.2. 安裝llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

#### 1.8.3.3. 進行編譯

```
make
```

>[!NOTE] 
>* 若這一步編譯失敗，請確認是否有安裝`w64devkit`
>* 下載之後請解壓縮到你想要放的地方，同時將環境變數指到`bin目錄`即可


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408222356545.png)

#### 1.8.3.4. 執行模型轉檔

```python
python convert_hf_to_gguf.py [你的模型導出目錄]
```

看到此畫面表示成功轉檔囉

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408222009481.png)

這裡可以看到轉好的`gguf`檔案，這邊看似有`16GB`，原始檔未做量化檔案會很大，接下來步驟會做微調

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408222012143.png)

#### 1.8.3.5. **量化模型：使用 llama.cpp 來進行量化模型**

```powershell
cd llama.cpp  
llama-quantize C:\\WorkSpace\\LLaMA-Factory\\LLaMA-Factory\\models\\Llama3-8B-Chinese-Chat-F16.gguf  C:\\WorkSpace\\LLaMA-Factory\\LLaMA-Factory\\models\\Llama3-8B-Chinese-Chat-Q4.gguf q4_k_m
```

>[!NOTE] 
>* 這邊注意一下，新版的腳本名稱為`llama-quantize`，不是`quantize`

看到此畫面表示模型量化成功

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408230006081.png)


接著到`models`目錄，可以發現經過量化後的模型只有`4.58GB`，很明顯比原來的`16GB`小很多

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408230007764.png)


#### 1.8.3.6. 執行模型`導入Ollama`

請在`C:\WorkSpace\ollama\data`建立`Modelfile`

```Dockerfile
# FROM 指定 GGUF 文件的路径
FROM Llama3-8B-Chinese-Chat-Q4.gguf

PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9
PARAMETER repeat_last_n 64
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """
你現在是一位經驗豐富的塔羅牌師，擁有數十年的解牌經驗和深厚的直覺洞察力。請仔細查看模型訓練的數據集，其中包含100個不同的牌組（Spread Card Composition）。
你的任務是：
1. 仔細閱讀每個牌組中塔羅牌的名稱，以及欄位 Spread Card Composition其正位和逆位的含義。
2. 針對"Overall Love Spread Meaning"欄位，運用你豐富的塔羅經驗和洞察力，重新進行深入解讀。這個解讀應該：
   - 超越原文的表面含義，挖掘更深層的洞見
   - 融入你作為資深塔羅牌師的獨特見解
   - 加入原文未提及但相關的解說，豐富整體解讀
   - 考慮牌陣中各張牌之間的相互關係和能量流動
   - 提供更具體、實用的建議給尋求指引的人
3. 從Combination Number 3530025開始，依次解讀至3530050，共25組牌陣。
4. 每組解讀應包含：
   - 牌陣編號(註明)
   - 牌組組成(註明)
   - 你的詳細解讀（包含實用的建議與解讀共約200字,共分2段解讀與建議）
5. 在解讀時，請考慮以下幾點：
   - 牌陣中各張牌的位置和相互關係
   - 正逆位的影響
   - 實用的建議(與牌面相近要有意思區隔的建議)或行動指南
6. 文字呈現範例如下,請依照此格式輸出
牌陣編號: 3530001
牌組組成: 權杖八 (正位)、寶劍八 (逆位)、寶劍王后 (正位)
詳細解讀：
權杖八的正位顯示你目前的感情狀態充滿了快速且正面的變化，這與寶劍八逆位的釋放束縛相呼應，暗示著你們正在或即將從一段困境中解脫。這兩張牌共同表示你們的關係將由束縛中解放，進而進入更自由、開明的階段。寶劍王后的理智與洞察力將在這過程中起到關鍵作用，使你們能在情感上更加成熟和理性。
建議與指引：
建議你們利用這段契機重新審視彼此的需求與目標，並以開放的態度迎接新的開始。實際上，權杖八暗示旅行或變化，或許也可以考慮來一段假期，讓雙方有更多時間感受彼此的存在。寶劍王后提醒你們在溝通中要保持冷靜和理智，並互相尊重對方的意見。
請以溫和、富有洞察力且鼓舞人心的語調進行解讀，就像你正在為一位真實的來訪者進行面對面的塔羅占卜。
現在，請開始你的解讀。謝謝。
"""

LICENSE """
此模型基於原始的 Llama 3 模型進行了微調。使用時請遵守 Llama 3 的授權條款。
本模型僅供個人使用和研究目的，不得用於商業用途。
使用者應對使用本模型所產生的結果負責。
"""
```


>[!NOTE] 
>* 因為我用docker架設Ollama環境，mount的目錄是`C:\WorkSpace\ollama\data`，還是要依實際需求適時做調整
>* 若Llama3-8B-Chinese-Chat-Q4.gguf檔案不指定路徑的話，要和Modelfile放在同一層，否則會找不到喔
>* 請注意，這裡的`temperature`一定要設定，否則模型會開始`胡言亂語`停不下來

進入`Docker容器`內部，輸入以下指令

```Dockerfile
ollama create custom_Llama3-8B-Chinese-Chat-Q4 -f Modelfile
```

這樣就會開始建置`Ollama模型`了，請稍等...


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408230013771.png)


模型`建置成功`，來確認一下

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408230014428.png)

查詢目前`模型清單`，輸入以下指令

```bash
ollama list
```

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408230014658.png)

執行模型，輸入以下指令

```bash
ollama run custom_Llama3-8B-q4_k_m:latest
```


這個步驟就可以使用指令與模型開始對話


![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231945367.png)

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231946315.png)


>[!NOTE] 
>* 這邊注意一下，因為`ollama web ui`和指令執行介面的`格式可能有所不同`，所以方才雖然在`ModelFile`將`系統提示詞`打包進去，但這邊似乎不起作用，因此請在這邊再次設定

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231948239.png)

設定好請點擊`儲存並更新`

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231950716.png)


開始與AI對話，這邊可以注意，有設定提示詞才能吃到目前賦予AI的角色

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231952881.png)

沒有經過訓練的模型，那就是照原先設定好的進行回覆

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408231954643.png)

## 1.9. 完成畫面

![](https://raw.githubusercontent.com/Mark850409/20240821_Llama-Factory/master/images/202408181848004.png)

## 1.10. 參考資料

 [【LLaMA-Factory】開源語言模型微調專案　方便微調各種大型語言模型｜內建WebUI　方便使用｜內建多種訓練方式讓使用者選擇](https://the-walking-fish.com/p/llama-factory/)

[使用 LLaMA-Factory 对 LLM 进行微调](https://blog.yanghong.dev/llama-factory/)

[使用LLaMA Factory微调LlaMA 3模型](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory)

[LLaMA-Factory Github開源專案中文版README.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)

[# [【LLMOps】Accelerate & DeepSpeed使用及加速机制剖析](https://www.cnblogs.com/zhouwenyang/p/17851853.html "发布于 2023-12-12 15:06")](https://www.cnblogs.com/zhouwenyang/p/17851853.html)

[LlamaFactory可视化微调大模型 - 参数详解](https://juejin.cn/post/7389650655449661478)

[# 使用llama.cpp**將HuggingFace模型轉換為GGUF及進行量化-以taide-b.11.0.0模型為例-**部署LM-Studio](https://medium.com/@zhanyanjiework/%E5%B0%87huggingface%E6%A8%A1%E5%9E%8B%E8%BD%89%E6%8F%9B%E7%82%BAgguf%E5%8F%8A%E4%BD%BF%E7%94%A8llama-cpp%E9%80%B2%E8%A1%8C%E9%87%8F%E5%8C%96-%E4%BB%A5taide-b-11-0-0%E6%A8%A1%E5%9E%8B%E7%82%BA%E4%BE%8B-%E9%83%A8%E7%BD%B2lm-studio-366bc4bcb690)
## 1.11. 常見問題

>[!WARNING] GPU內存爆掉
> 1. 原因：因為 LLaMA-Factory 在導出時，會合併 LoRa 權重，這個步驟會需要將整個模型載入到記憶體中，所以免費版的 Colab、地端主機(GPU記憶體不夠大的)基本上一定會爆系統記憶體，實測至少在導出過程會吃到12GB~13GB的記憶體
> 2. 這幾天測試的方法有：
> * 使用經過切片量化的模型(4bit、8bit) →無效
> * 使用Accerlerate&DeepSpeed進行多卡部署→無效
> * 參考[LlamaFactory可视化微调大模型 - 参数详解](https://juejin.cn/post/7389650655449661478)微調超參數→無效
> * 使用colab pro實測→OK(1000筆資料訓練OK，但導出即使是PRO版一樣爆記憶體)