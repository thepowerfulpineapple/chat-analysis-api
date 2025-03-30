#1.两个的词云图，我不太会并行化操作，请学长来弄吧
#2.csv文件操作中的df['DateTime']均为详细时间时间戳
#3.625行无法确定sender是数字1还是文本1，改一下就行
#179行可能书写格式有问题
#135-148的请求头格式随api不同而变化
#img的最终返回值与txt和csv的不同


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import defaultdict
import json
import os
from tqdm import tqdm
import multiprocessing
import logging
from typing import List, Dict, Any, Literal
from pathlib import Path
from enum import Enum
import requests
from snownlp import SnowNLP
from itertools import groupby
from collections import Counter
import pytesseract
import cv2
import re
import numpy as np
from typing import List, Dict
from paddleocr import PaddleOCR
from dataclasses import dataclass

# 初始化配置处理图片,路径随电脑有变化
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

s ='''名分析师，现在需要对一段聊天记录进行总结。要求按照以下维度输出，每一个结论之后都应该至少举出有一个例子。一、 对一段关系的分析：
    1.时间、频率分析：
	    互动频率
	    活跃时段（频数分布表）
    2.词频、话题与兴趣：
	词频统计
	话题类型
	话题、兴趣重合点
	聊天关键词
    3.对话分析：
	    对话总结
	    主动发起比例
	    回复间隔与频率
	    对话连续性
	    单次聊天回合数
	    终止模式
	    对话时长
	    统计对话的总时长和平均回合时长。
	    话题主导权
	    话题转移与回避
	    对话深度
    4.重大事件概要：
        时间+地点+人物+事件+感情（开心）
        关系里程碑
    5.情感分析：
        情感词汇密度
        情感倾向与情绪流动
        情绪传递方向
        情感深度
        长期趋势
    6. 表达与沟通： 
        表达方式
        语言风格
        语言权力
    7.关系动态/角色定位
        依赖程度
        边界感
        角色标签
        权力平衡
    8.潜在诉求/未言之意
        显性需求
        隐性需求
        回避内容
    9.冲突与合作
        显性冲突
        隐性摩擦
        价值观差异
    10.社交风险/不足之处
    11.未来导向与可能性
    共同目标评价
    关系潜力评价
    风险预警
    12.交流建议
    13.推荐适合双方的礼物'''

@dataclass
class Config:
    IDLE_HOURS: int = 6  # 设定空闲时间（小时）
    MIN_MESSAGES_FOR_ANALYSIS: int = 3  # 最小消息数（用于分析）
    API_BASE_URL: str = r"https://api.siliconflow.com/v1/chat/completions"  # API地址
    MODEL_NAME: str = r"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 选择的模型
    OUTPUT_DIR: str = "output"  # 输出目录

#该函数仅对conversation_manager_txt使用
def generate_chart(talking_time, talking_week: list, talking_sender) -> None:
    # 聊天时段折线图
    plt.figure(figsize=(48, 10000))
    plt.subplot(2, 2, 1)
    plt.plot(range(24), talking_time[:, 0], marker='o', linestyle='-', color='b', label='talking_time')
    plt.xlabel('time')
    plt.ylabel('num')
    plt.xticks(range(24), [f'{str(i)}-{str(i + 1)}' for i in range(24)])
    for i in range(talking_time.shape[0]):
        plt.annotate(f'{talking_time[i, 0]}', (i, talking_time[i, 0]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    plt.title('talking_time')
    plt.legend()

    # 聊天周几环状图
    plt.subplot(2, 2, 3)
    plt.pie(talking_week, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], autopct='%1.1f%%', wedgeprops={'width': 0.3})
    plt.title('talking_week')
    plt.legend()

    # 聊天对象饼状图
    plt.subplot(2, 2, 4)
    talking_sender.keys()
    plt.pie(talking_sender.values(), labels=talking_sender.keys(), autopct='%1.1f%%')
    plt.title('talking_sender')
    plt.legend()
    plt.show()

class DeepSeekClient:
    """Handles all DeepSeek API interactions"""

    def __init__(self, api_key: str, base_url: str = Config.API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    def generate_summary(self, prompt: str) -> str:
        try:
            # 构建请求头，包含 API 密钥
            headers = {
                "Authorization":  self.api_key,
                "Content-Type": "application/json"
            }

            # 构建请求数据
            data = {
                "messages": [{'role': 'user', 'content': prompt}, ],
                "temperature": 0.7,
                "max_tokens": 4095,
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            }

            # 发送请求到 DeepSeek API
            response = requests.post(self.base_url, headers=headers, json=data)
            # 检查响应状态
            if response.status_code == 200:
                response_data = response.json()
                print(response_data)  # 打印返回的数据以供调试
                # 根据实际返回的结构进行提取
                return response_data['choices'][0]['message']['content'].strip()
            else:
                logging.error(f"DeepSeek API error, status code: {response.status_code}")
                raise Exception(f"Error: {response.status_code}")

        except Exception as e:
            logging.error(f"DeepSeek API error: {e}")
            raise

    def summarize_conversation(self,conversations):
        """使用ChatGPT总结对话内容"""  
        # 格式化对话内容
        formatted_messages = []
        for conversation in conversations:
            for msg in conversation['messages']:
                if msg['sender'] == 1 or msg['sender'] =='None\n' or msg['sender'] == '我':
                    formatted_messages.append(f'我:{msg['content']}')
                else:formatted_messages.append(f"{'对方'}: {msg['content']}")
            
            conversation_text = ",".join(formatted_messages)
        
        # 构建提示
        prompt = s + f"""
对话时间：{conversation['time']} 
对话内容：{conversation_text}

总结："""
        
        try:
            return self.generate_summary(prompt)
        except Exception as e:
            return f"总结失败: {str(e)}"


class ImageProcessor:
    """图像处理类"""
    def __init__(self, target_width=1170, split_height=2352):
        self.target_width = target_width
        self.split_height = split_height
        self.status_bar_ratio = 0.15  # 状态栏占比

    def process_image(self, image_path: str) -> List[np.ndarray]:
        """完整的图像处理流程"""
        raw_image = cv2.imread(image_path)
        resized = self._resize(raw_image)
        return self._split(resized)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        h, w = image.shape[:2]
        r = self.target_width / w
        dim = (self.target_width, int(h * r))
        return cv2.resize(image, dim)

    def _split(self, image: np.ndarray) -> List[np.ndarray]:
        """分割长图像"""
        h = image.shape[0]
        if h <= 3000:
            return [self._crop_initial(image)]
        
        chunks = []
        for i in range(h // self.split_height + 1):
            chunk = image[i * self.split_height:(i + 1) * self.split_height]
            if i == 0:
                chunk = self._crop_initial(chunk)
            chunks.append(chunk)
        return chunks

    def _crop_initial(self, image: np.ndarray) -> np.ndarray:
        """裁剪初始区域"""
        h = image.shape[0]
        return image[int(h * self.status_bar_ratio):, :]

class OCRProcessor:
    """OCR处理类"""
    def __init__(self):
        self.engine = PaddleOCR(use_angle_cls=True, lang="ch")
        
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """执行OCR识别"""
        result = self.engine.ocr(image, cls=True)
        return [self._format_result(line) for line in result[0]]

    @staticmethod
    def _format_result(line) -> Dict:
        """格式化OCR结果"""
        text = line[1][0]
        box = line[0]
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return {
            'text': text,
            'left': int(min(x_coords)),
            'top': int(min(y_coords)),
            'width': int(max(x_coords) - min(x_coords)),
            'height': int(max(y_coords) - min(y_coords))
        }

class MessageAnalyzer:
    """消息分析类"""
    def __init__(self):
        self.color_ranges = {
            'self': [
                (np.array([45, 130, 229]), np.array([55, 145, 255])),
                (np.array([70, 191, 178]), np.array([75, 209, 192])),
                (np.array([105, 170, 229]), np.array([121, 176, 255]))
            ],
            'other': [
                (np.array([0, 0.0, 230]), np.array([10, 10.0, 240])),
                (np.array([0, 0.0, 40]), np.array([5, 5.0, 43]))
            ]
        }
        self.time_pattern = re.compile(
            r'\d{1,2}月\d{1,2}日(中午|晚上|上午|下午)?\d{1,2}:\d{2}'
        )
        self.file_pattern = re.compile(r'\.(pdf|docx|xlsx)\s+\d+\.?\d*[MKG]?B')

    def analyze(self, ocr_data: List[Dict], image: np.ndarray) -> List[Dict]:
        """完整分析流程"""
        filtered = self._filter_data(ocr_data)
        grouped = self._group_messages(filtered, image)
        return [self._parse_block(block) for block in grouped]

    def _filter_data(self, data: List[Dict]) -> List[Dict]:
        """过滤无效数据"""
        return [item for item in data 
               if item['text'] not in ['微信电脑版', 'PDF']]

    def _group_messages(self, data: List[Dict], image: np.ndarray) -> List[List[Dict]]:
        """消息分组"""
        blocks = []
        current_block = []
        
        for item in sorted(data, key=lambda x: (x['top'], x['left'])):
            item['direction'] = self._detect_direction(image, item)
            
            if current_block and self._should_split(current_block[-1], item):
                blocks.append(current_block)
                current_block = []
            current_block.append(item)
        
        if current_block:
            blocks.append(current_block)
        return blocks

    def _detect_direction(self, image: np.ndarray, item: Dict) -> str:
        """检测消息方向"""
        roi = self._get_roi(image, item)
        if roi.size == 0:
            return 'unknown'
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self_ratio = self._calc_color_ratio(hsv, 'self')
        other_ratio = self._calc_color_ratio(hsv, 'other')
        return 'self' if self_ratio > other_ratio else 'other'

    def _get_roi(self, image: np.ndarray, item: Dict) -> np.ndarray:
        """获取检测区域"""
        return image[item['top']:item['top'] + item['height'],
                    item['left']:item['left'] + item['width']]

    def _calc_color_ratio(self, hsv: np.ndarray, color_type: str) -> float:
        """计算颜色占比"""
        max_ratio = 0.0
        for lower, upper in self.color_ranges[color_type]:
            mask = cv2.inRange(hsv, lower, upper)
            ratio = np.count_nonzero(mask) / mask.size
            max_ratio = max(max_ratio, ratio)
        return max_ratio  # 保底阈值

    def _should_split(self, last: Dict, current: Dict) -> bool:
        """判断是否需要分割消息块"""
        vertical_gap = current['top'] - (last['top'] + last['height'])
        return vertical_gap > 30 or current['direction'] != last['direction']

    def _parse_block(self, block: List[Dict]) -> Dict:
        """解析消息块"""
        return {
            'sender': self._get_sender(block),
            'time': self._get_time(block),
            'content': self._get_content(block),
            'type': self._get_type(block)
        }

    def _get_sender(self, block: List[Dict]) -> str:
        """获取发送者"""
        return '对方' if block[0]['direction'] == 'other' else '我'

    def _get_time(self, block: List[Dict]) -> str:
        """提取时间信息"""
        for item in block:
            if self.time_pattern.fullmatch(item['text']):
                return item['text']
        return ''

    def _get_content(self, block: List[Dict]) -> str:
        """合并消息内容"""
        return ''.join(item['text'] for item in block 
                      if not self.time_pattern.fullmatch(item['text']))

    def _get_type(self, block: List[Dict]) -> str:
        """判断消息类型"""
        if self.file_pattern.search(''.join(item['text'] for item in block)):
            return 'file' 
        else:
            return 'text'

class ChatProcessor:
    """聊天记录处理器"""
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.ocr_processor = OCRProcessor()
        self.msg_analyzer = MessageAnalyzer()
        self.output = [{'messages': [], 'time': None}]

    def process(self, image_path: str) -> List[Dict]:
        """处理完整流程"""
        for image in self.image_processor.process_image(image_path):
            ocr_data = self.ocr_processor.extract_text(image)
            analyzed = self.msg_analyzer.analyze(ocr_data, image)
            self._update_output(analyzed)
        return self.output

    def _update_output(self, messages: List[Dict]):
        """更新输出结果"""
        for msg in messages:
            if msg['time']:
                self.output.append({'messages': [], 'time': msg['time']})
            else:
                self.output[-1]['messages'].append({
                    'sender': msg['sender'],
                    'content': msg['content'],
                    'type': msg['type']
                })

class ChatAnalyzer:
    def __init__(self, data_file,output_dir:str = Config.OUTPUT_DIR):
        """Initialize chat analyzer with better error handling and logging"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_file = data_file
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_dir / 'analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        try:
            if data_file.endswith('.txt'):
                self.deepseek_client = None
                self.data_file = data_file
            else:    
                self.df = self._load_data(data_file)
                # self.conversation_manager = csv_conversation(self.df)
                self.deepseek_client = None
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise
    
    #获取txt文件聊天记录,直接得到conversation还有分析图表
    def conversation_manager_txt(self,idle_hour: int = Config.IDLE_HOURS):
        pre_time = None
        talking_time = np.zeros((24,1))#value =0代表0-1时间段
        talking_week = [0]*7
        talking_sender = defaultdict(int)
        with open(self.data_file,'r',encoding = 'UTF-8') as f:
            conversations = []
            current_conversations = []
            lines = iter(f.readlines())
            while True:
                try:
                    line = next(lines)
                except StopIteration :
                    conversations.append({
                        'messages':current_conversations,
                        'time':current_conversations[0]['start_time'],
                        'end_time':current_conversations[-1]['start_time']
                    })
                    break
                if line[0] == '\n':
                    line = next(lines)
                    if line[0] == '\n':
                        break
                if line[0] == '2' and line[1] == '0':
                    parts = line.split(' ', 2)
                    tmp = parts[0]+parts[1]
                    current_time = datetime.strptime(tmp,'%Y-%m-%d%H:%M:%S')
                    current_content = []
                    current_emotion=[]
                    talking_week[current_time.weekday()-1] += 1 
                    talking_time[current_time.hour,0] += 1
                    if pre_time is not None:
                        time_diff = (current_time - pre_time).total_seconds() / 3600  # 转换为小时
                        if time_diff >= idle_hour and current_conversations:
                            conversations.append(
                                {'messages':current_conversations,
                                'time':current_conversations[0]['start_time'],
                                'end_time':current_conversations[-1]['start_time']}
                            )
                            if len(conversations) == 20:
                                pass
                            current_conversations = []
                    a = next(lines)
                    while not a.startswith('\n'):
                        if a.startswith('【合并'):
                            current_content.append('聊天记录')
                            while not a.startswith('\n'):
                                a = next(lines)
                            for j in range(2):
                                a = next(lines)
                            break
                        current_content .append( ''.join(a))
                        emo=SnowNLP(a).sentiments
                        if emo>0 and emo<0.25:
                            current_emotion.append('负面')
                        elif emo<0.25 and emo<0.5:
                            current_emotion.append('较负面')  
                        elif emo<0.5 and emo<0.75: 
                            current_emotion.append('较积极') 
                        elif emo<1 and emo>0.75: 
                            current_emotion.append('积极')   
                        a = next(lines)
                    current_conversations.append({
                        'sender':str(parts[2]),
                        'start_time':current_time,
                        'content':current_content,
                        'emotion':current_emotion
                    })    
                    talking_sender[parts[2]] += 1
                    pre_time = current_time
            generate_chart(talking_time,talking_week,talking_sender)
            return conversations 
   
    #获取csv文件聊天记录，但无法直接得到分析图表
    def get_conversations_csv(self, idle_hour: int = Config.IDLE_HOURS) -> List[Dict]:
        conversations = []
        current_conversation = []
        prev_time = None
        for _, row in self.df.iterrows():
            current_time = row['DateTime']
            msg_content = row['MsgContent']
            
            # 解码消息内容
            if 'DecodedMsg' in row and pd.notna(row['DecodedMsg']):
                msg_content = row['DecodedMsg']
            
            if prev_time is not None:
                time_diff = (current_time - prev_time).total_seconds() / 3600  # 转换为小时
                
                if time_diff > idle_hour:
                    if current_conversation:
                        conversations.append({
                            'messages': current_conversation,
                            'time': current_conversation[0]['start_time'],
                            'end_time': current_conversation[-1]['start_time']
                        })
                    current_conversation = []
            
            current_conversation.append({
                'start_time': current_time,
                'sender': row['SenderUin'],
                'content': msg_content
            })
            prev_time = current_time
        
        # 添加最后一个对话
        if current_conversation:
            conversations.append({
                'messages': current_conversation,
                'time': current_conversation[0]['start_time'],
                'end_time': current_conversation[-1]['start_time']
            })
        
        return conversations
    
    #获取csv文件读取dataframe格式，还需进行get_conversation操作
    def _load_data(self, data_file) -> pd.DataFrame:
        df = pd.read_csv(data_file)
            
        # Convert CreateTime to datetime and localize to Shanghai timezone
        df['DateTime'] = pd.to_datetime(df['StrTime'], format='%Y-%m-%d %H:%M:%S')
        df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            
            # Map message types to content
        def get_message_content(row):
            if row['Type'] == 1:  # Text message
                    return row['StrContent']
            elif row['Type'] == 3:  # Image
                    return '[图片]'
            elif row['Type'] == 47:  # Emoji
                    return '[表情]'
            elif row['Type'] == 49:  # System message
                    return '[系统消息]'
            else:
                    return '[其他类型消息]'
            
        df['MsgContent'] = df.apply(get_message_content, axis=1)
            
        # Rename columns to match expected format
        df = df.rename(columns={
                'IsSender': 'SenderUin',
            })
            
        # Filter relevant columns
        df = df[['DateTime', 'SenderUin', 'MsgContent']]  
        return df   
    
    #以下方法对于conversation使用（无论txt还是csv）
    def _save_conversations(self, conversations: List[Dict]) -> None:
        """Save conversation data to file"""
        output_path = self.output_dir / 'conversations.json'
        try:
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(self._format_conversations_for_save(conversations), f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving conversations: {e}")
            raise


    def _format_conversations_for_save(self, conversations: List[Dict]) -> List[Dict]:
        """Format conversation data for JSON serialization"""
        formatted_conversations = []
        for conv in conversations:
            formatted_messages = []
            for msg in conv['messages']:
                formatted_msg = {
                    'time': msg['time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'sender': msg['sender'],
                    'content': msg['content']
                }
                formatted_messages.append(formatted_msg)
            
            formatted_conv = {
                'messages': formatted_messages,
                'end_time': conv['end_time'].strftime('%Y-%m-%d %H:%M:%S')
            }
            formatted_conversations.append(formatted_conv)
        return formatted_conversations
    
    #对csv文件使用
    def generate_activity_chart(self) -> None:
        """Generate activity chart showing monthly message counts from raw data"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        talking_time = np.zeros((24,1))#value =0代表0-1时间段
        talking_week = [0]*7
        talking_sender ={'对方': 0 , '自己' : 0}
        
        # 直接从原始DataFrame按月统计消息数
        for _, row in self.df.iterrows():
            tmp = row['DateTime']
            current_time = datetime.strptime(tmp,'%Y-%m-%d%H:%M:%S')
            sender = row['sender']
            talking_week[current_time.weekday()-1] += 1 
            talking_time[current_time.hour,0] += 1
            if sender == 1:
                talking_sender['自己'] += 1 
            else :talking_sender['对方'] += 1
        plt.figure(figsize=(48, 10000))
        plt.subplot(2, 2, 1)
        plt.plot(range(24), talking_time[:, 0], marker='o', linestyle='-', color='b', label='talking_time')
        plt.xlabel('time')
        plt.ylabel('num')
        plt.xticks(range(24), [f'{str(i)}-{str(i + 1)}' for i in range(24)])
        for i in range(talking_time.shape[0]):
            plt.annotate(f'{talking_time[i, 0]}', (i, talking_time[i, 0]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
        plt.title('talking_time')
        plt.legend()

        # 聊天周几环状图
        plt.subplot(2, 2, 3)
        plt.pie(talking_week, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], autopct='%1.1f%%', wedgeprops={'width': 0.3})
        plt.title('talking_week')
        plt.legend()

        # 聊天对象饼状图
        plt.subplot(2, 2, 4)
        plt.pie(talking_sender.values(), labels=talking_sender.keys(), autopct='%1.1f%%')
        plt.title('talking_sender')
        plt.legend()
        plt.show()

class ultimate_process:
    def __init__(self,filelist_path,output_dir,api_key,url:str = Config.API_BASE_URL):
        self.filelist_path = filelist_path
        self.api_key = api_key
        self.url = url
        self.output_dir = output_dir
    
    def get_conversation(self):
        conversations = []
        for file_path in self.filelist_path:
            if file_path.endswith('.txt'):
                tmp = ChatAnalyzer(file_path, self.output_dir)
                conversation = tmp.conversation_manager_txt()
                conversations.extend(conversation)
            elif file_path.endswith('.csv'):
                tmp = ChatAnalyzer( file_path, self.output_dir)
                conversation = tmp.get_conversations_csv()
                conversations.extend(conversation)
            else :
                tmp = ChatProcessor()
                result = tmp.process(r"C:\Users\26922\Pictures\baize\5.jpg")
                conversations.extend(result)
        return conversations
        
    def main(self):
        Ai = DeepSeekClient(self.api_key)
        conversations = self.get_conversation()
        print(conversations)
        answer = Ai.summarize_conversation(conversations)
        return answer
    

if __name__ == '__main__':
    
    filelist_path = [r'C:\Users\26922\Pictures\baize\5.jpg']#仅允许在其中添加txt,csv,和图片类型的文件，txt文件具有一定格式
    manager = ultimate_process(filelist_path=filelist_path,api_key='Bearer sk-edrloouofqgcirlrhrjhtqglfmxthkpspepupjxagxpjmvaz',output_dir=Config.OUTPUT_DIR)
    result = manager.main()
    print(result)