#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小批量测试程序 - 使用当前目录的样本数据验证完整流程
"""

import json
import os
import requests
import base64
import time
from typing import List, Dict, Optional
import csv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmallBatchProcessor:
    def __init__(self):
        """
        使用当前目录的样本数据进行批量处理测试（所有可用样本）
        """
        self.current_dir = "/Users/huangxinyue/Multi model distillation"
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct:free"
        self.results = []
        
        # 创建输出目录
        self.output_dir = os.path.join(self.current_dir, "batch_all_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_sample_data(self) -> List[Dict]:
        """
        加载当前目录的样本数据
        """
        logger.info("加载样本数据...")
        
        json_dir = os.path.join(self.current_dir, "json_samples")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]  # 处理所有可用样本
        
        samples = []
        for json_file in json_files:
            try:
                json_path = os.path.join(json_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                
                # 提取caption
                caption = ""
                if 'edge_media_to_caption' in post_data and post_data['edge_media_to_caption']['edges']:
                    caption = post_data['edge_media_to_caption']['edges'][0]['node']['text']
                
                post_id = post_data.get('id', json_file.replace('.json', ''))
                
                samples.append({
                    'post_id': post_id,
                    'caption': caption,
                    'json_file': json_file
                })
                
            except Exception as e:
                logger.warning(f"加载{json_file}失败: {e}")
        
        logger.info(f"成功加载{len(samples)}个样本")
        return samples
    
    def call_qwen_api(self, caption: str) -> Optional[dict]:
        """
        调用Qwen API进行性别偏见评分（纯文本版本）
        """
        try:
            # 构建API请求
            content = [
                {
                    "type": "text",
                    "text": f"请根据以下Instagram帖子的内容判断其性别倾向程度，性别倾向越强分数越高。\n\n要求：\n1. 先给出0到10之间的数字分数\n2. 然后用一句话解释打分原因\n3. 格式：分数: X.X | 原因: [解释]\n\n帖子内容：{caption}"
                }
            ]
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 100,  # 增加token数量以容纳解释
                "temperature": 0.1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multi-model-distillation",
                "X-Title": "Multi-Model Gender Bias Analysis"
            }
            
            # 发送API请求
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 解析分数和解释
                try:
                    # 尝试按格式解析: "分数: X.X | 原因: [解释]"
                    if '|' in content:
                        parts = content.split('|')
                        score_part = parts[0].strip()
                        reason_part = parts[1].strip() if len(parts) > 1 else ""
                        
                        # 提取分数
                        import re
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match:
                            score = float(score_match.group(1))
                            if 0 <= score <= 10:
                                # 提取解释
                                reason = reason_part.replace('原因:', '').replace('原因：', '').strip()
                                return {
                                    'score': score,
                                    'reason': reason,
                                    'raw_response': content
                                }
                        
                    # 备用解析方式：如果格式不标准，尝试提取数字
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            return {
                                'score': score,
                                'reason': content,  # 整个回复作为解释
                                'raw_response': content
                            }
                    
                    logger.warning(f"无法解析返回的分数和解释: {content}")
                    return None
                    
                except Exception as e:
                    logger.warning(f"解析响应时出错: {e}, 原始内容: {content}")
                    return None
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return None
    
    def process_samples(self, samples: List[Dict]) -> None:
        """
        处理样本并保存结果
        """
        logger.info(f"开始处理{len(samples)}个样本...")
        
        for i, sample in enumerate(samples):
            if (i + 1) % 5 == 0 or i == 0 or i == len(samples) - 1:
                logger.info(f"处理第{i+1}/{len(samples)}个样本: {sample['post_id']}")
            
            try:
                # 调用API获取分数和解释
                api_result = self.call_qwen_api(sample['caption'])
                
                # 保存结果
                result = {
                    'post_id': sample['post_id'],
                    'caption': sample['caption'][:200] + "..." if len(sample['caption']) > 200 else sample['caption'],
                    'gender_bias_score': api_result['score'] if api_result else None,
                    'explanation': api_result['reason'] if api_result else None,
                    'raw_response': api_result['raw_response'] if api_result else None,
                    'json_file': sample['json_file']
                }
                
                self.results.append(result)
                
                if api_result is not None:
                    logger.info(f"✅ 获得分数: {api_result['score']}")
                    logger.info(f"💭 解释: {api_result['reason']}")
                else:
                    logger.warning(f"⚠️ 分数获取失败")
                
                # 每5个样本显示一次进度统计
                if (i + 1) % 5 == 0 or i == len(samples) - 1:
                    success_count = len([r for r in self.results if r['gender_bias_score'] is not None])
                    logger.info(f"📊 已处理{i+1}个，成功{success_count}个，成功率{success_count/(i+1)*100:.1f}%")
                
                # API限流控制
                time.sleep(1.5)  # 每次请求间隔1.5秒
                
            except Exception as e:
                logger.error(f"处理样本时出错 (PostID: {sample['post_id']}): {e}")
    
    def save_results(self) -> None:
        """
        保存处理结果
        """
        if not self.results:
            logger.warning("没有结果数据可保存")
            return
        
        # 保存CSV文件
        csv_path = os.path.join(self.output_dir, "batch_all_results.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"结果已保存到: {csv_path}")
            
        except Exception as e:
            logger.error(f"保存CSV结果时出错: {e}")
        
        # 保存JSON文件
        json_path = os.path.join(self.output_dir, "batch_all_results.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON结果已保存到: {json_path}")
            
        except Exception as e:
            logger.error(f"保存JSON结果时出错: {e}")
    
    def generate_summary(self) -> None:
        """
        生成结果摘要
        """
        if not self.results:
            logger.warning("没有结果数据，无法生成摘要")
            return
        
        valid_scores = [r['gender_bias_score'] for r in self.results if r['gender_bias_score'] is not None]
        
        if valid_scores:
            summary = {
                'total_samples': len(self.results),
                'valid_scores_count': len(valid_scores),
                'success_rate': len(valid_scores) / len(self.results) * 100,
                'mean_score': sum(valid_scores) / len(valid_scores),
                'min_score': min(valid_scores),
                'max_score': max(valid_scores),
                'scores': valid_scores
            }
            
            logger.info("=== 处理结果摘要 ===")
            logger.info(f"总样本数: {summary['total_samples']}")
            logger.info(f"成功获得分数: {summary['valid_scores_count']}")
            logger.info(f"成功率: {summary['success_rate']:.1f}%")
            logger.info(f"平均分数: {summary['mean_score']:.2f}")
            logger.info(f"分数范围: {summary['min_score']:.1f} - {summary['max_score']:.1f}")
            logger.info(f"所有分数: {summary['scores']}")
            
            # 保存摘要
            summary_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"摘要已保存到: {summary_path}")
        else:
            logger.warning("没有有效的分数数据")
    
    def run(self) -> None:
        """
        运行小批量处理流程
        """
        logger.info("🚀 开始处理所有可用样本...")
        
        # 1. 加载样本数据
        samples = self.load_sample_data()
        if not samples:
            logger.error("❌ 样本加载失败，程序退出")
            return
        
        # 2. 处理样本
        self.process_samples(samples)
        
        # 3. 保存结果
        self.save_results()
        
        # 4. 生成摘要
        self.generate_summary()
        
        logger.info("✅ 所有样本批量处理完成！")
        logger.info("💡 如果结果满意，可以运行完整的extract_and_score_samples.py处理50K样本")


def main():
    """
    主程序入口
    """
    processor = SmallBatchProcessor()
    processor.run()


if __name__ == "__main__":
    main()
