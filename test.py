#!/usr/bin/env python3
"""
LLM 性能测试脚本
测试指标：首token延迟、平均输出速度、完整输出时间
每个模型测试5次，记录每次结果和平均值
"""

import time
import requests
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TestResult:
    """单次测试结果"""
    first_token_latency: float  # 首token延迟(秒)
    avg_output_speed: float     # 平均输出速度(tokens/秒)
    total_time: float           # 完整输出时间(秒)
    output_tokens: int          # 输出token数量
    error: Optional[str] = None

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model: str
    api_key: str
    api_base: str

# 模型配置
MODELS = [
    ModelConfig(
        name="Qwen3-30B-A3B (DashScope)",
        model="qwen3-30b-a3b-instruct-2507",
        api_key="sk-8dd3264e37d3474398f8ea5dc586cd8a",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    ModelConfig(
        name="Qwen3-14B-AWQ (Local)",
        model="Qwen3-14B-AWQ",
        api_key="sk-8dd3264e37d3474398f8ea5dc586cd8a",
        api_base="http://10.0.169.144:8003/v1"
    ),
    ModelConfig(
        name="Qwen3-30B-A3B (Image-Derivative)",
        model="Qwen3-30B-A3B-Instruct-2507",
        api_key="sk-tXTGCiEnCYzaNu1EIisg0FuRTH9Qd16A",
        api_base="https://llm.image-derivative.com/open-api/maas/admin/v1"
    ),
]

# 生成约1000 token的输入文本
def generate_input_text() -> str:
    """生成约1000 token的输入文本"""
    base_text = """
请详细分析以下关于人工智能发展的内容，并给出你的见解：

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、
图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

机器学习是人工智能的核心，是使计算机具有智能的根本途径。机器学习理论主要是设计和分析一些让计算机
可以自动"学习"的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。

深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。深度学习的概念源于
人工神经网络的研究，含多个隐藏层的多层感知器就是一种深度学习结构。深度学习通过组合低层特征形成更加
抽象的高层表示属性类别或特征，以发现数据的分布式特征表示。

自然语言处理是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；自然语言处理包括
多方面和步骤，基本有认知、理解、生成等部分。自然语言认知和理解是让电脑把输入的语言变成有意思的符号
和关系，然后根据目的再处理。自然语言生成系统则是把计算机数据转化为自然语言。

计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是指用摄影机和电脑代替人眼对目标进行识别、
跟踪和测量等机器视觉，并进一步做图形处理，使电脑处理成为更适合人眼观察或传送给仪器检测的图像。

强化学习是机器学习的一个重要分支，是多学科多领域交叉的一个产物，它的本质是解决决策问题，即自动进行
决策，并且可以做连续决策。强化学习的目标是学习一个策略，使得智能体在与环境的交互过程中获得最大的累积奖励。

大语言模型是一种基于深度学习的自然语言处理模型，通过在大规模文本数据上进行预训练，学习语言的统计规律
和语义信息。这些模型通常具有数十亿甚至数千亿的参数，能够理解和生成人类语言，执行各种自然语言处理任务。

请从以下几个方面进行分析：
1. 人工智能的发展历程和重要里程碑
2. 当前人工智能技术的主要应用领域
3. 人工智能面临的挑战和伦理问题
4. 未来人工智能的发展趋势和可能的突破方向
5. 人工智能对社会和经济的影响

请给出详细、有深度的分析。
"""
    return base_text

def test_model_streaming(config: ModelConfig, input_text: str) -> TestResult:
    """测试单个模型的流式输出性能"""
    url = f"{config.api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": input_text}],
        "stream": True,
        "max_tokens": 512
    }
    
    try:
        start_time = time.time()
        first_token_time = None
        output_tokens = 0
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                # 简单估算token数（中文约1.5字符/token，英文约4字符/token）
                                output_tokens += max(1, len(content) // 2)
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        
        if first_token_time is None:
            return TestResult(0, 0, 0, 0, error="No tokens received")
        
        first_token_latency = first_token_time - start_time
        total_time = end_time - start_time
        generation_time = end_time - first_token_time
        avg_speed = output_tokens / generation_time if generation_time > 0 else 0
        
        return TestResult(
            first_token_latency=first_token_latency,
            avg_output_speed=avg_speed,
            total_time=total_time,
            output_tokens=output_tokens
        )
        
    except requests.exceptions.Timeout:
        return TestResult(0, 0, 0, 0, error="Request timeout")
    except requests.exceptions.RequestException as e:
        return TestResult(0, 0, 0, 0, error=str(e))
    except Exception as e:
        return TestResult(0, 0, 0, 0, error=str(e))

def run_tests(num_tests: int = 5):
    """运行所有测试"""
    input_text = generate_input_text()
    print(f"输入文本长度: {len(input_text)} 字符 (约 {len(input_text)//2} tokens)")
    print("=" * 100)
    
    for config in MODELS:
        print(f"\n{'='*100}")
        print(f"测试模型: {config.name}")
        print(f"Model ID: {config.model}")
        print(f"API Base: {config.api_base}")
        print("=" * 100)
        
        results: List[TestResult] = []
        
        for i in range(num_tests):
            print(f"\n第 {i+1} 次测试...")
            result = test_model_streaming(config, input_text)
            results.append(result)
            
            if result.error:
                print(f"  错误: {result.error}")
            else:
                print(f"  首token延迟: {result.first_token_latency:.3f} 秒")
                print(f"  平均输出速度: {result.avg_output_speed:.2f} tokens/秒")
                print(f"  完整输出时间: {result.total_time:.3f} 秒")
                print(f"  输出token数: {result.output_tokens}")
            
            # 测试间隔，避免请求过于频繁
            if i < num_tests - 1:
                time.sleep(2)
        
        # 计算平均值（排除错误的结果）
        valid_results = [r for r in results if r.error is None]
        
        print(f"\n{'-'*50}")
        print(f"模型 {config.name} 测试汇总:")
        print(f"成功测试次数: {len(valid_results)}/{num_tests}")
        
        if valid_results:
            avg_first_token = sum(r.first_token_latency for r in valid_results) / len(valid_results)
            avg_speed = sum(r.avg_output_speed for r in valid_results) / len(valid_results)
            avg_total_time = sum(r.total_time for r in valid_results) / len(valid_results)
            avg_tokens = sum(r.output_tokens for r in valid_results) / len(valid_results)
            
            print(f"平均首token延迟: {avg_first_token:.3f} 秒")
            print(f"平均输出速度: {avg_speed:.2f} tokens/秒")
            print(f"平均完整输出时间: {avg_total_time:.3f} 秒")
            print(f"平均输出token数: {avg_tokens:.0f}")
            
            # 详细结果表格
            print(f"\n详细结果:")
            print(f"{'测试次数':<10} {'首token延迟(秒)':<18} {'输出速度(t/s)':<16} {'总时间(秒)':<14} {'输出tokens':<12}")
            print("-" * 70)
            for i, r in enumerate(results):
                if r.error:
                    print(f"{i+1:<10} {'错误: ' + r.error[:40]}")
                else:
                    print(f"{i+1:<10} {r.first_token_latency:<18.3f} {r.avg_output_speed:<16.2f} {r.total_time:<14.3f} {r.output_tokens:<12}")
            print("-" * 70)
            print(f"{'平均':<10} {avg_first_token:<18.3f} {avg_speed:<16.2f} {avg_total_time:<14.3f} {avg_tokens:<12.0f}")

if __name__ == "__main__":
    print("LLM 性能测试")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    run_tests(num_tests=5)
