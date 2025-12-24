#!/usr/bin/env python3
"""
LLM 流式输出性能基准测试 v2

测试指标：
- TTFT (Time to First Token): 首字延迟
- Generation Speed: 生成速度 (tokens/s)
- Max Concurrent Requests: 最大并发请求数
- Total Throughput: 总吞吐量 (tokens/s)
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI, AsyncOpenAI


# ============ 配置 ============

API_KEY = "not-needed"
BASE_URL = "http://10.1.1.212:8080/v1"
MODEL = "Qwen2.5-32B-Instruct"
MAX_TOKENS = 2048
TEST_PROMPT = "请详细解释一下量子计算的基本原理，包括量子比特、叠加态、纠缠等概念，以及它与经典计算的区别。"

# 并发测试的 TTFT 阈值（秒），超过此值视为不可用
TTFT_THRESHOLD = 5.0


# ============ 数据结构 ============

@dataclass
class StreamingMetrics:
    """流式输出性能指标"""
    ttft: float = 0.0
    total_time: float = 0.0
    generation_time: float = 0.0
    total_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    success: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    ttft_results: List[float] = field(default_factory=list)
    speed_results: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    total_tokens_list: List[int] = field(default_factory=list)
    success_count: int = 0
    fail_count: int = 0
    errors: List[str] = field(default_factory=list)
    # 新增：总吞吐量相关
    total_output_tokens: int = 0
    total_duration: float = 0.0


# ============ 单次流式请求 ============

def run_streaming_request(prompt: str, verbose: bool = True) -> StreamingMetrics:
    """执行单次流式请求并测量性能指标"""
    metrics = StreamingMetrics()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    try:
        request_start_time = time.perf_counter()
        first_token_received = False
        
        if verbose:
            print("      --- 实时输出 ---", flush=True)
        
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        
        first_token_time = None
        chunk_count = 0
        
        for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                metrics.total_tokens = chunk.usage.total_tokens
                metrics.output_tokens = chunk.usage.completion_tokens
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunk_count += 1
                
                if not first_token_received:
                    metrics.ttft = time.perf_counter() - request_start_time
                    first_token_time = time.perf_counter()
                    first_token_received = True
                    if verbose:
                        print(f"\n      [首字到达] TTFT={metrics.ttft:.3f}s\n", flush=True)
                
                if verbose:
                    print(content, end="", flush=True)
        
        metrics.total_time = time.perf_counter() - request_start_time
        
        if first_token_time:
            metrics.generation_time = time.perf_counter() - first_token_time
        
        # 如果 API 没返回 usage，用 chunk 数估算
        if metrics.output_tokens == 0:
            metrics.output_tokens = chunk_count
        
        if metrics.generation_time > 0:
            metrics.tokens_per_second = metrics.output_tokens / metrics.generation_time
        elif metrics.total_time > 0:
            metrics.tokens_per_second = metrics.output_tokens / metrics.total_time
        
        metrics.success = True
        
        if verbose:
            print(f"\n      --- 输出结束 ---", flush=True)
            print(f"      [完成] 总耗时 {metrics.total_time:.2f}s, 生成 {metrics.output_tokens} tokens", flush=True)
        
    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"\n      [错误] {metrics.error}", flush=True)
    
    return metrics


async def run_streaming_request_async(prompt: str, request_id: int = 0, max_tokens: int = 256) -> StreamingMetrics:
    """异步执行单次流式请求（用于并发测试）"""
    metrics = StreamingMetrics()
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    try:
        request_start_time = time.perf_counter()
        first_token_received = False
        first_token_time = None
        chunk_count = 0  # 用于估算 tokens
        
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        
        async for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                metrics.total_tokens = chunk.usage.total_tokens
                metrics.output_tokens = chunk.usage.completion_tokens
            
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_count += 1  # 统计有内容的 chunk 数
                if not first_token_received:
                    metrics.ttft = time.perf_counter() - request_start_time
                    first_token_time = time.perf_counter()
                    first_token_received = True
        
        metrics.total_time = time.perf_counter() - request_start_time
        if first_token_time:
            metrics.generation_time = time.perf_counter() - first_token_time
        
        # 修复：如果 API 没返回 output_tokens，用 chunk_count 估算
        if metrics.output_tokens == 0:
            metrics.output_tokens = chunk_count
        
        # 基于生成时间计算速度
        if metrics.generation_time > 0 and metrics.output_tokens > 0:
            metrics.tokens_per_second = metrics.output_tokens / metrics.generation_time
        elif metrics.total_time > 0 and metrics.output_tokens > 0:
            metrics.tokens_per_second = metrics.output_tokens / metrics.total_time
        
        metrics.success = True
        
    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)}"
    
    return metrics


# ============ TTFT & Speed 测试 ============

def run_ttft_speed_benchmark(iterations: int = 3, verbose: bool = True) -> BenchmarkResult:
    """运行 TTFT 和 Generation Speed 基准测试"""
    print(f"\n{'='*60}")
    print(f"📊 TTFT & Generation Speed 测试 ({iterations} 次迭代)")
    print(f"{'='*60}\n")
    
    result = BenchmarkResult()
    
    for i in range(iterations):
        print(f"迭代 {i+1}/{iterations}:")
        metrics = run_streaming_request(TEST_PROMPT, verbose=verbose)
        
        if metrics.success:
            result.ttft_results.append(metrics.ttft)
            result.speed_results.append(metrics.tokens_per_second)
            result.total_times.append(metrics.total_time)
            result.total_tokens_list.append(metrics.output_tokens)
            result.success_count += 1
            print(f"    TTFT: {metrics.ttft:.3f}s | "
                  f"Speed: {metrics.tokens_per_second:.1f} tokens/s | "
                  f"GenTime: {metrics.generation_time:.2f}s | "
                  f"Tokens: {metrics.output_tokens}")
        else:
            result.fail_count += 1
            result.errors.append(metrics.error or "Unknown error")
            print(f"    ❌ 失败: {metrics.error}")
        
        print()
    
    return result


# ============ 并发测试 ============

async def run_concurrent_test(concurrency: int, max_tokens: int = 256) -> BenchmarkResult:
    """运行指定并发数的测试"""
    result = BenchmarkResult()
    
    start_time = time.perf_counter()
    
    tasks = [
        run_streaming_request_async(TEST_PROMPT, i, max_tokens)
        for i in range(concurrency)
    ]
    
    metrics_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    result.total_duration = time.perf_counter() - start_time
    
    for metrics in metrics_list:
        if isinstance(metrics, Exception):
            result.fail_count += 1
            result.errors.append(str(metrics))
        elif metrics.success:
            result.ttft_results.append(metrics.ttft)
            result.speed_results.append(metrics.tokens_per_second)
            result.total_times.append(metrics.total_time)
            tokens = metrics.output_tokens if metrics.output_tokens > 0 else metrics.total_tokens
            result.total_tokens_list.append(tokens)
            result.total_output_tokens += tokens
            result.success_count += 1
        else:
            result.fail_count += 1
            result.errors.append(metrics.error or "Unknown error")
    
    return result


def run_concurrency_benchmark(max_concurrency: int = 10, step: int = 2, start: int = 1, 
                               max_tokens: int = 256, ttft_threshold: float = TTFT_THRESHOLD) -> dict:
    """运行最大并发请求数测试"""
    print(f"\n{'='*60}")
    print(f"📊 Max Concurrent Requests 测试 (max_tokens={max_tokens})")
    print(f"   TTFT 阈值: {ttft_threshold}s (超过视为不可用)")
    print(f"{'='*60}\n")
    
    results = {}
    
    if step == 0:
        concurrency_list = [start]
    else:
        concurrency_list = list(range(start, max_concurrency + 1, step))
        if max_concurrency not in concurrency_list:
            concurrency_list.append(max_concurrency)
    
    max_usable_concurrency = 0
    
    for concurrency in concurrency_list:
        print(f"测试并发数: {concurrency}")
        
        result = asyncio.run(run_concurrent_test(concurrency, max_tokens))
        results[concurrency] = result
        
        if result.success_count > 0:
            avg_ttft = statistics.mean(result.ttft_results)
            avg_speed = statistics.mean(result.speed_results) if result.speed_results else 0
            total_throughput = result.total_output_tokens / result.total_duration if result.total_duration > 0 else 0
            
            # 判断是否可用
            is_usable = avg_ttft <= ttft_threshold and result.fail_count == 0
            status = "✅" if is_usable else "⚠️"
            
            if is_usable:
                max_usable_concurrency = concurrency
            
            print(f"  {status} 成功: {result.success_count}/{concurrency} | "
                  f"平均 TTFT: {avg_ttft:.3f}s | "
                  f"平均 Speed: {avg_speed:.1f} tokens/s | "
                  f"总吞吐: {total_throughput:.1f} tokens/s")
            
            if not is_usable:
                print(f"     (TTFT {avg_ttft:.1f}s > 阈值 {ttft_threshold}s，视为不可用)")
        else:
            print(f"  ❌ 全部失败!")
        
        # 如果失败率超过 50%，停止测试
        if result.fail_count > concurrency / 2:
            print(f"\n⚠️ 失败率过高，停止并发测试")
            break
    
    # 记录最大可用并发数
    results['max_usable'] = max_usable_concurrency
    
    return results


# ============ 结果汇总 ============

def print_summary(ttft_result: Optional[BenchmarkResult], concurrency_results: Optional[dict] = None,
                  ttft_threshold: float = TTFT_THRESHOLD):
    """打印测试结果汇总"""
    print(f"\n{'='*60}")
    print(f"📈 测试结果摘要")
    print(f"{'='*60}\n")
    
    print(f"测试配置:")
    print(f"  - API 地址: {BASE_URL}")
    print(f"  - 模型: {MODEL}")
    if ttft_result:
        print(f"  - 成功率: {ttft_result.success_count}/{ttft_result.success_count + ttft_result.fail_count}")
    print()
    
    if ttft_result and ttft_result.ttft_results:
        print("【TTFT (Time to First Token)】")
        print(f"  最小值: {min(ttft_result.ttft_results):.3f}s")
        print(f"  最大值: {max(ttft_result.ttft_results):.3f}s")
        print(f"  平均值: {statistics.mean(ttft_result.ttft_results):.3f}s")
        if len(ttft_result.ttft_results) > 1:
            print(f"  标准差: {statistics.stdev(ttft_result.ttft_results):.3f}s")
        print(f"  P50:    {statistics.median(ttft_result.ttft_results):.3f}s")
        print()
    
    if ttft_result and ttft_result.speed_results:
        print("【Generation Speed】")
        print(f"  最小值: {min(ttft_result.speed_results):.1f} tokens/s")
        print(f"  最大值: {max(ttft_result.speed_results):.1f} tokens/s")
        print(f"  平均值: {statistics.mean(ttft_result.speed_results):.1f} tokens/s")
        if len(ttft_result.speed_results) > 1:
            print(f"  标准差: {statistics.stdev(ttft_result.speed_results):.1f} tokens/s")
        print(f"  P50:    {statistics.median(ttft_result.speed_results):.1f} tokens/s")
        print()
    
    if ttft_result and ttft_result.total_tokens_list:
        print("【Total Tokens】")
        print(f"  最小值: {min(ttft_result.total_tokens_list)}")
        print(f"  最大值: {max(ttft_result.total_tokens_list)}")
        print(f"  平均值: {statistics.mean(ttft_result.total_tokens_list):.0f}")
        print()
    
    if concurrency_results:
        print("【Max Concurrent Requests】")
        max_usable = concurrency_results.get('max_usable', 0)
        
        # 找出技术上成功的最大并发（不考虑 TTFT）
        max_technical = 0
        for concurrency, result in concurrency_results.items():
            if isinstance(concurrency, int) and result.success_count == concurrency:
                max_technical = max(max_technical, concurrency)
        
        print(f"  ✅ 最大可用并发数 (TTFT<{ttft_threshold}s): {max_usable if max_usable > 0 else '无'}")
        print(f"  📊 最大技术并发数 (仅看成功率): {max_technical}")
        
        # 显示各并发级别的吞吐量
        print(f"\n  【各并发级别吞吐量】")
        for concurrency, result in sorted((k, v) for k, v in concurrency_results.items() if isinstance(k, int)):
            if isinstance(concurrency, int) and result.success_count > 0:
                throughput = result.total_output_tokens / result.total_duration if result.total_duration > 0 else 0
                avg_ttft = statistics.mean(result.ttft_results)
                status = "✅" if avg_ttft <= ttft_threshold else "⚠️"
                print(f"    {status} {concurrency:4d} 并发: {throughput:7.1f} tokens/s (TTFT={avg_ttft:.2f}s)")
        print()
    
    if ttft_result and ttft_result.errors:
        print("【错误汇总】")
        for i, err in enumerate(ttft_result.errors[:5], 1):
            print(f"  {i}. {err}")
        if len(ttft_result.errors) > 5:
            print(f"  ... 还有 {len(ttft_result.errors) - 5} 个错误")
        print()


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="LLM 流式输出性能基准测试 v2")
    parser.add_argument("-n", "--iterations", type=int, default=3,
                        help="TTFT/Speed 测试迭代次数 (默认: 3)")
    parser.add_argument("-c", "--max-concurrency", type=int, default=0,
                        help="最大并发测试数 (默认: 0, 不测试并发)")
    parser.add_argument("--concurrency-start", type=int, default=1,
                        help="并发测试起始值 (默认: 1)")
    parser.add_argument("--concurrency-step", type=int, default=2,
                        help="并发测试步长 (默认: 2, 设为 0 则只测试起始值)")
    parser.add_argument("--concurrency-only", action="store_true",
                        help="只运行并发测试，跳过 TTFT/Speed 测试")
    parser.add_argument("--concurrency-tokens", type=int, default=256,
                        help="并发测试时的 max_tokens (默认: 256)")
    parser.add_argument("--ttft-threshold", type=float, default=TTFT_THRESHOLD,
                        help=f"TTFT 阈值秒数，超过视为不可用 (默认: {TTFT_THRESHOLD})")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="安静模式，不显示实时输出")
    parser.add_argument("--prompt", type=str, default=None,
                        help="自定义测试 prompt")
    args = parser.parse_args()
    
    global TEST_PROMPT
    if args.prompt:
        TEST_PROMPT = args.prompt
    
    print(f"\n{'='*60}")
    print(f"🚀 LLM 流式输出性能基准测试 v2")
    print(f"{'='*60}\n")
    
    print(f"配置:")
    print(f"  - API 地址: {BASE_URL}")
    print(f"  - 模型: {MODEL}")
    if not args.concurrency_only:
        print(f"  - TTFT/Speed 迭代次数: {args.iterations}")
    print(f"  - 最大并发测试: {'跳过' if args.max_concurrency == 0 else args.max_concurrency}")
    if args.max_concurrency > 0:
        print(f"  - 并发测试 max_tokens: {args.concurrency_tokens}")
        print(f"  - TTFT 可用阈值: {args.ttft_threshold}s")
    
    ttft_result = None
    if not args.concurrency_only:
        ttft_result = run_ttft_speed_benchmark(args.iterations, verbose=not args.quiet)
    
    concurrency_results = None
    if args.max_concurrency > 0:
        concurrency_results = run_concurrency_benchmark(
            args.max_concurrency, 
            step=args.concurrency_step,
            start=args.concurrency_start,
            max_tokens=args.concurrency_tokens,
            ttft_threshold=args.ttft_threshold
        )
    
    print_summary(ttft_result, concurrency_results, args.ttft_threshold)


if __name__ == "__main__":
    main()
