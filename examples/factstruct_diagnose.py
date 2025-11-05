#!/usr/bin/env python3


"""
FactStruct Stage 1 诊断脚本

用于诊断 LLM 连接问题。
"""

import sys
from langchain_core.messages import HumanMessage


def check_llm_config():
    """检查 LLM 配置"""
    print("=== 检查 LLM 配置 ===\n")

    try:
        from src.config import load_yaml_config
        from pathlib import Path
        from src.config.agents import AGENT_LLM_MAP

        conf_path = Path("conf.yaml").resolve()
        conf = load_yaml_config(str(conf_path))

        llm_type = AGENT_LLM_MAP.get("outline", "basic")
        print(f"✓ Outline agent 映射到 LLM 类型: {llm_type}")

        llm_config_key = f"{llm_type.upper()}_MODEL"
        llm_conf = conf.get(llm_config_key, {})

        if not llm_conf:
            print(f"✗ 错误: 配置文件中没有找到 {llm_config_key}")
            return False

        print(f"✓ 找到 {llm_config_key} 配置:")
        print(f"  - base_url: {llm_conf.get('base_url', '未设置')}")
        print(f"  - model: {llm_conf.get('model', '未设置')}")
        print(f"  - api_key: {'已设置' if llm_conf.get('api_key') else '未设置'}")

        if not llm_conf.get("api_key"):
            print(f"✗ 错误: API key 未设置")
            return False

        return True

    except Exception as e:
        print(f"✗ 配置检查失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_connection():
    """测试 LLM 连接"""
    print("\n=== 测试 LLM 连接 ===\n")

    try:
        from src.llms.llm import get_llm_by_type
        from src.config.agents import AGENT_LLM_MAP

        llm_type = AGENT_LLM_MAP.get("outline", "basic")
        print(f"获取 LLM 实例 (类型: {llm_type})...")

        llm = get_llm_by_type(llm_type)
        print(f"✓ LLM 实例创建成功: {type(llm).__name__}")

        print("发送测试消息...")
        messages = [HumanMessage(content="请回复'测试成功'")]

        response = llm.invoke(messages)
        print(f"✓ LLM 响应成功: {response.content[:100]}")

        return True

    except Exception as e:
        print(f"✗ LLM 连接测试失败: {e}")
        import traceback

        print("\n详细错误信息:")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("FactStruct Stage 1 诊断工具\n")
    print("=" * 50 + "\n")

    # 检查配置
    config_ok = check_llm_config()
    if not config_ok:
        print("\n请先修复配置问题。")
        sys.exit(1)

    # 测试连接
    connection_ok = test_llm_connection()
    if not connection_ok:
        print("\n连接测试失败。可能的原因:")
        print("1. API key 无效或过期")
        print("2. base_url 配置错误")
        print("3. 网络连接问题")
        print("4. API 服务限流或不可用")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✓ 所有检查通过！可以正常运行 FactStruct Stage 1。")


if __name__ == "__main__":
    main()
