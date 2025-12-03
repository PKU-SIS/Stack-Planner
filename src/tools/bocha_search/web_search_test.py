# import pytest

# from src.rag_web import create_web_searcher

# def test_english_web_searcher():
#     web_searcher = create_web_searcher("english")
#     assert web_searcher is not None
#     # result = web_searcher.search("why is the sky blue?")
#     # assert isinstance(result, list)


# def test_chinese_web_searcher():
#     web_searcher = create_web_searcher("chinese")
#     assert web_searcher is not None
#     # result = web_searcher.search("为什么天空是蓝色的？")
#     # assert isinstance(result, list)

# def test_unsupported_language():
#     with pytest.raises(ValueError):
#         create_web_searcher("unsupported")


from src.tools.bocha_search import create_web_searcher

def test_english_web_searcher(api_key):
    print("Running English Web Searcher test...")
    web_searcher = create_web_searcher("english",api_key)
    assert web_searcher is not None
    result = web_searcher.search("why is the sky blue?")
    assert isinstance(result, list)
    print("English test passed ✓\n")
    print("result",result)


def test_chinese_web_searcher(api_key):
    print("Running Chinese Web Searcher test...")
    web_searcher = create_web_searcher("chinese",api_key)
    assert web_searcher is not None
    result = web_searcher.search("为什么天是蓝的?")
    assert isinstance(result, list)
    print("Chinese test passed ✓\n")
    print("result",result)


def test_unsupported_language(api_key):
    print("Running unsupported language test...")
    try:
        create_web_searcher("unsupported",api_key)
    except ValueError:
        print("Unsupported language test passed ✓\n")
        return
    # 如果没有抛错，就说明失败
    print("Unsupported language test FAILED ✗\n")


if __name__ == "__main__":
    #bocha的
    api_key_zh="sk-7e8e5cab9e7d48b29a149ba2405a3395"
    #对应链接
    #"https://api.bocha.cn/v1/web-search"
    #"https://api.bochaai.com/v1/web-search"
    
    #LangSearch(日宏的)
    api_key_en="sk-f2452ad87c2749fb93b24fc03d102265"
    # 对应链接
    # "https://api.langsearch.com/v1/web-search"

    test_english_web_searcher(api_key_en)
    test_chinese_web_searcher(api_key_zh)
    test_unsupported_language(api_key_zh)
