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


from src.rag_web import create_web_searcher

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
    api_key="sk-7e8e5cab9e7d48b29a149ba2405a3395"
    test_english_web_searcher(api_key)
    test_chinese_web_searcher(api_key)
    test_unsupported_language(api_key)
