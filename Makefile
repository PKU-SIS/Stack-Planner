.PHONY: lint format install-dev serve test coverage

install-dev:
	uv pip install -e ".[dev]" && uv pip install -e ".[test]"

format:
	uv run black --preview .

lint:
	uv run black --check .

serve:
	uv run server.py --reload

test:
	uv run pytest tests/

langgraph-dev:
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking

coverage:
	uv run pytest --cov=src tests/ --cov-report=term-missing --cov-report=xml
	
run1:
	uv run main.py "以后你给我解释治疗建议时直接告诉我“现在严不严重、我现在先做什么”，别一上来讲太多背景。" --enable_memory=True

run11:
	uv run main.py "我今天透析后胸闷，还有点喘，你怎么看？" --enable_memory=True

run2:
	uv run main.py "我今天透析后胸闷，还有点喘，你怎么看？"
	