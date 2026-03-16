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
	
run:
	uv run main.py "我希望以后每次问你颜色是什么，你都告诉我它的对比色。紫色是什么？" --enable_memory=True

run2:
	uv run main.py "红色是什么？" --enable_memory=True