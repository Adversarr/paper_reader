.PHONY: setup extract process clean

setup:
	pip install -r requirements.txt
	pip install -e .

extract:
	python extractor.py

process:
	python main.py

clean:
	rm -rf __pycache__
	rm -rf paper_reader/__pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make setup     - Install dependencies"
	@echo "  make extract   - Extract content from PDFs in raw/"
	@echo "  make process   - Process all extracted papers"
	@echo "  make clean     - Clean up cache files"
