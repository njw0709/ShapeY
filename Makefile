install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C hello.py

upload:
	rm dist/*
	python setup.py sdist
	twine upload --repository pypi dist/*

all: install format lint upload