install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C ./shapey/

upload:
	[ -d "dist/" ] && rm -r dist/*
	python setup.py sdist
	twine upload --repository pypi dist/*

all: install format lint upload