install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C ./shapey/

upload:
	if [ -d "dist" ]; then rm -r dist; fi
	mkdir dist
	python setup.py sdist
	twine upload --repository pypi dist/*

all: install format lint upload