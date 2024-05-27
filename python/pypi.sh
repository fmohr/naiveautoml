#!/bin/bash

python setup.py sdist
twine upload -r pypi dist/*
