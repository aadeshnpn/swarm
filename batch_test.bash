#!/bin/bash

~/anaconda2/bin/conda-env remove --name travis -y
~/anaconda2/bin/conda create --name travis python=3.5 -y

source ~/anaconda2/bin/activate travis 

python --version
pip install -r requirements.txt

pip install coveralls flake8 nose

#pip uninstall swarm -y

pip install .

flake8 . --ignore=F403,E501,E123,E128 --exclude=docs,build

nosetests --with-coverage --cover-package=swarm