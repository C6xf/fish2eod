#!/usr/bin/env bash

conda_folder=$HOME/miniconda

if ! type "conda" >/dev/null; then
  if [ "$(ls -A "$conda_folder"/etc/profile.d)" ]; then
    echo "Found existing un-sourced conda"
    source "$conda_folder"/etc/profile.d/conda.sh
  else
    echo "Conda Not Found: installing from miniconda.sh"
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
      curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -L --output miniconda.sh
      # need to use 4.7.10 as -latest- has fragile timeout
    elif [[ "$OSTYPE" == "darwin"* ]]; then
      curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -L --output miniconda.sh
    else
      echo "Unknown Opperating System: Exiting"
      exit 1
    fi
    if [ -d $conda_folder ]; then # if conda folder exists and there was no conda.sh - the folder is invalid
      rm -fr $conda_folder
    fi
    bash miniconda.sh -b -p $conda_folder
    source "$conda_folder"/etc/profile.d/conda.sh
    echo 'source "$HOME/miniconda/etc/profile.d/conda.sh"' >> ~/.bashrc
    hash -r
    conda config --set always_yes yes --set changeps1 no
  fi
else
  echo "Using System Conda"
fi

conda deactivate # incase active environment
conda update -q conda
conda create python=$TRAVIS_PYTHON_VERSION --name fish2eod
conda activate fish2eod
conda env update -f conda_environment.yml
pip install .
