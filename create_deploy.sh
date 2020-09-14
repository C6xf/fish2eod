#!/bin/sh

# only run for one instance py3.6 on linux when it's not a PR
if [[ "$TRAVIS_PYTHON_VERSION" == "3.6" && "$TRAVIS_OS_NAME" = "linux" && "$TRAVIS_PULL_REQUEST" == "false" ]]; then
  echo "Uploading to Codecov"
  bash <(curl -s https://codecov.io/bash)

  echo "Building Docs"
  make -C docs buildapi
  make -C docs html

  echo "Deploying docs to Github"
  bash push.sh

  exit 0
else
  echo "Note uploading for this config"
  exit 0
fi