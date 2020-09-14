#!/usr/bin/env bash

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

if ! type "gmsh" >/dev/null; then
  echo "gmsh not found installing"

  sudo apt-get update
  sudo apt-get install gmsh
else
  gmsh_version="$(gmsh -version 2>&1)"
  min_version=3.0.0

  if version_gt "$gmsh_version" $min_version; then
    echo "Using system gmsh"
  else
    echo "gmsh version<$min_version" please update
    exit 1
  fi
fi

bash -i install_conda.sh
