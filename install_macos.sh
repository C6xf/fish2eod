#!/usr/bin/env bash

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }
gmsh_path="/Applications/Gmsh.app/Contents/MacOS/gmsh"

if ! type $gmsh_path >/dev/null; then
  echo "gmsh not found installing"

  wget http://gmsh.info/bin/MacOSX/gmsh-3.0.6-MacOSX.dmg -O gmsh.dmg
  sudo hdiutil attach gmsh.dmg
  sudo cp -R /Volumes/gmsh-3.0.6-MacOSX/Gmsh.app /Applications
  sudo hdiutil unmount /Volumes/gmsh-3.0.6-MacOSX

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