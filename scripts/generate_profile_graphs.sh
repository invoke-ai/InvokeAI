#!/bin/bash

# Accepts a path to a directory containing .prof files and generates a graphs
# for each of them. The default output format is pdf, but can be changed by
# providing a second argument.

# Usage: ./generate_profile_graphs.sh <path_to_profiles> <type>
# <path_to_profiles> is the path to the directory containing the .prof files
# <type> is the type of graph to generate. Defaults to 'pdf' if not provided.
# Valid types are: 'svg', 'png' and 'pdf'.

# Requires:
# - graphviz: https://graphviz.org/download/
# - gprof2dot: https://github.com/jrfonseca/gprof2dot

if [ -z "$1" ]; then
  echo "Missing path to profiles directory"
  exit 1
fi

type=${2:-pdf}

for file in $1/*.prof; do
  base_name=$(basename "$file" .prof)
  gprof2dot -f pstats "$file" | dot -T$type -Glabel="Session ID ${base_name}" -Glabelloc="t" -o "$1/$base_name.$type"
  echo "Generated $1/$base_name.$type"
done
