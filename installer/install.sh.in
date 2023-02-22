#!/bin/bash

# make sure we are not already in a venv
# (don't need to check status)
deactivate >/dev/null 2>&1
scriptdir=$(dirname "$0")
cd $scriptdir

function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

MINIMUM_PYTHON_VERSION=3.9.0
MAXIMUM_PYTHON_VERSION=3.11.0
PYTHON=""
for candidate in python3.10 python3.9 python3 python ; do
    if ppath=`which $candidate`; then
        python_version=$($ppath -V | awk '{ print $2 }')
        if [ $(version $python_version) -ge $(version "$MINIMUM_PYTHON_VERSION") ]; then
	    if [ $(version $python_version) -lt $(version "$MAXIMUM_PYTHON_VERSION") ]; then
		PYTHON=$ppath
		break
	    fi
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "A suitable Python interpreter could not be found"
    echo "Please install Python 3.9 or higher before running this script. See instructions at $INSTRUCTIONS for help."
    read -p "Press any key to exit"
    exit -1
fi

exec $PYTHON ./lib/main.py ${@}
read -p "Press any key to exit"
