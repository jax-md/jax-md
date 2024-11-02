#!/usr/bin/bash
. /etc/profile && module load amber/23 quick/23.08 xtb/6.6.1 geometric/1.0.1

python_executables=("python" "python3" "python3.11" "python3.10" "python3.9" "python3.8" "python3.7" "python3.6"  )

PYTHON_CMD=""

for py_cmd in "${python_executables[@]}"; do
    if command -v $py_cmd &> /dev/null; then
        echo "$py_cmd is installed."
        PYTHON_CMD=$py_cmd
        version_string=`$PYTHON_CMD --version | cut -d' ' -f2`
        major=`echo "$version_string" | cut -d'.' -f1`
        minor=`echo "$version_string" | cut -d'.' -f2`
        if [ "$major" -gt 2 ] && [ "$minor" -gt 5 ]; then break; fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Python is not installed."
    exit 1
else
    rm -f scipyopt.pid run_task.pid
    $PYTHON_CMD scipyopt.py > scipyopt.log 2>&1 &
    echo $! > scipyopt.pid
fi

module unload amber/23 quick/23.08 xtb/6.6.1 geometric/1.0.1