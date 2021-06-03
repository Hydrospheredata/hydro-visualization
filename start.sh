#!/usr/bin/env sh

if [ -z $1 ]; then
    echo "Choose mode: 'service' or 'worker'"
    exit 1
fi

if test $1 == 'service'; then
    python app.py
elif test $1 == 'worker'; then
    celery -A app.utils.conf.celery worker -l info -Q visualization
else
    echo "'$1' mode is incorrect. Supported modes are 'service' or 'worker'"
    exit 1
fi