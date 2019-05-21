#!/bin/sh
exec gunicorn -b :9999 --access-logfile - --error-logfile - flaskAppp:app
