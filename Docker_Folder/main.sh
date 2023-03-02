#!/bin/bash 

cd /home/ubuntu/Docker_Folder

uvicorn main:app --host 0.0.0.0 --port 8000
