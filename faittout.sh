#!/bin/bash

python label.py $1 $2
mv $1 Fall/
rm $2
