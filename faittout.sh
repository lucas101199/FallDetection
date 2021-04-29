#!/bin/bash

python label.py $1 $2
mv $1 NotFall/
rm $2
