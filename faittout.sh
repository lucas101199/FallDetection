#!/bin/bash

python label.py $1 $2
mv $1 NotFAll/
rm $2
