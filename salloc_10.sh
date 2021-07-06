#!/bin/bash

echo "sallocing node for 10 minutes."
salloc -C gpu -q interactive -t 5 -c 10 -G 1 -A mp309
