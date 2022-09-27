#!/bin/bash

#Ubuntu
#MIN_LINES_COV=74.9
#MIN_FUNCTION_COV=86.5

#Rhel
#MIN_LINES_COV=73.3
#MIN_FUNCTION_COV=73.6


LINES_COV=`cat genhtml/index.html | grep headerCovTableEntryLo | grep -oP  ">\K(\d*.\d*)" | head -n 1`
FUNC_COV=`cat genhtml/index.html | grep headerCovTableEntryLo | grep -oP  ">\K(\d*.\d*)" | tail -n 1`

if (( $(echo "$MIN_LINES_COV > $LINES_COV" | bc -l) )); then
    echo "Error: $LINES_COV % Lines coverage is lower than minimal $MIN_LINES_COV %"
    exit 1
fi

if (( $(echo "$MIN_FUNCTION_COV > $FUNCTION_COV" | bc -l) )); then
    echo "Error: $FUNCTION_COV % Functions coverage is lower than minimal $MIN_FUNCTION_COV %"
    exit 1
fi

echo "Coverage check success"
exit 0