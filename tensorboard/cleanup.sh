#!/bin/bash

while IFS= read -r fname; do
    if [ -f "$fname" ]; then
	rm "$fname"
	echo "Removed \"$fname\""
    fi
done <<EOF
$(find . -type f -name \*tfevents\* -size -100c)
EOF

find . -type d -empty -delete

