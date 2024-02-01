#!/bin/bash

# Path to the input and output files
input_file="output.txt"
output_file="parsed_results.txt"

# AWK script to parse the output file and extract required information
awk '
BEGIN {
    print "Executable,Grid Size,MPI Ranks,Execution Time" > "'$output_file'"
}
/^jacobi/ {
    if (record != "") print record > "'$output_file'"
    split($0, parts, ",")
    executable = parts[1]
    gridSize = parts[2]
    mpiRanks = parts[3]
    isSerial = (executable == "jacobi-serial-fortran")
    record = executable "," gridSize "," mpiRanks ","
    next
}
isSerial && /real/ {
    match($0, /real[ \t]*[0-9]+m([0-9.]+)s/, timeMatch)
    record = record timeMatch[1]
}
!isSerial && /runtime=/ {
    match($0, /runtime=[ \t]*([0-9.]+)/, timeMatch)
    record = record timeMatch[1]
}
END {
    if (record != "") print record > "'$output_file'"
}
' "$input_file"

echo "Parsed results saved to $output_file"

