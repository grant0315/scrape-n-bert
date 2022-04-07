#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -c config file -o output directory (an existing folder)"
}

while getopts "c:o:h" opt
do
    case "$opt" in
        c ) config_file="$OPTARG" ;;
        o ) output_file_directory="$OPTARG" ;;
        h ) helpFunction ;;
        ? ) helpfunction ;;
    esac
done

# Print helpFunction if parameters is missing
if [ -z "$config_file" ] || [ -z "$output_file_directory" ]
then
    echo "Some or all parameters are empty";
    helpFunction
else # Actual meat of the bash file
    # Print config file path
    echo "$config_file"
    echo "$output_file_directory"

    # Create scraped content subfolder
    mkdir $output_file_directory/scraped_data

    # Run spider from config file
    python3 config_parser.py $config_file $output_file_directory
fi

