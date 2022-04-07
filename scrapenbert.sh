#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -t type_of_snb_run"
    echo "Types (the types of snb runs):"
    echo "  Optional -t individual-snb Runs scrape-n-bert from config file, and run bert instance on each domain"
    echo "  Optional -t only-scrape Runs spider on domains in config file"
    echo "  Optional -t only-bert Runs bertopic instance on each domain in config file without scraping"
    echo "  Optional -t combined-bert Combines multiple scraped data files, and runs bertopic on top of the large file"
    echo
}

while getopts "t:" opt
do
    case "$opt" in
        t ) TYPE="$OPTARG" ;;
        h ) helpFunction ;;
        ? ) helpFunction ;;
        * ) helpFunction ;;
    esac
done

case "$TYPE" in
    "individual-snb") 
        echo "Running Scrape-n-bert in type: $TYPE\n"
        cd src/py
        python3 entry.py individual-snb
        ;;

    "only-scrape")
        echo "Running Scrape-n-bert in type: $TYPE\n"
        cd src/py
        python3 entry.py only-scrape 
        ;;

    "only-bert")
        echo "Running Scrape-n-bert in type: $TYPE\n"
        ;;

    "combined-bert")
        echo "Running Scrape-n-bert in type: $TYPE\n"
        ;;

    *)
        helpFunction ;;
esac