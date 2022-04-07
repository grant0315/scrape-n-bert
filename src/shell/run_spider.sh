#!/bin/bash
# EXAMPLE OF USAGE: sh run.sh -d "learn.g2.com" -c "#hs_cos_wrapper_post_body > p" -l 6 -p 2000

# CURRENT ORDER OF ARGS 
# -- DOMAIN - CSS_SELECTOR - DEPTH_LIMIT - CLOSERSPIDER_PAGECOUNT

# EXAMPLE OF SCRAPY COMMAND = scrapy crawl main -a domain="learn.g2.com" -a css_selector="#hs_cos_wrapper_post_body > p" -s DEPTH_LIMIT=6 -s CLOSESPIDER_PAGECOUNT=200

# Get all CLI inputs from user
helpFunction() 
{
    echo ""
    echo "Usage: $0 -o output_file_name -d domain -c css_selector -l depth_limit -p closespider_pagecount"
    echo -e "\t-o the output file name without an extension"
    echo -e "\t-d the base domain (without the https://) to scrape from"
    echo -e "\t-c the css selector used to pull content from"
    echo -e "\t-l the depth limit, or the number of subdomains in the url to scrape before blocking past that point"
    echo -e "\t-p the total amount of pages scraped before closing the spider"
}

createFileName()
{
    echo ${domain}
}

while getopts "o:d:c:l:p:h" opt
do 
    case "$opt" in
        o ) output_file_name="$OPTARG" ;;
        d ) domain="$OPTARG" ;;
        c ) css_selector="$OPTARG" ;;
        l ) depth_limit="$OPTARG" ;;
        p ) closespider_pagecount="$OPTARG" ;;
        h ) helpFunction ;;
        ? ) helpFunction ;; # Print the help function
    esac
done

# Print helpFunction in case parameters are empty
if [ -z "$output_file_name" ] || [ -z "$domain" ] || [ -z "$css_selector" ] || [ -z "$depth_limit" ] || [ -z "$closespider_pagecount" ]
then 
    echo "Some or all of the parameters are empty";
    helpFunction
else
    # Print entered options
    echo "$output_file_name"
    echo "$domain"
    echo "$css_selector"
    echo "$depth_limit"
    echo "$closespider_pagecount"

    # Run spider
    echo "!==== RUNNING SPIDER ====!"
    cd ../py/recursive_spider # go to spider directory
    # run spider
    scrapy crawl main -O $output_file_name -a url="$domain" -a css_selector="$css_selector" -s DEPTH_LIMIT=$depth_limit -s CLOSESPIDER_PAGECOUNT=$closespider_pagecount 
    cd ../ # return to root directory

    # Then move generated .jl file into scraped_data in output_file_directory


fi

