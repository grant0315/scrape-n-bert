import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor

class MainSpider(scrapy.Spider):
    name = "main"

    def __init__(self, url, css_selector):
        self.allowed_domains = [determine_domain(url)]
        self.start_urls = ["https://" + url]
        self.css_selector = css_selector

        # Print for logging information
        print(f"USING CSS SELECTOR: {css_selector}")
        print(f"SCRAPING DOMAIN: {url}")
    
    def parse(self, response):
        whole_page_content = ""
        
        for content in response.css(self.css_selector + "::text"):
            whole_page_content += str(content.get()) + " "
        
        if whole_page_content != "":
            yield {
                'content': whole_page_content,
                'url': response.url, 
            }

        le = LinkExtractor()
        for link in le.extract_links(response):
            if does_link_contain_base_url(str(link), self.start_urls[0]) == True:   
                yield scrapy.Request(link.url, callback=self.parse)

def determine_domain(url):
    domain = ""
    
    for i in url:
        if i == "/":
            break

        domain += i

    return domain

def does_link_contain_base_url(in_link, base_url):
    if base_url in in_link:
        return True
    else:
        return False

print(determine_domain("www.trustradius.com/buyer-blog"))
