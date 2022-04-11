import configparser
import shutil
import bertopic_wrapper.main as bert
import os
import glob
import json
import sys


class EntryPoint:
    """ 
    This class is the main entry point for running scrape-n-bert in its various options.
    """
    def __init__(self, config_path=""):
        # Check if config_path exists
        if config_path == "":
            print("[LOG]: Running without config file")
        
        elif self.__check_if_file_exists(config_path) != True:
            print("[LOG]: Stopping scrape-n-bert")
            return -1
        
        else:
            self.config = configparser.ConfigParser()
            self.config.read(config_path)

    
    # Currently working
    def scrape_and_run_bertopic_per_domain(self):
        """
        This function uses scrapy to scrape desired data from a .ini (must be formatted correctly),
        and then feeds it into a bertopic instance to generate data for each individual domain provided.
        -----------------------------------------------------------------
        """
        # Get general settings from config file
        output_file_directory = self.config['General Settings']['OUTPUT_FILE_DIRECTORY']

        # Run scrapey spider over domains listed in config file
        self.__config_scrape_loop(output_file_directory)

        # Run bertopic over .jl files created
        self.__bert_training_loop(output_file_directory)

    # --- WIP ---
    def scrape_only(self):
        """
        This function uses scrapy to scrape desired domains from a .ini (must be formatted correctly),
        and DOES NOT run the bertopic model on top.
        -----------------------------------------------------------------
        Args: 
            path_to_config_file: path to config file (must be formatted correctly)
            output_directory: path to output directory for scraped data from spider.
        
        Raises:
            placeHolder
        """

        # Get general settings from config file
        output_file_directory = self.config['General Settings']['OUTPUT_FILE_DIRECTORY']

        # Run scrapey spider over domains listed in config file
        self.__config_scrape_loop(output_file_directory)

    # Currently working
    def bertopic_only(self, input_file_path, output_directory, search_term):
        """
        This function takes a desired directory of .jl files created from scrapy spider and,
        feed them into a bertopic instance, and returns bertopic topic files
        -----------------------------------------------------------------
        Args:
            input_file_path: path to .jl file to run bertopic instance on.
            output_directory: The directory you want bertopic files to be written to.

        Raises:
            placeHolder 
        """
        # Check input_file_path is real
        try:
            self.__check_if_file_exists(input_file_path)
        except Exception as e:
            print("[ERROR]: " + e)

        # Check output_directory is real
        try:
            self.__check_if_directory_exists(output_directory)
        except Exception as e:
            print("[ERROR]: " + e)

        scraped_data_name = os.path.basename(input_file_path).replace(".jl", "")
        domain_folder_path = self.__create_folder_for_domain(output_directory, scraped_data_name)
        print(domain_folder_path)

        bt = bert.BertopicTraining(input_file_path, domain_folder_path, "bertopic_only", search_term)
        bt.trainModel()
    
    # --- WIP ----
    def compile_scrape_data_and_run_bertopic(self, input_data_directory, output_directory, output_filename):
        """
        This function compiles several .jl files from the input_data_path and then 
        feeds it into a bertopic instance, and returns bertopic topic files
        -----------------------------------------------------------------
        Args: 
            input_data_path: Path to folder containing .jl files to compile
            output_directory: the directory you want bertopic files to be written to

        Raises: 
            placeHolder
        """
        temp_result = []
        jl_path_list = self.__get_all_jl_files_in_directory(input_data_directory)
        print(jl_path_list)

        # Read each file in return from __get_all_jl_files_in_directory
        # read each line in indexed file, and try appending them to temp_result
        for file in jl_path_list:
            with open(file, 'r', encoding='utf-8-sig') as infile:
                for line in infile.readlines():
                    try:
                        temp_result.append(json.loads(line) + "\n")
                    except ValueError:
                        print(file)
        
        # Write temp_result to combined .jl file at output_directory
        combined_file_path = output_directory + "/" + output_filename + '_merged_file.jl'
        print(combined_file_path)
        with open(combined_file_path, 'w', encoding='utf-8-sig') as outfile:
            outfile.write("\n".join(map(json.dumps, temp_result)) + "\n")
            outfile.close()
            print("Combining file")

        bt = bert.BertopicTraining(combined_file_path, output_directory, "_bertopic_only")
        bt.trainModel()

    def __config_scrape_loop(self, output_file_directory):
        """
        This function scrapes every domain listed in a .ini file, 
        with the configuration provided.
        -----------------------------------------------------------------
        Args: 
            output_file_directory: the root directory that the data will be written to.

        Raises: 
            placeHolder
        """
        sections = self.config.sections()

        # For each domain, run the run.sh file in order to scrape.
        for section in sections:
            if section != "General Settings":
                print(str(section))
                section_file_name = self.__create_scrapy_content_file_name(str(section)) # Generate output filename
                section_folder_name = self.__create_domain_folder_name(str(section))
                scraped_data_folder_path = output_file_directory + "/" + section_folder_name + "/" + section_file_name

                print(section_file_name)
                print(section_folder_name)
                print(scraped_data_folder_path)
                
                # Create domain name folder and scraped data sub-folder for specified domain
                self.__create_folder_for_domain(output_file_directory, section_folder_name)

                self.__run_scrape_shell_command(section_file_name, str(section), self.config[section]["CSS_SELECTORS"], self.config[section]["DEPTH_LIMIT"], self.config[section]["CLOSESPIDER_PAGECOUNT"])
                
                # Get absolute path to created scrape data file and path to output_directory
                data_file_abs_path = os.path.abspath("recursive_spider/" + section_file_name)

                # Move scraped data to scraped_data in output directory
                self.__move_file_to_folder(data_file_abs_path, scraped_data_folder_path)
                # os.replace("./recursive_spider/recursive_spider/" + section_file_name, section_folder_path)

    def __bert_training_loop(self, output_file_directory):
        """
        This function runs a bertopic instance from a .ini (must be formatted correctly),
        and then returns bertopic topic data.
        -----------------------------------------------------------------
        Args:
            output_file_directory: The root directory that the data will be written to.
        """

        # Pull data from config file
        search_term_from_config = self.config['General Settings']['BERT_SEARCH_TERM']
        sections = self.config.sections()

        # For each domain, run the run.sh file in order to scrape.
        for section in sections:
            if section != "General Settings":
                # Create ml_data and visualization folder
                formatted_folder_name = self.__create_domain_folder_name(section)
                self.__create_visualization_folder(output_file_directory + formatted_folder_name) # Create visualization folder
                self.__create_ml_data_folder(output_file_directory + formatted_folder_name) # Create ML data folder

                # Determine path for the scraped data file
                in_file_name = self.__create_scrapy_content_file_name(section)
                in_file_path = output_file_directory + "/" + formatted_folder_name + "/" + in_file_name

                out_file_name = "individual_domain"
                out_directory_path = output_file_directory + "/" + formatted_folder_name

                bt = bert.BertopicTraining(in_file_path, out_directory_path, out_file_name, search_term_from_config)
                bt.trainModel()

    def __run_scrape_shell_command(self, output_file_name, domain, css_selector, depth_limit, close_spider_page_count):
        """
        This is a function that runs a shell file with the provided parameters as CLI args.
        ----------------------------------------------------------------
        Args: 
            output_file_name: the output file name without an extension
            domain: the base domain (without the https://) to scrape from
            css_selector: the css selector used to pull content from
            depth_limit: the depth limit, or the number of subdomains in the url to scrape before blocking past that point
            close_spider_page_count: the total amount of pages scraped before closing the spider
        
        Raises:
            placeHolder
        """
        shell_command = "sh ../shell/run_spider.sh " + "-o " + output_file_name + " " + "-d " + domain + " " + "-c " +  css_selector + " " + "-l " +  depth_limit + " " + "-p " + close_spider_page_count
        os.system(shell_command)

        return shell_command

    def __check_if_file_exists(self, path):
        if os.path.isfile(path):
            return True
        else:
            raise ValueError("[ERROR]: Could not find given file -> " + path)

    def __check_if_directory_exists(self, path):
        if os.path.isdir(path):
            return True
        else:
            raise ValueError("[ERROR]: Could not find given directory -> " + path)

    def __get_all_jl_files_in_directory(self, directory):
        """
        This is a function that returns a list of all .jl files in the given directory
        ----------------------------------------------------------------
        Args: 
            directory: Full path to folder that contains the .jl files

        Raises:
            ValueError: Could not find given directory
        """
        if os.path.isdir(directory):
            target_pattern = directory + "/" + "*.jl"
            return glob.glob(target_pattern)
        else:
            raise ValueError("Could not find given directory")

    def __create_folder_for_domain(self, root_folder_path, root_folder_name):
        """
        Creates the directory structure for storing generated data
        ----------------------------------------------------------------
        Args:
            root_folder_path: path to destination to store new new data
            root_folder_name: name of the folder that will be created in root_folder_path

        Raises:
            FileExistsError: Folder already exists

        Returns: 
            String: full path to domain folder
        """

        try:
            full_path = root_folder_path + "/" + root_folder_name
            os.mkdir(full_path)
            return full_path
        except FileExistsError as e:
            print("!=== Root folder already exists ===!") 
            pass

    def __create_ml_data_folder(self, directory):
        """
        Creates a ml_data folder in a desired directory
        ----------------------------------------------------------------
        Args:
            directory: Path to the directory to create new folder in.

        Returns:
            String: full path to ml_data folder
        """
        try:
            dir_path = os.mkdir(directory + "/ml_data")
            return dir_path
        except FileExistsError as e:   
            print("!=== Domain ml_data folder already exists ===!")
            pass

    def __create_visualization_folder(self, directory):
        """
        Creates a visualization folder in a desired directory
        ----------------------------------------------------------------
        Args:
            directory: Path to the directory to create new folder in.

        Returns:
            String: full path to visualization folder
        """
        try:
            dir_path = os.mkdir("/" + directory + "/visualizations")
            return dir_path
        except FileExistsError as e:
            print("!=== Domain visualization folder already exists ===!")
            pass

    def __create_scrapy_content_file_name(self, domain):
        """
        Formats provided domain into a file name that works in directory structure 
        ----------------------------------------------------------------
        Args:
            domain: URL domain
        
        Raise: 
            placeHolder

        Returns: 
            string: file name with '.jl' extension
        """
        file_name = domain.replace('.', '_').replace('/', '_')
        file_name += '.jl'
        return str(file_name)

    def __create_domain_folder_name(self, domain):
        """
        Formats provided domain into a string that works in directory structure
        (This function should be used for referencing other files)
        ----------------------------------------------------------------
        Args:
            domain: URL domain
        
        Raise: 
            placeHolder

        Returns: 
            string: a folder/file name without a .jl extension for naming and finding purposes.
        """
        folder_name = domain.replace('.', '_').replace('/', '_')
        return str(folder_name)

    def __move_file_to_folder(self, in_file_path, out_file_path):
        """
        Moves one file to a specified folder.
        ----------------------------------------------------------------
        Args: 
            in_file_path: Path to target file
            dest_folder_path: Path to desired destination folder.
        
        Raise:

        """
        try:
            shutil.move(in_file_path, out_file_path)
        except ValueError as e:
            print("[ERROR]: " + str(e))

# Detect the arg passed from the main shell script, and run related EntryPoint function
if __name__ == "__main__":
    cli_arg = sys.argv[1]

    
    if cli_arg == "individual-snb":
        config_path = input("Full path to config: ")
        EntryPoint = EntryPoint(config_path)
        EntryPoint.scrape_and_run_bertopic_per_domain()
    
    elif cli_arg == "only-scrape":
        config_path = input("Full path to config: ")
        EntryPoint = EntryPoint(config_path)
        EntryPoint.scrape_only(config_path)
    
    elif cli_arg == "only-bert":
        EntryPoint = EntryPoint()
        scraped_data_path = input(".jl file full path: ")
        output_file_directory = input("Output folder full path: ")
        search_term = input("Specify search term: ")
        EntryPoint.bertopic_only(scraped_data_path, output_file_directory, search_term)
    
    elif cli_arg == "combined-bert":
        EntryPoint = EntryPoint()
        combined_folder_path = input("Combined folder full path: ")
        output_file_directory = input("Output folder full path: ")
        EntryPoint.compile_scrape_data_and_run_bertopic(combined_folder_path, output_file_directory, "test")
