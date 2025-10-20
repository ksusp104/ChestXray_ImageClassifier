# ChestXray_ImageClassifier
Active learning system for labeling medical chest x-rays to reduce manual annotation effort
## Team Members
**CS 4850 - Section 02/03/04 - Fall 2025**\
Image of headshots
### Development
- Josh Smith
- Elijah Merrill
### Documentation
+ Noah Lane
+ Matthew Hall
### Advisor & Professor
* Sharon Perry
## Final Report
Information about final project, amount of code, hours spent, status complete, and links to final report & presentation
## Artifacts
- [Project Proposal](readmefiles/SP-104_ALS_Images-Proposal.pdf)
- [Project Plan](readmefiles/SP-104_ALS_Images-ProjectPlan.pdf)
- [System Design Document](readmefiles/SP-104_ALS_Images-Design.pdf)
- [System Requirements Specification](readmefiles/SP-104_ALS_Images-Requirements.pdf)
## Development Timeline
- Gantt Chart
- Milestones
## C-Day Files
## Resources
## Set-up and Execution (As of 10/19/25)
In the GitHub repository, there are a total of 6 files that are important to the total functionality of the system. The files that need to be downloaded onto the user system can be found in the “source code” folder in the repository. It is highly recommended to keep track of where these source code files are kept. 

The next important thing needed is the dataset itself. This data set can be found on Kaggle through this link: https://www.kaggle.com/datasets/nih-chest-xrays/data. First time users will likely need to create a Kaggle user account, and then this dataset can be downloaded through that page. The user should then be able to extract the dataset which is contained within a folder named “archive”. It is highly recommended to store this folder wherever your source code files are located. If that is not possible, keep track of that path.

Of course, any Python version of 3.1xx or later will be necessary to compile and run this system. Downloads for the latest Python version can be found here: https://www.python.org/downloads/. 

The next step involves several dependencies and packages that are necessary for the system to work. The main one is PyTorch, which can be installed by running this command in your PowerShell terminal: 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'. Any further instructions or questions can be answered here: https://pytorch.org/. Additional packages needed are: “pandas”, “numpy”, and “pyarrow”. These can be installed with this command in the PowerShell terminal: 'pip install pandas pyarrow numpy'. (Lastly, further packages that will likely be used are: “tqdm” and “matplotlib”, which can be installed with: 'pip install tqdm matplotlib'.

After all dependencies are installed, the dataset folder, archive, and source files must be verified to ensure that all file paths match the system’s local directory structure. If the dataset or output folders are stored in non-default locations, the script variables or configuration entries referencing them, such as those specifying archive/Data_Entry_2017.csv or output directories for generated files, should be manually updated before running the program.

Proper setup can be confirmed by executing the script smoke.py using the command python smoke.py. This process reads the dataset metadata and generates an indexed output file named index_table.csv, and optionally, index_table.parquet, inside the “outputs” directory. Successful creation of these files verifies that the dataset, file paths, and environment are correctly configured, and that the system is ready for further training or evaluation.

### Dependencies
### All Resources
