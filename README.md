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
## Set-up and Execution (As of 10/26/25)
In the GitHub repository, there are a total of six files that are essential to the overall functionality of the system. The files that need to be downloaded to the user’s system can be found in the “source code” folder within the repository. It is highly recommended to keep track of where these source code files are stored for easier configuration and troubleshooting.

The next important component is the dataset itself, which can be found on Kaggle at the following link: https://www.kaggle.com/datasets/nih-chest-xrays/data. First-time users will likely need to create a Kaggle account to access and download the dataset. Once downloaded, the dataset will be contained in a folder named “archive.” It is recommended to store this folder in the same directory as your source code files. If that is not possible, make sure to record the exact file path for later reference.

A Python version of 3.10 or later is required to compile and run this system. The latest Python version can be downloaded from https://www.python.org/downloads/. Several dependencies and packages must also be installed for the system to function properly. The primary one is PyTorch, which can be installed by running the command `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` in the PowerShell terminal. Further instructions or troubleshooting information can be found on the official PyTorch website at https://pytorch.org/. 

Additional required packages include pandas, numpy, and pyarrow, which can be installed using the command “pip install pandas pyarrow numpy.” Possible packages that will be implemented include tqdm and matplotlib, which can be installed using `pip install tqdm matplotlib`.

After all dependencies are installed, ensure that the dataset folder (archive) and source files are properly configured and that all file paths match your local directory structure. If the dataset or output folders are stored in non-default locations, update the script variables or configuration entries that reference them, such as those specifying archive/Data_Entry_2017.csv or the output directories for generated files. 

To verify that everything is set up correctly, execute the script “smoke.py” by running “python smoke.py” in the terminal. This script reads the dataset metadata and generates an indexed output file named index_table.csv, and optionally index_table.parquet, inside the “outputs” directory. Successful creation of these files confirms that the dataset, file paths, and environment are properly configured, and that the system is ready for further training or evaluation.
