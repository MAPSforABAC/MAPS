# LAPS
LAPS ( LLM-based ABAC Policy Synthesis from Unstructured Text ) is a research prototype that uses Large Language Models (LLMs) to automatically transform unstructured natural language access control documents (PDF/DOCX/TXT) into structured, actionable security policies (ASPs). The system supports policy extraction, dataset generation, and policy visualization, and is evaluated using fuzzy matrix analysis and BERTScore for accuracy and semantic alignment.

To run LAPS locally, please follow these steps-

1) Make sure that you are using Python version > 3.12.5 and pip version > 24.0
2) Please download the whole LAPS or clone it using - git clone HTTPS link
3) Please add LLM API key/keys in app10.py file, and there is no need to add API key for every LLM; you may use any one of the models
4) Create a virtual environment in the directory where the project files are present ( Use "cd path/to/your/project" ) and activate it
   To activate the virtual environment, please use these commands -
   For MacOS/Linux -
     python3 -m venv venv
     source venv/bin/activate
   For Windows -
     python -m venv venv
     venv\Scripts\activate
5) Please ensure that the virtual environment is active ( the name 'venv' will appear at the start of the command line )
6) In the virtual environment, please install the required modules by using "pip install -r requirements.txt"  ( requirements.txt is already provided with the project )
7) Now, you may run the project using "python app.py" ( Go to the localhost link on any browser and use LAPS )
   
   
