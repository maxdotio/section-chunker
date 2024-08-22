# Section Chunker
Converts an Azure DocumentAI contract into ready to use sections for search and embeddings. The whole process involves two major tasks; Section Identification and Section Merging. 
## Step 1: Section Identification
Section Identification is done using an ML pipeline which predicts the lines that are potential section headers and merges all the lines that aren't section headers until another section header is detected. The pipeline uses a pre-trained Random Forest Model, which was trained on several documents, to perform the task.
## Step 2: Section Merging
Section Merging is done as a post-processing step for the section identification. This pipeline takes the output sections from the section identification service and tries to merge them, whenever necessary.

This algorithm was developed using a robust LLM pipeline which takes the content from any two sections as input and tells whether they need to be merged or not. To be more accurate, the LLM pipeline was trained by optimizing the prompts using Python's DSPy module.
# Prerequisites
You need to have docker installed on your machine. If not, you can download it from [here](https://www.docker.com/products/docker-desktop). 

For section merging, the pipeline uses Azure OpenAI models, hence you should have an appropriate API key and endpoint, if you want to use that service.

# Installation
Clone the repository into your local machine.  

Open the app folder and create a .env file in it. In the file, you need to assign the API key and endpoint to the variables AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT, respectively.

This is how a sample .env file should look like

```ini
AZURE_OPENAI_API_KEY = [Your API key]
AZURE_OPENAI_ENDPOINT = [Your Endpoint]
```
# How to use 
Open up a powershell tab and type the command below to create a docker container for running the service. 

```bash
docker compose -f .\docker-compose.yml up
```

If you get an error doing this, open docker desktop and try again. 

Now, you can access the service at [https://localhost:8080](https://localhost:8080).

You can use this API service in your code to obtain sections from an Azure DocumentAI contract.

In the request body, you need to pass the contract as "json_data" and assign true to "merge" field if you want to perform the section merging as well or else false to just return the sections without any post-processing.  