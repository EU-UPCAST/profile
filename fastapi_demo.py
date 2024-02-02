from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from enum import Enum
import os
import logging
import uuid
import pandas as pd
from ydata_profiling import ProfileReport

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enum to represent available apps (profilers)
class AppName(str, Enum):
    App1 = "Ydata"
    App2 = "Abstat"
    App3 = "Profiler3"


# Function to run selected app
def run_App1(file: UploadFile):
    # Display selected app name and file
    logger.info(f"Selected App: Ydata")
    logger.info(f"File: {file.filename}")

    # Generate profile using Ydata profiler
    df = pd.read_csv(file.filename, sep=';', low_memory=False, on_bad_lines='warn')
    logger.info("Profiling using Ydata profiler")
    profile = ProfileReport(df, title="Ydata Profiling Report")
    logger.info("Profile report generated")
    # Generate the filename for the output file
    json_profile = f"{file.filename.split('.')[0]}_profile.json"
    output_file_path = f"output_files/{json_profile}"
    # Create output_files directory if it doesn't exist
    os.makedirs("output_files", exist_ok=True)

    logger.info("the json profile name is: " + json_profile)
    #jsonfile = profile.to_file(json_profile)

    jsonfile = profile.to_file(output_file_path)
    output_filename = json_profile
    output_file_path = output_filename

    return output_file_path, output_filename

# Function to run App2
def run_App2(file: UploadFile):
    # Display selected app name and file
    logger.info(f"Selected App: Abstat")
    logger.info(f"File: {file.filename}")
    # Implement logic to run App1 using the uploaded file
    # For demonstration purposes, let's assume the app simply writes to a new file
    output_filename = f"App2_{str(uuid.uuid4())}.txt"
    output_file_path = f"output_files/{output_filename}"
    with open(output_file_path, "wb") as output_file:
        output_file.write(file.file.read())
    return output_file_path, output_filename

# Function to run App3
def run_App3(file: UploadFile):
    # Display selected app name and file
    logger.info(f"Selected App: Profiler3")
    logger.info(f"File: {file.filename}")

    # Implement logic to run App1 using the uploaded file
    # For demonstration purposes, let's assume the app simply writes to a new file
    output_filename = f"App3_{str(uuid.uuid4())}.txt"
    output_file_path = f"output_files/{output_filename}"
    with open(output_file_path, "wb") as output_file:
        output_file.write(file.file.read())
    return output_file_path, output_filename

# Function to run selected app
def run_app(selected_app: str, file: UploadFile):
    # Display selected app name and file
    logger.info(f"Selected App: {selected_app}")
    logger.info(f"File: {file.filename}")

    # Call the appropriate function based on the selected app name
    if selected_app == AppName.App1.value:
        return run_App1(file)
    elif selected_app == AppName.App2.value:
        return run_App2(file)
    elif selected_app == AppName.App3.value:
        return run_App3(file)
    else:
        raise ValueError("Invalid app name")



# Endpoint to get list of available apps (profilers)
@app.get("/get_profilers")
async def get_available_profilers():
    return [app_name.value for app_name in AppName]

# Endpoint to generate profile using the selected app
@app.post("/generate_profile")
async def generate_profile_with_selected_profiler(selected_profiler: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Request Payload: selected_app={selected_profiler}, file={file.filename}")

    if selected_app not in [app_name.value for app_name in AppName]:
        return JSONResponse(content={"error": "Invalid app name"}, status_code=400)
    
    output_file_path, output_filename = run_app(selected_profiler, file)
    
    # Read content of the output file
    with open(output_file_path, "r") as output_file:
        file_content = output_file.read()
    
    # Return the filename and content of the output file
    return JSONResponse(content={"filename": output_filename, "file_content": file_content})

# Generate OpenAPI schema
from fastapi.openapi.utils import get_openapi

openapi_schema = get_openapi(
    title="UPCAST Data Profiling Service",
    version="0.1.0",
    description="OpenAPI specification for the UPCAST Data Profiling Service",
    routes=app.routes,
)

# Create the api folder if it does not exist
api_folder = "openapi"
if not os.path.exists(api_folder):
    os.makedirs(api_folder)

# Save the OpenAPI schema to a JSON file under the api folder
openapi_file = os.path.join(api_folder, "openapi.json")
with open(openapi_file, "w") as file:
    import json
    json.dump(openapi_schema, file)