import streamlit as st
import requests
import io

def main():
    st.title("UPCAST Data Profiling Demo")

    # Get list of available apps
    response = requests.get("http://localhost:8000/profile/get_profilers")
    available_apps = response.json()

    # Display dropdown menu for selecting profiler
    selected_app = st.selectbox("Select Profiler", available_apps)

    # Option 1: User upload a local file
    uploaded_file = st.file_uploader("Upload a local file")

    # Option 2: User enters a URL
    file_url = st.text_input("Or enter a file URL")

    # Process the file
    file_content = None
    filename = None

    # Submit button
    if st.button("Generate profile"):
        if uploaded_file:
            # Send request to FastAPI to run selected app with uploaded file
            files = {"file": uploaded_file}
            data = {"selected_app": selected_app}
            response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)

            # Parse JSON response
            response_data = response.json()
            filename = response_data["filename"]
            file_content = response_data["file_content"]

            # Display output filename
            st.markdown("### Output Filename")
            st.write(filename)

            # Display output file content
            st.markdown("### Output File Content")
            st.code(file_content, language="text")

        elif file_url:
            # Send the URL to FastAPI
            #files = {"file_url": file_url}
            data = {"selected_app": selected_app, "file_url": file_url}
            response = requests.post("http://localhost:8000/profile/generate_profile", data=data)

            # Parse JSON response
            response_data = response.json()
            filename = response_data["filename"]
            file_content = response_data["file_content"]

            # Display output filename
            st.markdown("### Output Filename")
            st.write(filename)

            # Display output file content
            st.markdown("### Output File Content")
            st.code(file_content, language="text")

        else:
            st.error("Please upload a file or enter a URL.")
            response = None

        # Show response from FastAPI
        if response and response.status_code == 200:
            st.success(response.json()["message"])
        elif response:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

        # Check if a file was uploaded
        #if uploaded_file is not None:
        #    # Send request to FastAPI to run selected app with uploaded file
        #    # Read the uploaded file
        #    file_content = uploaded_file.read()
        #    filename = uploaded_file.name
        #    st.success(f"Uploaded: {filename}")
        #    files = {"file": uploaded_file}
        #    data = {"selected_app": selected_app}
        #    response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)

        #elif file_url:
        #    try:
        #        # Fetch file from the URL
        #       response = requests.get(file_url)
        #        response.raise_for_status()  # Check for errors
        #        file_content = response.content  # Read file content
        #        filename = file_url.split("/")[-1] or "downloaded_file"
        #        st.success(f"Fetched file: {filename}")
        #        # Convert content to a file-like object
        #        file_obj = io.BytesIO(response.content)

        #        # Return as UploadFile
        #        uploaded_file = UploadFile(filename=filename, file=file_obj)
        #        files = {"file": uploaded_file}
        #        data = {"selected_app": selected_app}
        #        response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)
        #    except requests.RequestException as e:
        #        st.error(f"Error fetching file: {e}")

        #else:
        #    st.error("Please upload a file or give a url before submitting.")

        #files = {"file": uploaded_file}
        #data = {"selected_app": selected_app}
        #response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)

        # Parse JSON response
        #response_data = response.json()
        #filename = response_data["filename"]
        #file_content = response_data["file_content"]

        # Display output filename
        #st.markdown("### Output Filename")
        #st.write(filename)

        # Display output file content
        #st.markdown("### Output File Content")
        #st.code(file_content, language="text")

if __name__ == "__main__":
    main()
