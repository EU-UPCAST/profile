import streamlit as st
import requests

def main():
    st.title("UPCAST Data Profiling Demo")

    # Get list of available apps
    response = requests.get("http://localhost:8000/profile/get_profilers")
    available_apps = response.json()

    # Display dropdown menu for selecting profiler
    selected_app = st.selectbox("Select Profiler", available_apps)

    # Upload file
    uploaded_file = st.file_uploader("Upload File")

    # Submit button
    if st.button("Generate profile"):
        # Check if a file was uploaded
        if uploaded_file is not None:
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
        else:
            st.error("Please upload a file before submitting.")

if __name__ == "__main__":
    main()
