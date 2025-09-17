hftags_string = """Multimodal
    Audio-Text-to-Text
    Image-Text-to-Text
    Visual Question Answering
    Document Question Answering
    Video-Text-to-Text
    Visual Document Retrieval
    Any-to-Any
Computer Vision
    Depth Estimation
    Image Classification
    Object Detection
    Image Segmentation
    Text-to-Image
    Image-to-Text
    Image-to-Image
    Image-to-Video
    Unconditional Image Generation
    Video Classification
    Text-to-Video
    Zero-Shot Image Classification
    Mask Generation
    Zero-Shot Object Detection
    Text-to-3D
    Image-to-3D
    Image Feature Extraction
    Keypoint Detection
    Video-to-Video
Natural Language Processing
    Text Classification
    Token Classification
    Table Question Answering
    Question Answering
    Zero-Shot Classification
    Translation
    Summarization
    Feature Extraction
    Text Generation
    Fill-Mask
    Sentence Similarity
    Text Ranking
Audio
    Text-to-Speech
    Text-to-Audio
    Automatic Speech Recognition
    Audio-to-Audio
    Audio Classification
Voice Activity Detection
Tabular
    Tabular Classification
    Tabular Regression
    Time Series Forecasting
Reinforcement Learning
    Reinforcement Learning
    Robotics
Other
    Graph Machine Learning"""

current_broader = None
hftags_list = []
for string in hftags_string.split("\n"):
    if string.startswith("    "):
        hftags_list.append("(root)/"+current_broader+"/"+string[4:])
    else:
        hftags_list.append("(root)/"+string)
        current_broader=string


if __name__ == "__main__":
    print(hftags_list)
