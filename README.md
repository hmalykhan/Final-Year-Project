# Final-Year-Project: Video Similarity Detection.
Deep Learning Based Video Similarity Detection
Developed a comprehensive model that takes a query video and retrieves the
temporal matched segment or complete video from the reference set.
Fine tune Vision Transformers on Custom Dataset.
Used Cosine Similarity method to compare the similarity.
Utilized deep learning techniques to successfully complete the project.

# File Structure
fynal_year_Project/
├── documents/
├── embeddings/
│   └── ref_embd.json  # System will store the embeddings of the reference videos frames.
├── frames/            # System will store the extracted frames of the reference videos, These frames will be used for Vision Transformers training and further processing.
├── query_frames/      # System will store the frames of the query video.
├── ref/               # Reference videos should be stored manually with the naming format R1, R2, R3, etc.
│   └── R1/
├── result_query/      # System will store the matched frames of the reference video frames with the query matched frames.
├── result_videos/     # System will compose the video from the matched frames of the reference frames with the query frames.
├── Auxiliary.py       # File containing auxiliary functions.
├── Index.py           # File containing the interface driver code.
├── model.py           # File containing Vision Transformer fine-tuning code.
└── preprocess.py      # File containing data preprocessing code to be run once before the model training and driver code.

# How to use
Make the above given directories and file structure accordingly.
Run the preprocess.py.
Run the model.py to fine tune the model on custom data.
Run the Auxilary.py.
Run the index.py the driver code and turn on the given link.
