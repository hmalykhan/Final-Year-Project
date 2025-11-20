# Final-Year-Project: Video Similarity Detection.
Deep Learning Based Video Similarity Detection
Developed a comprehensive model that takes a query video and retrieves the
temporal matched segment or complete video from the reference set.
Fine tune Vision Transformers on Custom Dataset.
Used Cosine Similarity method to compare the similarity.
Utilized deep learning techniques to successfully complete the project.

# File Structure
fynal_year_Project/
├── documents/         # System related documents are stored.</br>
├── trianed_model/     # System will create this directory after training the model.</br>
├── results/           # System will save the results of the trained model.</br>
├── other_videos/      # File containing auxiliary functions.</br>
├── flagged/           # Flagged stamps will be saved. </br> 
├── embeddings/
│   └── ref_embd.json  # System will store the embeddings of the reference videos frames.</br>
├── frames/            # System will store the extracted frames of the reference videos, These frames will be used for Vision Transformers training and further processing.</br>
├── query_frames/      # System will store the frames of the query video.</br>
├── ref/               # Reference videos should be stored manually with the naming format R1, R2, R3, etc.</br>
│   └── R1/
├── result_query/      # System will store the matched frames of the reference video frames with the query matched frames.</br>
├── result_videos/     # System will compose the video from the matched frames of the reference frames with the query frames.</br>
├── Auxiliary.py       # File containing auxiliary functions.</br>
├── Index.py           # File containing the interface driver code.</br>
├── model.py           # File containing Vision Transformer fine-tuning code.</br>
└── preprocess.py      # File containing data preprocessing code to be run once before the model training and driver code.</br>

# How to use</br>
Make the above given directories and file structure accordingly.</br>
Run the preprocess.py.</br>
Run the model.py to fine tune the model on custom data.</br>
Run the Auxilary.py.</br>
Run the index.py the driver code and turn on the given link.</br>
