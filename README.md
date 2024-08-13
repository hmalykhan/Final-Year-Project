# Final-Year-Project
Deep Learning Based Video Similarity Detection
Developed a comprehensive model that takes a query video and retrieves the
temporal matched segment or complete video from the reference set.
Fine tune Vision Transformers on Custom Dataset.
Used Cosine Similarity method to compare the similarity.
Utilized deep learning techniques to successfully complete the project.
# File Structure
fynal_year_Project
|__embeddings
|            |__ref_embd.json #Sytem will store the embeddings of the reference videos frames. 
|__frames #System will store the extracted frames of the reference videos and used them for vision transformers training and further process.
|__query #
|__query_frames #System will store the frames of query video
|__ref #Manually the reference videos shold be stored with the naming format of R1, R2, R3.....
|     |__R1
|__result_query #System will store the matched frames of reference video frames with the query mathced frames. 
|__result_videos #System will compose the video from the matched frames of the reference frames with the query frames.
|__Auxilary.py #File containing the functions.
|__Index.py #File containing the interface driver code.
|__model.py #File containing Vision Trasformer finetunning code.
|__preprocess.py #File containing the data preprocessing code should be run once before the model training and driver code.
# How to use
Make the above given directories and file structure accordingly.
Run the preprocess.py.
Run the model.py to fine tune the model on custom data.
Run the Auxilary.py.
Run the index.py the driver code and turn on the given link.
