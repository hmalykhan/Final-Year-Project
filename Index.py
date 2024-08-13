from Auxilary import load_embeddings, extract_frames, diction, check, create_video_from_images
import gradio as gr
import os

def process_video(video):
    ref_embeddings=load_embeddings()
    extract_frames(video,0,"query_frames")
    query_embeddings=diction("/home/hmalykhan/Desktop/fynal_year_project/query_frames")
    # check(ref_embeddings,query_embeddings)
    if check(ref_embeddings,query_embeddings)==False:
        segment,reference,vid=create_video_from_images('/home/hmalykhan/Desktop/fynal_year_project/result_query','/home/hmalykhan/Desktop/fynal_year_project/result_videos' )
        # Save the video to a specific location
        # filename = os.path.basename(video)  # Getting the file name from the video object
        # ultimate_path = os.path.join("/home/hmalykhan/Desktop/mutawakkil/check_upload", filename)
        # output_path = "/home/hmalykhan/Desktop/mutawakkil/check_store/R4.mp4"
        # vid = "R3.mp4"
        
        # with open(ultimate_path, "wb") as f:
        #     f.write(video)
        
        return segment, reference, vid
    else:
        return None, None, "This video does not match reference dataset."

interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[gr.Video(label="Copied Segment"), gr.Video(label="Reference Video"), gr.Text(label="Reference Video")],
)

interface.launch()