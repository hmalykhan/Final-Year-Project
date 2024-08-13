import torch ,os, json, shutil ,cv2, torch
import re
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

# Load pre-trained Vision Transformer and feature extractor
model_path = "./trained_model"
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
# model = ViTModel.from_pretrained(model_path)

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = Compose([
        Resize((224, 224)),  # ViT models typically use 224x224 images
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to get image embeddings
def get_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        # Use the CLS token representation as the image embedding
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embedding1, embedding2).item()

# Funciotn which provides the dictionary of embeddings where key is id and value is embedding in lis
def diction(path):
    embd=dict()
    # print(len(os.listdir(path)))
    for i in range(len(os.listdir(path))):
        print("Descripting the frame : ",i)
        embedding = get_image_embedding(path+f"/{i}.png")
        # embd[i]=embedding.tolist
        embd[i]=embedding
    return embd

# Funciton extract the frames from video in source directory and save frames in destination directory.
def extract_frames( video_path,frame_count,pad):
    shutil.rmtree(f'/home/hmalykhan/Desktop/fynal_year_project/{pad}')
    os.mkdir(f'/home/hmalykhan/Desktop/fynal_year_project/{pad}')
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Initialize frame count
    # frame_count = 0

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as an image
        # output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_count:04d}.jpg")
        # cv2.imwrite(output_path, frame)
        # cv2.imwrite(f"/home/hmalykhan/Desktop/mutawakkil/frames/R{pad}_{frame_count}.png",frame)
        cv2.imwrite(f"/home/hmalykhan/Desktop/fynal_year_project/{pad}/{frame_count}.png",frame)
        # Increment frame count
        frame_count += 1

        # Display frame count
        print(f"Extracted frame {frame_count} from '{os.path.basename(video_path)}'")

    # Release the video capture object and close the video file
    cap.release()
    return frame_count

# Return the embedding of reference video frames
def load_embeddings():
    with open("/home/hmalykhan/Desktop/fynal_year_project/embeddings/ref_embd.json", 'r') as input_file:
        loaded_emb = json.load(input_file)
    emb = {int(k): np.array(v) for k, v in loaded_emb.items()}
    print("the length of dictionary is : ",len(emb))
    return emb

# convert the list to numpy array
def nmparray(dic):
    emb=dict()
    emb = {int(k): np.array(v) for k, v in dic.items()}
    return emb

# give the starting and ending frames of the mathced video
def extract_ref_frames( video_path,frame_count,pad,lin):
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Initialize frame count
    # frame_count = 0

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as an image
        # output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_count:04d}.jpg")
        # cv2.imwrite(output_path, frame)
        # cv2.imwrite(f"/home/hmalykhan/Desktop/mutawakkil/frames/R{pad}_{frame_count}.png",frame)
        cv2.imwrite(f"/home/hmalykhan/Desktop/fynal_year_project/{pad}/R{lin}_{frame_count}.png",frame)
        # Increment frame count
        frame_count += 1

        # Display frame count
        print(f"Extracted frame {frame_count} from '{os.path.basename(video_path)}'")

    # Release the video capture object and close the video file
    cap.release()
    return frame_count

# Function which create video from frames
def create_video_from_images(image_folder, output_video_path, frame_rate=30):
    # Get the list of all image files in the folder
    shutil.rmtree('/home/hmalykhan/Desktop/fynal_year_project/result_videos')
    os.mkdir('/home/hmalykhan/Desktop/fynal_year_project/result_videos')
    ref_video_path="/home/hmalykhan/Desktop/fynal_year_project/ref"
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    # Sort the images by name//
    images.sort()

    # # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    pattern = r'R(\d+)'
    string=images[0]
    match = re.search(pattern, string)
    number = match.group(1)
    vid=f"R{str(number)}.mp4"
    # print(number)
    print(f"R{str(number)}.mp4")
    output_video_path=os.path.join(output_video_path,f"R{str(number)}.mp4")
    ref_video_path=os.path.join(ref_video_path,f"R{str(number)}.mp4")
    # print(output_video_path)
    # print(ref_video_path)
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for .avi or 'mp4v' for .mp4
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write the images to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer object
    video.release()
    return output_video_path,ref_video_path,vid
def llm(num):
    for i in range(len("/home/hmalykhan/Desktop/fynal_year_project/ref")+1):
                    if(os.path.exists(f"/home/hmalykhan/Desktop/fynal_year_project/frames/R{i+1}_{num}.png")):
                        return i+1
def check(ref_embeddings,query_embeddings,stand=0):
    qc=0
    rc=0
    start=0
    end=0
    tokken=True
    # for real
    threshold = 0.96
    # for fine tunned
    # threshold=0.95
    
    for a in range(stand,len(query_embeddings)):
        for l in range(len(ref_embeddings)):
            similarity = cosine_similarity(query_embeddings[a], torch.tensor(ref_embeddings[l]))
            
            if similarity >= threshold:
                # print(a,l)
                qc=a
                rc=l
                start=rc
                tokken=False
                while qc < len(query_embeddings) and rc < len(ref_embeddings) and cosine_similarity(query_embeddings[qc], torch.tensor(ref_embeddings[rc]))>=threshold:
                    # print(similarity)
                    qc+=1
                    rc+=1
                    # print(qc,rc)
                # print(cosine_similarity(query_embeddings[qc], torch.tensor(ref_embeddings[rc])))
                count=qc
                end=rc-1
                # print(a,l)
                print("The matched frames from the reference dataset ",a," : ",l)
                shutil.rmtree('/home/hmalykhan/Desktop/fynal_year_project/result_query')
                os.mkdir('/home/hmalykhan/Desktop/fynal_year_project/result_query')
                for i in range(start,end+1):
                    num=llm(i)
                    src=os.path.join("/home/hmalykhan/Desktop/fynal_year_project/frames/",f"R{num}_{i}.png")
                    dst=os.path.join("/home/hmalykhan/Desktop/fynal_year_project/result_query/",f"R{num}_{i}.png")
                    shutil.copyfile(src,dst)
                # if tokken:
                #     return tokken
                return tokken