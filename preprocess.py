import shutil, os, json
from Auxilary import extract_frames, diction, nmparray
def reference_embeddings():
    shutil.rmtree(f'/home/hmalykhan/Desktop/fynal_year_project/frames')
    os.mkdir(f'/home/hmalykhan/Desktop/fynal_year_project/frames')
    count=0
    for i in range(len(os.listdir("/home/hmalykhan/Desktop/fynal_year_project/ref"))):
        count=extract_frames(f"/home/hmalykhan/Desktop/fynal_year_project/ref/R{i+1}.mp4",count,"frames")
    embd=diction("/home/hmalykhan/Desktop/fynal_year_project/frames")
    ref_embeddings=nmparray(embd)
    with open('/home/hmalykhan/Desktop/fynal_year_project/embeddings/ref_embd.json','w') as output:
        json.dump(ref_embeddings,output)