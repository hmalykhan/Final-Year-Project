import os
import torch
from torch.utils.data import DataLoader, Dataset,random_split
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments

def llm(num):
    for i in range(len("/home/hmalykhan/Desktop/fynal_year_project/ref")+1):
                    if(os.path.exists(f"/home/hmalykhan/Desktop/fynal_year_project/frames/R{i+1}_{num}.png")):
                        return i+1
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [f"R{llm(i)}_{i}.png" for i in range(0, 1644)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")
        label = idx  # Assuming label is the index; modify as needed for your case

        if self.transform:
            image = self.transform(image)

        return image, label

# Set up directories
img_dir = "/home/hmalykhan/Desktop/fynal_year_project/frames"  # Modify this path to your dataset directory

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1] range
])

# Load dataset
dataset = CustomDataset(img_dir, transform=transform)

# Split dataset into train and eval sets (80% train, 20% eval)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# Load the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=1644,
    ignore_mismatched_sizes=True
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Custom collate function
def collate_fn(batch):
    images, labels = zip(*batch)
    inputs = feature_extractor(images, return_tensors="pt")
    labels = torch.tensor(labels)
    return {"pixel_values": inputs['pixel_values'], "labels": labels}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the model
# trainer.save_model("./fine_tuned_vit")
trainer.save_model("./trained_model")
trainer.eval()
