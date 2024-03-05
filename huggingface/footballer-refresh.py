import os
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
import evaluate
import numpy as np

def create_label_mapping(dataset):
    # Extract player names (labels) from the folder structure
    labels = [os.path.basename(os.path.dirname(file_path)) for file_path in dataset]
    unique_labels = list(set(labels))

    label2id = {label: str(idx) for idx, label in enumerate(unique_labels)}
    id2label = {str(idx): label for idx, label in enumerate(unique_labels)}

    return label2id, id2label

def transforms(examples):
    # Use the correct key for accessing a single image
    examples["pixel_values"] = _transforms(examples["image"].convert("RGB"))

    # Extract player names (labels) from the file path in image metadata
    file_path = examples["image"].info.get("file_path", "")
    
    # Use os.path.dirname twice to get the parent folder (player name)
    examples["label"] = [label2id[os.path.basename(os.path.dirname(os.path.dirname(file_path)))]]

    return examples



# Load the dataset
footballer = load_from_disk("/home/isaac-flt/Projects/ML4D/MLProjects/custom_dataset/")
# footballer = footballer.train_test_split(test_size=0.2)

# Create label mapping
label2id, id2label = create_label_mapping(footballer)

# Load the image processor
checkpoint = "IsaacMwesigwa/footballer-retrain-41"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Define image transforms
# Define image transforms
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

#size = (
#    image_processor.size["shortest_edge"]
#    if "shortest_edge" in image_processor.size
#    else (image_processor.size["height"], image_processor.size["width"])
#)
#_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

# Apply transforms to the dataset
footballer = footballer.map(transforms)

# Define data collator
data_collator = DefaultDataCollator()

# Load the accuracy metric
accuracy = evaluate.load("accuracy")

# Define model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_train_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
        # Log relevant training metrics
        print(f"Epoch {state.epoch}: Training Loss {state.log_metrics['train_loss']}")

    def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
        # Log relevant evaluation metrics
        print(f"Epoch {state.epoch}: Evaluation Accuracy {state.log_metrics['eval_accuracy']}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_football_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=footballer["train"],
    eval_dataset=footballer["train"],
    tokenizer=image_processor,
    compute_metrics=lambda p: accuracy.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids),
    callbacks=[LoggingCallback()],
)

# Train the model
trainer.train()


