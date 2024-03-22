import shutil
from tqdm import tqdm
import jsonlines
import os

coco_train = "data/coco/train2017"
coco_val = "data/coco/val2017"

os.makedirs("data/coco/lncoco_val2017", exist_ok=True)

with jsonlines.open('data/coco_fid_files/lncoco_captions_val2017.jsonl') as reader:
    for obj in tqdm(reader):
        if os.path.isfile(os.path.join(coco_val, f"{int(obj['image_id']):012d}.jpg")):
            shutil.copy(os.path.join(coco_val, f"{int(obj['image_id']):012d}.jpg"), "data/coco/lncoco_val2017")
        elif os.path.isfile(os.path.join(coco_train, f"{int(obj['image_id']):012d}.jpg")):
            shutil.copy(os.path.join(coco_train, f"{int(obj['image_id']):012d}.jpg"), "data/coco/lncoco_val2017")
        else:
            raise ValueError(f"Image {int(obj['image_id'])} not found in COCO dataset")