import argparse

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file):
    # urls = {
    #     "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
    #     "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    # }
    # filenames = {
    #     "val": "coco_karpathy_val_gt.json",
    #     "test": "coco_karpathy_test_gt.json",
    # }

    # download_url(urls[split], coco_gt_root)
    # annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(coco_gt_root)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gtFile", type=str, required=True)
    parser.add_argument("--resFile", type=str, required=True)
    args = parser.parse_args()
    coco_caption_eval(args.gtFile, args.resFile + "/results_final.json")
