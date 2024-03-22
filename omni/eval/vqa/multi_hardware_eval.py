import os
from multiprocessing import Pool

from omni.utils.eval_utils import get_parser


def run_eval(
    chunk_id,
    model_name,
    gtfile_path,
    image_path,
    out_path,
    num_chunks,
    datatype,
    img_aug,
    beamsearch,
    prompt,
    post_prompt,
    system_prompt,
    task_type,
    clip,
):
    # fmt: off
    os.system(
        "CUDA_VISIBLE_DEVICES=" + str(chunk_id) + " "
        + "python -m omni.eval.vqa.vqa_inference" + " "
        + "--model_name" + " " + model_name + " "
        + "--gtfile_path" +  " " + gtfile_path + " "
        + "--image_path" + " " + image_path + " "
        + "--out_path" +  " " + out_path + " "
        + "--num-chunks" + " " +  str(num_chunks) + " "
        + "--chunk-idx" + " " +  str(chunk_id) + " "
        + "--datatype" + " " + datatype + " "
        + "--img_aug" + " " + img_aug + " "
        + "--beamsearch" + " " + beamsearch + " "
        + "--prompt" + " " + prompt + " "
        + "--post_prompt" + " " + post_prompt + " "
        + "--system_prompt" + " " + system_prompt + " "
        + "--task_type" + " " + task_type + " "
        + "--clip" + " " + clip + " "
    )
    # fmt: on


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if os.path.exists(args.out_path) == False:
        os.makedirs(args.out_path)

    # Multi-GPU evaluation
    with Pool(int(args.num_chunks)) as p:
        for i in range(int(args.num_chunks)):
            chunk_id = i
            p.apply_async(
                run_eval,
                (
                    chunk_id,
                    args.model_name,
                    args.gtfile_path,
                    args.image_path,
                    args.out_path,
                    int(args.num_chunks),
                    args.datatype,
                    args.img_aug,
                    args.beamsearch,
                    args.prompt,
                    args.post_prompt,
                    args.system_prompt,
                    args.task_type,
                    args.clip,
                ),
            )
        p.close()
        p.join()
