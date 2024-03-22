import glob
import pickle

import megfile

from omni.config.lazy import LazyCall as L

from .datasets.conversation_dataset import ConversationDataset
from .datasets.hf_it_pair_dataset import HFITPairDataset
from .datasets.conversation_it_interleaved_dataset import InstructInterleavedITConversationDataset
from .datasets.simple_it_pair_dataset import SimpleITPairDataset
from .datasets.unified_ii_pair_webdataset import UnifiedIIPairWebdataset
from .datasets.unified_it_interleaved_webdataset import UnifiedInterleavedITWebdataset
from .datasets.unified_it_pair_pretokenized_webdataset import UnifiedITTokenPairWebdataset
from .datasets.unified_it_pair_webdataset import UnifiedITPairWebdataset
from .datasets.webvid_vt_pair_dataset import WebVidVTPairDataset
from .manager.data_registry import DataRegistry
from .manager.dataset_info import HFITDatasetInfo, JsonDatasetInfo, SimpleITDatasetInfo, WebDatasetInfo, WebVidDatasetInfo
from .manager.dataset_type import DatasetType


def get_shard_list_and_size_from_index(index_list):
    shard_list = []
    size = 0
    for _index in index_list:
        with megfile.smart_open(_index, "rb") as f:
            content = pickle.load(f)
            for _content in content:
                shard_list.append(_content["url"])
                size += _content["nsamples"]
    return shard_list, size


OBELICS_SHARD_LIST, OBELICS_SIZE = get_shard_list_and_size_from_index(
    megfile.smart_glob("data/resources/oblisc-tarfiles/*.pkl")
)


DATASETS_INFO_TABLE = [
    # ----------------------------------------------------------------------------------------
    # Image-Text Pair
    # ----------------------------------------------------------------------------------------
    L(WebDatasetInfo)(
        name="laion_coco",
        description="""Captioned 600M images from the english subset of Laion-5B with an ensemble of BLIP L/14 and 2 CLIP versions (L/14 and RN50x64).
The method we used to generate these captions was to:
- We use Blip L/14 to generate 40 captions
- Rank them using openai Clip Open AI L/14 ; selected the best 5 captions
- Rank using Open AI RN50x64 Clip model to select the best one
- Use a small, fine-tuned T0 model to roughly repair grammar and punctuation of the texts
The hyperparameters were chosen through a grid search (settings) by Andreas Köpf to best match the style ( ROUGE scores ) of MS COCO texts.
More details: https://laion.ai/blog/laion-coco/
""",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="104.9M",
        shard_list_path="data/resources/pair_data_shards_list/laion_coco_shard_list.json",
    ),
    L(WebDatasetInfo)(
        name="laion2b_en",
        description="Laion 5B project, 2.32 billion of these contain texts in the English language. More details: https://laion.ai/blog/laion-5b/",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="2B",
        shard_list_path="data/resources/laion2b_shard_list.json",
        json_caption_key="caption",
    ),
    L(WebDatasetInfo)(
        name="laion400m",
        description="The data is complete 400M, considering the packet loss problem may be the actual size of 270M, but due to the download command error, resulting in all the images were resized to 256.",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="270M",
        shard_list_path="data/resources/laion400m_resize256_shard_list.json",
    ),
    L(WebDatasetInfo)(
        name="blip_laion",
        description="""115M images whose shorter edge is larger than 256 pixels from the original LAION400M. Then use CapFilt from BLIP to filter high-quality captions.
More details: https://github.com/salesforce/BLIP/tree/main#pre-training-datasets-download
""",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="65M",
        shard_list_path="data/resources/blip_laion_65m_shard_list.json",
    ),
    L(WebDatasetInfo)(
        name="laion400m_orig",
        description="The length and width of the image are the original size, but only 20M was downloaded.",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="20M",
        shard_list_path="data/resources/laion400m_origin20m_shard_list.json",
    ),
    L(WebDatasetInfo)(
        name="journeydb",
        description="4M high-resolution Midjourney images, but we only download 2M. More details: https://journeydb.github.io",
        dataset_type=DatasetType.ImageTextPair,
        cls=UnifiedITPairWebdataset,
        approx_size="2.37M",
        shard_list_path="data/resources/journeydb_2m_tarfile_list.json",
    ),
    L(HFITDatasetInfo)(
        name="pokemon-gpt4-captions",
        description="This dataset is just lambdalabs/pokemon-blip-captions but the captions come from GPT-4 (Turbo).",
        dataset_type=DatasetType.ImageTextPair,
        cls=HFITPairDataset,
        approx_size=833,
        format="arrow",
        data_files="data/resources/huggingface_cache/diffusers--pokemon-gpt4-captions/parquet-train.arrow",
        image_column="image",
        text_column="text",
    ),
    L(HFITDatasetInfo)(
        name="dalle3-by-laion",
        description="""This dataset consists of prompt and image URL pairs scraped from the LAION share-dalle-3 discord channel.
More details: https://huggingface.co/datasets/laion/dalle-3-dataset
""",
        dataset_type=DatasetType.ImageTextPair,
        cls=HFITPairDataset,
        approx_size=13020,
        format="arrow",
        data_files=glob.glob("data/resources/huggingface_cache/laion--dalle-3-dataset/*.arrow"),
        image_column="image",
        text_column="caption",
    ),
    L(WebDatasetInfo)(
        name="laion_coco_ii",
        description="From Laion400M",
        dataset_type=DatasetType.ImageImagePair,
        cls=UnifiedIIPairWebdataset,
        approx_size="100M",
        shard_list_path="data/resources/pair_data_shards_list/laion_coco_shard_list.json",
    ),
    L(WebDatasetInfo)(
        name="blip_laion_ii",
        description="From Laion400M",
        dataset_type=DatasetType.ImageImagePair,
        cls=UnifiedIIPairWebdataset,
        approx_size="65M",
        shard_list_path="data/resources/blip_laion_65m_shard_list.json",
    ),
    # ----------------------------------------------------------------------------------------
    # Interleaved Image Text
    # ----------------------------------------------------------------------------------------
    L(WebDatasetInfo)(
        name="mmc4_core",
        description="MMC4 core dataset.",
        dataset_type=DatasetType.InterleavedImageText,
        cls=UnifiedInterleavedITWebdataset,
        approx_size="7M",
        shard_list=megfile.smart_glob("data/resources/mmc4-webdataset/*.tar"),
    ),
    L(WebDatasetInfo)(
        name="obelics",
        description="Obelics dataset.",
        dataset_type=DatasetType.InterleavedImageText,
        cls=UnifiedInterleavedITWebdataset,
        approx_size=OBELICS_SIZE,  # 113M
        shard_list=OBELICS_SHARD_LIST,
    ),
    L(JsonDatasetInfo)(
        name="mmc4_instruct_filtered224",
        description="Instruction-following interleaved content creation data constructed by ChatGPT-3.5.",
        dataset_type=DatasetType.InstructInterleavedImageText,
        cls=InstructInterleavedITConversationDataset,
        approx_size="20321",
        json_list=megfile.smart_glob("data/resources/instruction_data/mmc4sft_filter_repeat_res224/*.json"),
        root="data/resources/vision-language-data/mmc4-core-raw/",
    ),
    # ----------------------------------------------------------------------------------------
    # Video-Text Pair
    # ----------------------------------------------------------------------------------------
    L(WebVidDatasetInfo)(
        name="webvid",
        description="""Large-scale text-video dataset, containing 10 million video-text pairs scraped from the stock footage sites. The video is about 15 seconds on average.
More details: https://maxbain.com/webvid-dataset/""",
        dataset_type=DatasetType.VideoTextPair,
        cls=WebVidVTPairDataset,
        approx_size="10.7M",
        shard_list=glob.glob("data/resources/webvid_shards/*.json"),
        data_dir="data/resources/video/webvid_0912/10Mtrain/videos/",
    ),
    # ----------------------------------------------------------------------------------------
    # Instruction
    # ----------------------------------------------------------------------------------------
    L(JsonDatasetInfo)(
        name="llava_pretrain",
        description="558K unique language-image instruction-following samples for image description, constructed from BLIP-CC (CC3M).",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size="558K",
        json_path="data/resources/vision-language-data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        root="data/resources/vision-language-data/VisionPretrainDatasets/BLIP-CCSL-558k/",
    ),
    L(JsonDatasetInfo)(
        name="gqa",
        description="GQA dataset.",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size=13532530,
        json_path="data/resources/dreamllm_data/gqa_sft_train_short_filtered.json",
        root="data/resources/VisionPretrainDatasets/GQA/images/",
    ),
    L(JsonDatasetInfo)(
        name="llava_instruct",
        description="158K unique language-image instruction-following samples in total, including 58K in conversations, 23K in detailed description, and 77K in complex reasoning, respectively.",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size="158K",
        json_path="data/resources/LLaVA-Instruct-150K/llava_instruct_150k.json",
        root="data/COCO/train2017/",
    ),
    L(JsonDatasetInfo)(
        name="llavav1.5_instruct",
        description="665K unique language-image instruction-following samples in total, constructed by LLaVA1.5 by mixing different VQA data.",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size="665298",
        json_path="/data/hypertext/yancie/dataset/LLaVA1.5/llava_v1_5_mix665k_s3path.json",
        root="",
    ),
    L(JsonDatasetInfo)(
        name="llava_instruct_filter",
        description="llava_instruct_158K filtered.",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size="80K",
        json_path="data/resources/instruction_data/llava_instruct_80k.json",
        root="data/COCO/train2017/",
    ),
    L(JsonDatasetInfo)(
        name="instruct_blip_laion",
        description="Instruction-following text-to-image generation data constructed by ChatGPT-3.5.",
        dataset_type=DatasetType.Conversation,
        cls=ConversationDataset,
        approx_size="22K",
        json_path="data/resources/text2image_sft_data/instruct_pair_blip_laion_22k.json",
        root="data/resources/blip_laion/",
    ),
]

DataManager = DataRegistry("DataManager")
DataManager.register(DATASETS_INFO_TABLE)
