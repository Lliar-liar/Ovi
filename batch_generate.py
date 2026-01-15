import os
import sys
import json
import re
import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import validate_and_process_user_prompt
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info,
)
from ovi.ovi_fusion_engine import OviFusionEngine


def _init_logging(rank: int):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def sanitize_filename(name: str, max_len: int = 180) -> str:
    """Make 'content' safe for filename."""
    name = "" if name is None else str(name)
    name = name.strip()
    if not name:
        name = "empty"

    # Replace forbidden characters for common filesystems
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name)
    # Collapse spaces
    name = re.sub(r"\s+", " ", name).strip()

    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def load_tasks_from_json_folder(json_dir: str):
    """
    Return list of tasks:
      [{"json_name": "...", "content": "...", "prompt": "..."}, ...]
    """
    json_files = [
        os.path.join(json_dir, fn)
        for fn in os.listdir(json_dir)
        if fn.lower().endswith(".json")
    ]
    json_files.sort()

    tasks = []
    for jf in json_files:
        json_name = os.path.splitext(os.path.basename(jf))[0]
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{jf} must be a JSON list, got {type(data)}")

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logging.warning(f"Skip non-dict: file={jf} idx={i} type={type(item)}")
                continue
            if "prompt" not in item or "content" not in item:
                logging.warning(
                    f"Skip missing keys: file={jf} idx={i} keys={list(item.keys())}"
                )
                continue
            tasks.append(
                {
                    "json_name": json_name,
                    "content": item["content"],
                    "prompt": item["prompt"],
                }
            )
    return tasks


def main():
    """
    Usage:
      torchrun --nproc_per_node=8 batch_infer_json_folder.py /path/to/config.yaml /path/to/json_dir /output
    """
    if len(sys.argv) < 4:
        print(
            "Usage: batch_infer_json_folder.py <config.yaml> <json_dir> <output_root>\n"
            "Example: torchrun --nproc_per_node=8 batch_infer_json_folder.py config.yaml ./prompts /output"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    json_dir = sys.argv[2]
    output_root = sys.argv[3]

    config = OmegaConf.load(config_path)

    # ===== distributed init (same as your inference) =====
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)

    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, (
        "sp_size must be <= world_size and world_size % sp_size == 0"
    )

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size,
        )
    else:
        assert sp_size == 1, f"When world_size=1, sp_size must be 1, got {sp_size}"

    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")

    # ===== load tasks =====
    tasks = load_tasks_from_json_folder(json_dir)
    if global_rank == 0:
        logging.info(f"Loaded {len(tasks)} tasks from {json_dir}")

    # ===== SP distribution (same logic as your code) =====
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    this_rank_tasks = tasks[sp_group_id :: num_sp_groups] if tasks else []

    # ===== validate mode =====
    assert config.get("mode") in ["t2v", "i2v", "t2i2v"], (
        f"Invalid mode {config.get('mode')}"
    )

    # ===== load engine =====
    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=torch.bfloat16)
    logging.info("OVI Fusion Engine loaded!")

    # ===== generation parameters =====
    video_frame_height_width = config.get("video_frame_height_width", None)
    seed0 = config.get("seed", 100)
    solver_name = config.get("solver_name", "unipc")
    sample_steps = config.get("sample_steps", 50)
    shift = config.get("shift", 5.0)
    video_guidance_scale = config.get("video_guidance_scale", 4.0)
    audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
    slg_layer = config.get("slg_layer", 11)
    video_negative_prompt = config.get("video_negative_prompt", "")
    audio_negative_prompt = config.get("audio_negative_prompt", "")
    each_example_n_times = config.get("each_example_n_times", 1)

    # config里的 image_path 如果你是 t2v 通常没用；这里只保持和原逻辑一致
    default_image_path = config.get("image_path", None)

    # ===== run =====
    # 进度条只在 rank0 显示，避免刷屏
    iterator = enumerate(this_rank_tasks)
    if global_rank == 0:
        iterator = tqdm(list(iterator), total=len(this_rank_tasks))

    for idx, task in iterator:
        prompt = task["prompt"]
        content = task["content"]
        json_name = task["json_name"]

        # validate prompt format (and image in i2v)
        text_prompts, image_paths = validate_and_process_user_prompt(
            prompt, default_image_path, mode=config.get("mode")
        )
        text_prompt = text_prompts[0]
        image_path = image_paths[0] if image_paths else None

        if config.get("mode") != "i2v":
            image_path = None
        else:
            if not (image_path and os.path.isfile(image_path)):
                raise FileNotFoundError(f"i2v requires valid image_path, got: {image_path}")

        out_dir = os.path.join(output_root, json_name)
        safe_stem = sanitize_filename(content)
        base_mp4 = os.path.join(out_dir, f"{safe_stem}.mp4")

        # 如果你想“已存在就跳过”，打开下面两行
        # if sp_rank == 0 and os.path.exists(base_mp4):
        #     continue

        for rep in range(each_example_n_times):
            seed = seed0 + rep

            generated_video, generated_audio, generated_image = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_path,
                video_frame_height_width=video_frame_height_width,
                seed=seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
            )

            if sp_rank == 0:
                os.makedirs(out_dir, exist_ok=True)

                out_mp4 = base_mp4
                # 多次采样时避免覆盖
                if each_example_n_times > 1:
                    out_mp4 = os.path.join(out_dir, f"{safe_stem}_seed{seed}.mp4")

                save_video(out_mp4, generated_video, generated_audio, fps=24, sample_rate=16000)

                if generated_image is not None:
                    generated_image.save(out_mp4.replace(".mp4", ".png"))


if __name__ == "__main__":
    main()
