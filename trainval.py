import argparse
from utils.config import get_exp_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/config.json", type=str, help="Config file path.")
    parser.add_argument('--dataset', default="eth", type=str, help="Dataset name.")
    parser.add_argument('--tag', default="LMTraj", type=str, help="Personal tag for the model.")
    parser.add_argument('--test', default=False, action='store_true', help="Evaluation mode.")
    parser.add_argument('--steps-dir', default=None, type=str, help="Optional override for trajectory txt directory.")
    parser.add_argument('--image-dir', default=None, type=str, help="Optional override for scene image directory.")
    parser.add_argument('--homography-dir', default=None, type=str, help="Optional override for homography matrix directory.")
    parser.add_argument('--caption-dir', default=None, type=str, help="Optional override for caption directory.")
    parser.add_argument('--caption-suffix', default=None, type=str, help="Caption filename suffix (e.g., _caption_chatgpt4.txt).")
    parser.add_argument('--strip-token', action='append', default=None, help="Suffix/pattern to strip from scene names when matching assets. Can be used multiple times.")
    parser.add_argument('--reference-suffix', default=None, type=str, help="Reference scene image suffix (default: _reference.png).")
    parser.add_argument('--oracle-suffix', default=None, type=str, help="Oracle/segmentation image suffix (default: _oracle.png).")

    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    cfg = get_exp_config(args.cfg)
    print_arguments(cfg)

    # Update configs
    cfg.dataset_name = args.dataset
    cfg.checkpoint_name = args.tag
    cfg.trajectory_dir = args.steps_dir or getattr(cfg, "trajectory_dir", None)
    cfg.image_dir = args.image_dir or getattr(cfg, "image_dir", None)
    cfg.homography_dir = args.homography_dir or getattr(cfg, "homography_dir", None)
    cfg.caption_dir = args.caption_dir or getattr(cfg, "caption_dir", None)
    cfg.caption_suffix = args.caption_suffix or getattr(cfg, "caption_suffix", None)
    cfg.strip_scene_tokens = args.strip_token or getattr(cfg, "strip_scene_tokens", None)
    cfg.reference_image_suffix = args.reference_suffix or getattr(cfg, "reference_image_suffix", None)
    cfg.oracle_image_suffix = args.oracle_suffix or getattr(cfg, "oracle_image_suffix", None)

    if not args.test:
        # Training phase
        from model.trainval_accelerator import *
        trainval(cfg)

    else:
        # Evaluation phase
        from model.eval_accelerator import *
        test(cfg)
