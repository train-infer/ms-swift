# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json

from swift.loss_scale.role import inject_role_loss_scale_jsonl


def parse_args():
    """Parse command-line arguments for role loss-scale injection."""
    parser = argparse.ArgumentParser(description='Inject role-based loss_scale values into an Agent JSONL dataset.')
    parser.add_argument('--input', required=True, help='Source JSONL path.')
    parser.add_argument('--config', required=True, help='Role loss-scale JSON config path.')
    parser.add_argument('--output', required=True, help='Destination JSONL path. Must differ from --input.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing message.loss_scale values.')
    return parser.parse_args()


def main():
    """Run streaming injection and print the processing summary."""
    args = parse_args()
    summary = inject_role_loss_scale_jsonl(
        args.input, args.config, args.output, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
