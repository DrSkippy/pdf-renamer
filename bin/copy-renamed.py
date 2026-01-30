#!/usr/bin/env python3
"""
Utility to copy files based on JSON metadata in the output directory.

Reads each JSON file in ./output, copies the file specified in 'source'
to the output directory with the filename specified in 'destination'.
"""

import json
import shutil
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "output"

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in output directory.")
        sys.exit(0)

    copied = 0
    errors = 0

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            source = data.get("source")
            destination = data.get("destination")

            if not source or not destination:
                print(f"Skipping {json_file.name}: missing source or destination")
                continue

            source_path = Path(source)
            dest_path = output_dir / destination

            if not source_path.exists():
                print(f"Skipping {json_file.name}: source file not found: {source}")
                errors += 1
                continue

            shutil.copy2(source_path, dest_path)
            print(f"Copied: {source_path.name} -> {destination}")
            copied += 1

        except json.JSONDecodeError as e:
            print(f"Error parsing {json_file.name}: {e}")
            errors += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            errors += 1

    print(f"\nDone. Copied: {copied}, Errors: {errors}")


if __name__ == "__main__":
    main()
