import argparse
import logging
import json
import sys
import tqdm
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.file_name import make_filename_safe
from utils.pdf_content import extract_from_pdf

DEFAULT_PDF_ROOT_PATH = "/home/scott/ownCloud/Documents/Articles and Papers/"
DEFAULT_LOG_PATH = "process.log"
DEFAULT_PLAN_FILE = "./rename_plan.json"
FORMAT = "[%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s():%(lineno)d] %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename PDF files based on their extracted or synthesized title."
    )
    parser.add_argument(
        "--pdf-root",
        default=DEFAULT_PDF_ROOT_PATH,
        help=f"Directory containing PDF files to process (default: {DEFAULT_PDF_ROOT_PATH})",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        help=f"Path to the log file (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: DEBUG)",
    )
    parser.add_argument(
        "--plan-file",
        default=DEFAULT_PLAN_FILE,
        help=f"Path to the rename plan JSON file used by --dry-run and --apply (default: {DEFAULT_PLAN_FILE})",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write a metadata JSON file per PDF to this directory (created if absent). "
             "Can be combined with the default rename mode.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run LLM extraction and print proposed renames without moving files. "
            "Saves the plan to --plan-file for later use with --apply."
        ),
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Read the rename plan from --plan-file and perform the file renames. "
            "Does not re-run LLM extraction."
        ),
    )
    return parser.parse_args()


def run_dry_run(pdf_root: Path, plan_file: Path) -> int:
    """Run LLM extraction over all PDFs, print proposed renames, and save the plan."""
    logging.info(f"Dry run — reading PDFs from {pdf_root}")
    plan: list[dict] = []

    pdfs = list(pdf_root.glob("*.pdf"))
    for filename in tqdm.tqdm(pdfs):
        try:
            logging.info(f"Processing {filename}")
            title, authors, date, summary = extract_from_pdf(filename)
            if not title["title"]:
                logging.info("Falling back to file title.")
                title["title"] = make_filename_safe(filename.stem)
            clean_stem = make_filename_safe(title["title"])
            destination = filename.parent / (clean_stem + ".pdf")

            print(f"{filename.name}  →  {destination.name}")
            plan.append({
                "source": str(filename),
                "destination": str(destination),
                "title": title,
                "authors": authors,
                "date": date,
                "summary": summary,
            })
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}", exc_info=True)
            print(f"  ERROR  {filename.name}: {e}")

    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nPlan saved to {plan_file}  ({len(plan)} files)")
    return len(plan)


def run_apply(plan_file: Path) -> None:
    """Read the rename plan and perform the file renames."""
    if not plan_file.exists():
        print(f"Error: plan file not found: {plan_file}")
        sys.exit(1)

    with open(plan_file) as f:
        plan: list[dict] = json.load(f)

    logging.info(f"Applying rename plan from {plan_file} ({len(plan)} entries)")
    renamed, skipped = 0, 0

    for entry in plan:
        source = Path(entry["source"])
        destination = Path(entry["destination"])

        if not source.exists():
            logging.warning(f"Source not found, skipping: {source}")
            print(f"  SKIP (not found)  {source.name}")
            skipped += 1
            continue

        if destination.exists() and destination != source:
            logging.warning(f"Destination already exists, skipping: {destination}")
            print(f"  SKIP (exists)     {source.name}  →  {destination.name}")
            skipped += 1
            continue

        source.rename(destination)
        logging.info(f"Renamed {source} → {destination}")
        print(f"  OK  {source.name}  →  {destination.name}")
        renamed += 1

    print(f"\nDone — {renamed} renamed, {skipped} skipped")


def run_full(pdf_root: Path, output_dir: Path | None = None) -> tuple[int, int]:
    """Run LLM extraction and rename each PDF in place.

    :param pdf_root: Directory containing PDF files to process.
    :param output_dir: Optional directory to write one metadata JSON file per PDF.
                       Created automatically if it does not exist.
    """
    logging.info(f"Reading PDFs from {pdf_root}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    renamed, skipped, errors = 0, 0, 0

    for filename in tqdm.tqdm(list(pdf_root.glob("*.pdf"))):
        try:
            logging.info(f"Processing {filename}")
            title, authors, date, summary = extract_from_pdf(filename)
            if not title["title"]:
                logging.info("Falling back to file title.")
                title["title"] = make_filename_safe(filename.stem)
            clean_stem = make_filename_safe(title["title"])
            destination = filename.parent / (clean_stem + ".pdf")

            if destination.exists() and destination != filename:
                logging.warning(f"Destination already exists, skipping: {destination}")
                print(f"  SKIP (exists)  {filename.name}  →  {destination.name}")
                skipped += 1
                continue

            if output_dir:
                record = {
                    "title": title,
                    "authors": authors,
                    "date": date,
                    "summary": summary,
                    "source": str(filename),
                    "destination": clean_stem + ".pdf",
                }
                with open(output_dir / (clean_stem + ".json"), "w") as f:
                    json.dump(record, f, indent=2)
                logging.info(f"Wrote metadata to {output_dir / (clean_stem + '.json')}")

            filename.rename(destination)
            logging.info(f"Renamed {filename} → {destination}")
            print(f"  OK  {filename.name}  →  {destination.name}")
            renamed += 1
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}", exc_info=True)
            print(f"  ERROR  {filename.name}: {e}")
            errors += 1

    print(f"\nDone — {renamed} renamed, {skipped} skipped, {errors} errors")
    return renamed, skipped


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format=FORMAT,
        filename=args.log_path,
        level=getattr(logging, args.log_level),
    )

    if args.apply:
        run_apply(Path(args.plan_file))
    elif args.dry_run:
        run_dry_run(Path(args.pdf_root), Path(args.plan_file))
    else:
        output_dir = Path(args.json) if args.json else None
        run_full(Path(args.pdf_root), output_dir)
