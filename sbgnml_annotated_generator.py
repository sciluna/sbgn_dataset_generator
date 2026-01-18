#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "playwright",
#     "requests",
# ]
# ///

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from playwright.sync_api import sync_playwright
from sbgn_gilda_annotator import annotate_sbgn_files, MIN_GROUNDING_SCORE

DOWNLOAD_DIR = Path("./downloads")  # Directory for storing raw downloads.
URL = "https://sciluna.github.io/image-to-sbgn-analysis/dataset/index.html"  # Dataset page URL.

# DOM element identifiers used on the dataset generation page.
NUM_SAMPLES_ID = "num-samples"
GENERATE_BUTTON_ID = "generate-btn"
ALL_CHECKBOX_ID = "singleOrAll"
DOWNLOAD_BUTTON_ID = "download"

LOGGER = logging.getLogger("sbgnml_annotated_generator")


def generate_sbgnml_files(
    num_diagrams: int = 10,
    download_dir: Path = DOWNLOAD_DIR,
    *,
    url: str = URL,
    headless: bool = False,
    download_wait_ms: int = 30000,
) -> List[Path]:
    """Generate SBGN-ML diagrams from the dataset site.

    Args:
        num_diagrams: Number of diagrams to request from the page.
        download_dir: Directory where downloaded assets are stored.
        url: Dataset page URL, overridable for tests.
        headless: Whether to run the browser in headless mode.
        download_wait_ms: Time in milliseconds to wait for downloads to finish.

    Returns:
        A list of paths to downloaded `.sbgnml` files.
    """

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        LOGGER.info("Opening dataset page %s", url)
        page.goto(url)
        page.wait_for_selector(f"#{NUM_SAMPLES_ID}", timeout=10000)
        page.fill(f"#{NUM_SAMPLES_ID}", str(num_diagrams))

        page.click(f"#{GENERATE_BUTTON_ID}")
        page.wait_for_selector(f"#{ALL_CHECKBOX_ID}", timeout=10000)
        page.check(f"#{ALL_CHECKBOX_ID}")

        downloads = []
        page.on("download", lambda download: downloads.append(download))
        page.click(f"#{DOWNLOAD_BUTTON_ID}")
        page.wait_for_timeout(download_wait_ms)

        for idx, download in enumerate(downloads, start=1):
            target_path = download_dir / download.suggested_filename
            download.save_as(str(target_path))
            if target_path.suffix.lower() == ".sbgnml":
                saved_files.append(target_path)
            LOGGER.info("Saved download %s to %s", idx, target_path)

        context.close()
        browser.close()

    return saved_files


def parse_args() -> argparse.Namespace:
    """Build the command-line interface.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate SBGN diagrams and annotate glyphs with HGNC/CHEBI references."
    )
    parser.add_argument("--num-diagrams", type=int, default=10, help="Number of diagrams to request.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help="Directory where raw SBGN archives are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for annotated files. Defaults to in-place updates.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_GROUNDING_SCORE,
        help="Minimum acceptable grounding score.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the browser headlessly (pass --no-headless to show the window).",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Only download SBGN files without annotating them.",
    )

    return parser.parse_args()


def main() -> int:
    """Entrypoint for generating and annotating diagrams.

    Returns:
        Process exit code: 0 on success, non-zero on error conditions.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    generated_files = generate_sbgnml_files(
        num_diagrams=args.num_diagrams,
        download_dir=args.download_dir,
        url=URL,
        headless=args.headless,
    )

    if not generated_files:
        LOGGER.error("No SBGN diagrams were downloaded; aborting before annotation.")
        return 1

    if args.skip_annotation:
        LOGGER.info(
            "Skipping annotation per --skip-annotation; downloaded %s diagram(s).",
            len(generated_files),
        )
        return 0

    annotated = annotate_sbgn_files(
        generated_files,
        output_dir=args.output_dir,
        min_score=args.min_score,
    )

    if not annotated:
        LOGGER.warning("Generation succeeded but no glyphs met the annotation criteria.")
        return 2

    LOGGER.info("Annotated %s diagram(s).", len(annotated))
    return 0


if __name__ == "__main__":
    main()
