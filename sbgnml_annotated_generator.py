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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests
from playwright.sync_api import sync_playwright

DOWNLOAD_DIR = Path("./downloads")  # Directory for storing raw downloads.
URL = "https://sciluna.github.io/image-to-sbgn-analysis/dataset/index.html"  # Dataset page URL.
GROUNDING_SERVICE_URL = "http://grounding.indra.bio/"  # External grounding service base URL.
MIN_GROUNDING_SCORE = 0.75  # Minimum grounding confidence accepted for annotations.
ALLOWED_DATABASES = {"CHEBI", "HGNC"}  # Databases allowed for grounding references.
CLASS_DATABASE_HINTS: Dict[str, str] = {
    "macromolecule": "HGNC",
    "simple chemical": "CHEBI",
}  # Map SBGN glyph class to expected grounding database.

# DOM element identifiers used on the dataset generation page.
NUM_SAMPLES_ID = "num-samples"
GENERATE_BUTTON_ID = "generate-btn"
ALL_CHECKBOX_ID = "singleOrAll"
DOWNLOAD_BUTTON_ID = "download"

# XML namespaces used throughout SBGN and RDF annotations.
SBGN_NS = "http://sbgn.org/libsbgn/0.3"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
BQMODEL_NS = "http://biomodels.net/model-qualifiers/"

ET.register_namespace("", SBGN_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("bqmodel", BQMODEL_NS)

LOGGER = logging.getLogger("sbgnml_annotated_generator")
_GROUNDING_CACHE: Dict[tuple[str, Optional[str]], Optional[str]] = defaultdict(lambda: None)


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


def annotate_sbgnml_files(
    sbgn_files: Iterable[Path],
    *,
    output_dir: Optional[Path] = None,
    min_score: float = MIN_GROUNDING_SCORE,
) -> List[Path]:
    """Annotate glyphs in SBGN files using HGNC/CHEBI references.

    Args:
        sbgn_files: Paths to SBGN files that should be annotated.
        output_dir: Optional directory for writing annotated copies.
        min_score: Minimum acceptable grounding score.

    Returns:
        A list of paths pointing to annotated SBGN files.
    """

    annotated_paths: List[Path] = []
    missing_eligible: List[Path] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sbgn_file in sbgn_files:
        LOGGER.info("Processing %s", sbgn_file)
        tree = ET.parse(sbgn_file)
        root = tree.getroot()
        sbgn_ns = _detect_sbgn_namespace(root)

        modified, eligible_found = _annotate_tree(
            tree,
            sbgn_ns=sbgn_ns,
            grounding_service_url=GROUNDING_SERVICE_URL,
            min_score=min_score,
        )
        if not eligible_found:
            LOGGER.warning("No eligible glyphs found in %s; skipping annotation.", sbgn_file)
            missing_eligible.append(Path(sbgn_file))
            continue
        if not modified:
            LOGGER.info(
                "Eligible glyphs found but no annotations added to %s (groundings filtered).",
                sbgn_file,
            )
            continue

        target = output_dir / sbgn_file.name if output_dir else sbgn_file
        ET.register_namespace("", sbgn_ns)
        ET.indent(tree, space="  ", level=0)  # type: ignore[attr-defined]
        tree.write(target, encoding="utf-8", xml_declaration=True)
        annotated_paths.append(target)
        LOGGER.info("Annotated glyphs written to %s", target)

    if missing_eligible:
        LOGGER.warning(
            "Skipped %s file(s) without eligible glyphs: %s",
            len(missing_eligible),
            ', '.join(str(path) for path in missing_eligible),
        )

    return annotated_paths


def _annotate_tree(
    tree: ET.ElementTree,
    *,
    sbgn_ns: str,
    grounding_service_url: str,
    min_score: float,
) -> tuple[bool, bool]:
    """Annotate eligible glyphs found within an SBGN tree.

    Args:
        tree: Parsed SBGN XML tree.
        sbgn_ns: Namespace of the SBGN document.
        grounding_service_url: Grounding service endpoint.
        min_score: Minimum acceptable grounding score.

    Returns:
        A tuple of `(modified, eligible_found)` describing whether annotations
        were added and whether at least one eligible glyph was present.
    """

    root = tree.getroot()
    namespaces = {"sbgn": sbgn_ns}
    glyphs = root.findall(".//sbgn:glyph", namespaces=namespaces)
    modified = False
    eligible_found = False

    for glyph in glyphs:
        glyph_class = (glyph.get("class") or "").strip().lower()
        expected_db = CLASS_DATABASE_HINTS.get(glyph_class)
        if not expected_db:
            continue

        label = glyph.find("sbgn:label", namespaces=namespaces)
        if label is None:
            continue

        text = (label.get("text") or "").strip()
        if not text:
            continue

        eligible_found = True
        resource = _ground_label(
            text,
            grounding_service_url=grounding_service_url,
            min_score=min_score,
            expected_db=expected_db,
        )
        if not resource:
            continue

        if _attach_annotation(glyph, resource, sbgn_ns):
            modified = True

    return modified, eligible_found


def _ground_label(
    text: str,
    *,
    grounding_service_url: str,
    min_score: float,
    expected_db: Optional[str] = None,
) -> Optional[str]:
    """Ground a glyph label against the configured grounding service.

    Args:
        text: Glyph label text to ground.
        grounding_service_url: Base URL for the grounding service.
        min_score: Minimum acceptable grounding confidence.
        expected_db: Optional database that the grounding result must match.

    Returns:
        A formatted identifiers.org URI if grounding succeeds, otherwise None.
    """

    query = text.strip()
    if not query:
        return None

    cache_key = (query.lower(), expected_db)
    if cache_key in _GROUNDING_CACHE:
        return _GROUNDING_CACHE[cache_key]

    try:
        resp = requests.post(
            urljoin(grounding_service_url, "ground"),
            json={"text": query},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as err:
        LOGGER.warning("Grounding request failed for %s: %s", query, err)
        return None

    for candidate in resp.json():
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if score < min_score:
            continue

        term = candidate.get("term") or {}
        db = (term.get("db") or "").upper()
        if expected_db and db != expected_db:
            continue
        if db not in ALLOWED_DATABASES:
            continue

        term_id = term.get("id")
        if not term_id:
            continue

        resource = _format_resource_uri(db, term_id)
        _GROUNDING_CACHE[cache_key] = resource
        return resource

    _GROUNDING_CACHE[cache_key] = None
    return None


def _format_resource_uri(db: str, term_id: str) -> str:
    """Create an identifiers.org URI for a grounded term.

    Args:
        db: Database prefix (e.g., HGNC).
        term_id: Identifier returned by the grounding service.

    Returns:
        Fully-qualified identifiers.org URI.
    """
    upper_db = db.upper()
    accession = term_id
    if term_id.upper().startswith(f"{upper_db}:"):
        accession = term_id.split(":", 1)[1]
    return f"http://identifiers.org/{upper_db}:{accession}"


def _attach_annotation(glyph: ET.Element, resource_uri: str, sbgn_ns: str) -> bool:
    """Attach RDF annotations to a glyph if the resource is new.

    Args:
        glyph: Glyph element being annotated.
        resource_uri: identifiers.org URI to attach.
        sbgn_ns: Namespace of the SBGN document.

    Returns:
        True if the glyph was modified, False otherwise.
    """
    glyph_id = glyph.get("id")
    if not glyph_id:
        return False

    extension = glyph.find(f"./{{{sbgn_ns}}}extension")
    if extension is None:
        extension = ET.SubElement(glyph, f"{{{sbgn_ns}}}extension")

    annotation = extension.find(f"./{{{sbgn_ns}}}annotation")
    if annotation is None:
        annotation = ET.SubElement(extension, f"{{{sbgn_ns}}}annotation")

    rdf = annotation.find(f"./{{{RDF_NS}}}RDF")
    if rdf is None:
        rdf = ET.SubElement(annotation, f"{{{RDF_NS}}}RDF")

    description = None
    for candidate in rdf.findall(f"./{{{RDF_NS}}}Description"):
        about = candidate.attrib.get(f"{{{RDF_NS}}}about")
        if about in (glyph_id, f"#{glyph_id}"):
            description = candidate
            break
    if description is None:
        description = ET.SubElement(
            rdf,
            f"{{{RDF_NS}}}Description",
            attrib={f"{{{RDF_NS}}}about": f"#{glyph_id}"},
        )

    bqmodel_is = description.find(f"./{{{BQMODEL_NS}}}is")
    if bqmodel_is is None:
        bqmodel_is = ET.SubElement(description, f"{{{BQMODEL_NS}}}is")

    bag = bqmodel_is.find(f"./{{{RDF_NS}}}Bag")
    if bag is None:
        bag = ET.SubElement(bqmodel_is, f"{{{RDF_NS}}}Bag")

    for li in bag.findall(f"./{{{RDF_NS}}}li"):
        if li.attrib.get(f"{{{RDF_NS}}}resource") == resource_uri:
            return False

    ET.SubElement(bag, f"{{{RDF_NS}}}li", attrib={f"{{{RDF_NS}}}resource": resource_uri})
    return True


def _detect_sbgn_namespace(root: ET.Element) -> str:
    """Return the namespace used by the SBGN document.

    Args:
        root: Root element of the SBGN XML tree.

    Returns:
        Namespace URI defined on the root element.
    """
    tag = root.tag
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return SBGN_NS


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
        action="store_true",
        help="Run the browser in headless mode instead of showing the window.",
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

    annotated = annotate_sbgnml_files(
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
