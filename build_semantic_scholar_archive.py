import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


RESPONSES_PATH = Path("Dataset/gemini_references/responses_parsed.json")
BASE_DIR = Path("Dataset/semantic_scholar")
PAPERS_DIR = BASE_DIR / "papers"
PDF_DIR = BASE_DIR / "pdfs"
LOG_DIR = BASE_DIR / "logs"
PRINCIPLES_INDEX_PATH = BASE_DIR / "principles.json"
NOT_FOUND_LOG = LOG_DIR / "not_found.jsonl"
ERROR_LOG = LOG_DIR / "errors.jsonl"

NUMBERED_PREFIX_RE = re.compile(r"^\d+[\).\s]+")
MARKDOWN_MARKERS_RE = re.compile(r"[\*\u2013\u2014]")
TRAILING_NOTE_RE = re.compile(r"\s*\([^)]*[\u4e00-\u9fff][^)]*\)\s*$")
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\)]+)", re.IGNORECASE)

SEARCH_FIELDS_DEFAULT = "paperId,title,abstract,authors,venue,year,citationCount,openAccessPdf"

_S2_BASE_URL_CACHE: Optional[str] = None


def get_s2_base_url() -> str:
    global _S2_BASE_URL_CACHE
    if _S2_BASE_URL_CACHE is None:
        base = os.getenv("S2_BASE_URL")
        if not base:
            raise RuntimeError("S2_BASE_URL environment variable is not set.")
        base = base.rstrip("/") + "/"
        if "graph/v1/" not in base:
            base = base + "graph/v1/"
        _S2_BASE_URL_CACHE = base
    return _S2_BASE_URL_CACHE


@dataclass
class CitationInfo:
    principle: str
    citation_raw: str
    index: int
    doi: Optional[str]
    title_hint: Optional[str]


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def slugify(value: str, length: int = 80) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:length] or sha1(value.encode("utf-8")).hexdigest()[:length]


class PaperRegistry:
    def __init__(self) -> None:
        self.papers_dir = PAPERS_DIR
        self.pdf_dir = PDF_DIR
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.by_key: Dict[str, Dict] = {}
        self.by_doi: Dict[str, str] = {}
        self.by_paper_id: Dict[str, str] = {}
        self.by_title: Dict[str, str] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for path in self.papers_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            paper_key = path.stem
            self.by_key[paper_key] = data
            doi = data.get("doi")
            if doi:
                self.by_doi[doi.lower()] = paper_key
            paper_id = data.get("paperId")
            if paper_id:
                self.by_paper_id[paper_id] = paper_key
            title = data.get("title")
            if title:
                self.by_title[normalize_title(title)] = paper_key

    def get_by_identifiers(self, doi: Optional[str], title: Optional[str]) -> Optional[Tuple[str, Dict]]:
        if doi:
            key = self.by_doi.get(doi.lower())
            if key:
                return key, self.by_key[key]
        if title:
            norm = normalize_title(title)
            key = self.by_title.get(norm)
            if key:
                return key, self.by_key[key]
        return None

    def build_key(self, metadata: Dict, default_title: str) -> str:
        if metadata.get("paperId"):
            return f"s2_{metadata['paperId']}"
        doi = metadata.get("doi")
        if doi:
            return f"doi_{slugify(doi)}"
        base = metadata.get("title") or default_title or "unknown"
        norm_title = slugify(base)
        year = metadata.get("year")
        suffix = f"_{year}" if year else ""
        raw = f"{norm_title}{suffix}"
        if raw not in self.by_key:
            return raw
        digest = sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"{raw}_{digest}"

    def save_metadata(self, paper_key: str, metadata: Dict) -> None:
        metadata = dict(metadata)
        metadata["paper_key"] = paper_key
        path = self.papers_dir / f"{paper_key}.json"
        path.write_text(json.dumps(
            metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.by_key[paper_key] = metadata
        doi = metadata.get("doi")
        if doi:
            self.by_doi[doi.lower()] = paper_key
        paper_id = metadata.get("paperId")
        if paper_id:
            self.by_paper_id[paper_id] = paper_key
        title = metadata.get("title")
        if title:
            self.by_title[normalize_title(title)] = paper_key

    def pdf_path(self, paper_key: str) -> Path:
        return self.pdf_dir / f"{paper_key}.pdf"

    def pdf_exists(self, paper_key: str) -> bool:
        return self.pdf_path(paper_key).exists()


def parse_citation(principle: str, citation: str, idx: int) -> CitationInfo:
    cleaned = citation.replace("**", "")
    cleaned = MARKDOWN_MARKERS_RE.sub("", cleaned).strip()

    cleaned = re.split(r"\s(?:#{2,}|###)\s", cleaned)[0].strip()

    while True:
        updated = TRAILING_NOTE_RE.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated.strip()

    doi_match = DOI_RE.search(cleaned)
    doi = doi_match.group(1).rstrip(".") if doi_match else None

    title = None
    year_match = re.search(r"\(\d{4}\)", cleaned)
    if year_match:
        after_year = cleaned[year_match.end():].lstrip(". ").strip()
        if after_year:
            title_candidate = re.split(
                r"\.\s+", after_year, maxsplit=1)[0].strip(" ,;:")
            if title_candidate and not title_candidate.lower().startswith("in "):
                title = title_candidate

    if not title:
        segments = [seg.strip()
                    for seg in re.split(r"\.\s+", cleaned) if seg.strip()]
        for seg in segments:
            if re.search(r"\(\d{4}\)", seg):
                continue
            if seg.lower().startswith("in "):
                continue
            title = seg.strip(" ,;:")
            break

    if title:
        title = re.sub(r"\s+", " ", title).strip()

    return CitationInfo(
        principle=principle,
        citation_raw=citation,
        index=idx,
        doi=doi,
        title_hint=title or None,
    )


def load_responses(principles: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> List[CitationInfo]:
    data = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    selected = set(principles) if principles else None
    tasks: List[CitationInfo] = []
    for principle, citations in data.items():
        if selected and principle not in selected:
            continue
        upper = limit if limit is not None else len(citations)
        for idx, citation in enumerate(citations[:upper], start=1):
            tasks.append(parse_citation(principle, citation, idx))
    return tasks


def semantic_scholar_request(
    url: str,
    params: Dict,
    session: requests.Session,
    retries: int = 3,
    backoff: float = 2.0,
) -> Optional[Dict]:
    api_key = os.getenv("S2_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(1, retries + 1):
        try:
            print(f"[request] attempt={attempt} URL={url} params={params}")
            resp = session.get(url, params=params, headers=headers, timeout=30)
            print(f"[response] status={resp.status_code} url={resp.url}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text
                except Exception:
                    body = "<unavailable>"
                print(
                    f"[error] status={exc.response.status_code} url={exc.response.url} body={body[:200]}")
            else:
                print(f"[error] request failed: {exc}")
            if attempt == retries:
                raise exc
            time.sleep(backoff * attempt)
    return None


def fetch_metadata(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    fields = os.getenv("S2_SEARCH_FIELDS", SEARCH_FIELDS_DEFAULT)
    base_url = get_s2_base_url()
    if info.doi:
        try:
            result = semantic_scholar_request(
                base_url + f"paper/DOI:{info.doi}",
                {"fields": fields},
                session,
            )
            if result:
                print(f"[info] DOI lookup succeeded for {info.doi}")
                return result
        except requests.RequestException:
            pass

    if info.title_hint:
        query_variants = [info.title_hint]
        precise = f'title:"{info.title_hint}"'
        if precise not in query_variants:
            query_variants.append(precise)
        for query in query_variants:
            try:
                result = semantic_scholar_request(
                    base_url + "paper/search",
                    {"query": query, "limit": 1, "fields": fields},
                    session,
                    retries=1,
                )
            except requests.RequestException:
                continue
            if result:
                data = result.get("data")
                if data:
                    return data[0]
                print(f"[info] Query '{query}' returned no matches.")
    return None


def download_pdf(url: str, path: Path, session: requests.Session) -> bool:
    try:
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                return False
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(path)
            return True
    except requests.RequestException:
        return False


def append_log(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def determine_status(
    paper_key: Optional[str],
    meta: Optional[Dict],
    pdf_saved: bool,
    pdf_available: bool,
    fetch_failed: bool,
    pdf_attempted: bool,
) -> str:
    if fetch_failed:
        return "fetch_failed"
    if paper_key is None:
        return "not_found"
    if pdf_saved:
        return "pdf_saved"
    if pdf_available:
        if pdf_attempted:
            return "pdf_download_failed"
        return "open_access_available"
    return "metadata_only" if meta else "not_found"


def update_principles_index(entries: Dict[str, List[Dict[str, Optional[str]]]]) -> None:
    index: Dict[str, List[Dict[str, Optional[str]]]] = {}
    if PRINCIPLES_INDEX_PATH.exists():
        try:
            index = json.loads(
                PRINCIPLES_INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
    index.update(entries)
    PRINCIPLES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRINCIPLES_INDEX_PATH.write_text(json.dumps(
        index, ensure_ascii=False, indent=2), encoding="utf-8")


def process_citations(principles: Optional[List[str]], limit: Optional[int], download_pdfs: bool) -> None:
    base_url = get_s2_base_url()
    print(f"[info] Using Semantic Scholar base URL: {base_url}")
    tasks = load_responses(principles, limit)
    if not tasks:
        print("No citations found for the specified principles.")
        return

    registry = PaperRegistry()
    session = requests.Session()

    principle_entries: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for task in tasks:
        principle_entries.setdefault(task.principle, [])

    for idx, info in enumerate(tasks, start=1):
        print(
            f"[{idx}/{len(tasks)}] Processing principle='{info.principle}' citation #{info.index}")
        existing = registry.get_by_identifiers(info.doi, info.title_hint)
        metadata: Optional[Dict] = None
        paper_key: Optional[str] = None
        pdf_saved = False
        pdf_available = False
        fetch_failed = False
        pdf_attempted = False

        if existing:
            paper_key, metadata = existing
        else:
            try:
                metadata = fetch_metadata(info, session)
            except requests.RequestException as exc:
                fetch_failed = True
                append_log(ERROR_LOG, {
                    "principle": info.principle,
                    "index": info.index,
                    "citation": info.citation_raw,
                    "error": str(exc),
                })
                metadata = None

            if metadata:
                paper_key = registry.build_key(
                    metadata, info.title_hint or info.citation_raw)
                registry.save_metadata(paper_key, metadata)
            else:
                paper_key = None

        if metadata:
            pdf_info = metadata.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url")
            pdf_available = bool(pdf_url)
            if pdf_available and download_pdfs:
                pdf_attempted = True
                pdf_path = registry.pdf_path(paper_key)
                if registry.pdf_exists(paper_key):
                    pdf_saved = True
                else:
                    pdf_saved = download_pdf(pdf_url, pdf_path, session)
                    if not pdf_saved:
                        append_log(ERROR_LOG, {
                            "principle": info.principle,
                            "paper_key": paper_key,
                            "citation": info.citation_raw,
                            "error": "Failed to download PDF",
                            "pdf_url": pdf_url,
                        })
        else:
            append_log(NOT_FOUND_LOG, {
                "principle": info.principle,
                "citation": info.citation_raw,
                "index": info.index,
                "doi": info.doi,
                "title_hint": info.title_hint,
            })

        status = determine_status(
            paper_key,
            metadata,
            pdf_saved,
            pdf_available,
            fetch_failed,
            pdf_attempted,
        )
        entry = {
            "paper_key": paper_key,
            "status": status,
        }
        principle_entries[info.principle].append(entry)
        time.sleep(0.2)

    update_principles_index(principle_entries)
    print(f"[done] Updated principles index at {PRINCIPLES_INDEX_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert responses_parsed.json into Semantic Scholar metadata and PDF archive."
    )
    parser.add_argument(
        "--principles",
        nargs="*",
        help="Optional list of principle names to process. Defaults to all in responses_parsed.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of citations per principle.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip downloading PDFs even if open access links are available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_pdfs = not args.skip_pdf
    process_citations(args.principles, args.limit, download_pdfs)
