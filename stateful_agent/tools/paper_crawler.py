import json
import os
import re
import shutil
import sqlite3
import tarfile
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import arxiv
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from hyperpocket.tool import function_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

# Constants
DEFAULT_DATA_DIR = os.getenv("DEFAULT_DATA_DIR")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")

# Create directories if they don't exist
Path(DEFAULT_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.join(DEFAULT_DATA_DIR, "recommendation")).mkdir(
    parents=True, exist_ok=True
)


@function_tool
def crawl_scholar_papers(lab_name: str):
    """
    Crawl Google Scholar pages for lab members and download their arXiv papers.
    Creates a collection for the lab and adds paper embeddings to the vector database using arXiv ID as the unique identifier.
    """
    lab_name = lab_name.lower()

    # Get lab info from database
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    # Get lab members and their scholar URLs
    cursor.execute(
        "SELECT member_name, scholar_url FROM lab_members WHERE lab_name = ?",
        (lab_name,),
    )
    members = cursor.fetchall()

    if not members:
        conn.close()
        return f"No members found for lab '{lab_name}'"

    # Create paper_tracking table if it doesn't exist (using arxiv_id as paper_id)
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS paper_tracking
             (lab_name text, paper_id text, title text, author text, date_added text, 
             pdf_path text, PRIMARY KEY (lab_name, paper_id))"""
    )

    # Create a collection for the lab if it doesn't exist
    collection_name = f"{lab_name}_papers"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )

    papers_added = 0
    errors = []

    # Lab data directory
    lab_data_dir = os.path.join(DEFAULT_DATA_DIR, lab_name)
    Path(lab_data_dir).mkdir(parents=True, exist_ok=True)

    for member_name, scholar_url in members:
        if not scholar_url:
            continue

        try:
            # Get the Google Scholar page
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Add a timeout to the request
            response = requests.get(scholar_url, headers=headers, timeout=15)
            response.raise_for_status()  # Raise an exception for bad status codes

            soup = BeautifulSoup(response.text, "html.parser")

            # Find paper entries
            paper_entries = soup.find_all("tr", class_="gsc_a_tr")

            for entry in paper_entries:
                paper_title = "Unknown Title"  # Default title
                try:
                    # Get paper title
                    title_element = entry.find("td", class_="gsc_a_t").find("a")
                    if not title_element:
                        continue
                    paper_title = title_element.text.strip()

                    # Get paper year
                    year_element = entry.find("td", class_="gsc_a_y")
                    year = year_element.text.strip() if year_element else "Unknown"

                    # Get paper link (to details page)
                    paper_link = title_element.get("href")
                    if not paper_link:
                        continue

                    paper_url = f"https://scholar.google.com{paper_link}"

                    # Wait to avoid being rate limited
                    time.sleep(
                        2
                    )  # Consider increasing sleep or using exponential backoff

                    # Get the paper details page
                    details_response = requests.get(
                        paper_url, headers=headers, timeout=10
                    )
                    details_response.raise_for_status()

                    details_soup = BeautifulSoup(details_response.text, "html.parser")

                    # Find arxiv link
                    links = details_soup.find_all("a")
                    arxiv_link = None

                    for link in links:
                        href = link.get("href", "")
                        if "arxiv.org" in href:
                            arxiv_link = href
                            break

                    if not arxiv_link:
                        continue  # Skip if no arxiv link found

                    # Extract arxiv ID from the link (handle different URL formats)
                    match = re.search(
                        r"arxiv\.org/(?:abs|pdf|ps)/([\w.-]+?)(?:v\d+)?(?:.pdf)?$",
                        arxiv_link,
                    )
                    if not match:
                        match = re.search(
                            r"arxiv\.org/ftp/arxiv/papers/(\d{4})/(\d{4}\.\d+)",
                            arxiv_link,
                        )  # Handle ftp links
                        if match:
                            arxiv_id = f"{match.group(1)}.{match.group(2)}"
                        else:
                            errors.append(
                                f"Could not extract arXiv ID from link: {arxiv_link} for paper '{paper_title}'"
                            )
                            continue
                    else:
                        arxiv_id = match.group(1)

                    # Use arxiv_id as the unique paper identifier
                    paper_id = arxiv_id

                    # Check if paper already exists in the database using arxiv_id
                    cursor.execute(
                        "SELECT * FROM paper_tracking WHERE lab_name = ? AND paper_id = ?",
                        (lab_name, paper_id),
                    )
                    existing_paper = cursor.fetchone()

                    if existing_paper:
                        continue  # Skip if paper already exists

                    # Download the paper using the arxiv API
                    client = arxiv.Client(
                        num_retries=5, delay_seconds=5
                    )  # Add retries and delay
                    search = arxiv.Search(id_list=[arxiv_id])

                    # Use a flag to ensure we only process the first valid result
                    processed_result = False
                    for result in client.results(search):
                        if processed_result:
                            continue  # Skip if already processed

                        # Use arxiv_id for filename, replacing slashes/dots if necessary
                        safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                        filename = f"{safe_arxiv_id}.pdf"
                        filepath = os.path.join(lab_data_dir, filename)

                        # Download the PDF
                        result.download_pdf(dirpath=lab_data_dir, filename=filename)

                        # Add to tracking database using arxiv_id
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        cursor.execute(
                            "INSERT INTO paper_tracking (lab_name, paper_id, title, author, date_added, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                lab_name,
                                paper_id,
                                result.title,
                                member_name,
                                current_date,
                                filepath,
                            ),  # Use title from arxiv result
                        )

                        # Load and add to vector database
                        loader = PyPDFLoader(filepath)
                        documents = loader.load()

                        # Generate UUIDs for each document chunk
                        uuids = [str(uuid4()) for _ in range(len(documents))]

                        # Add metadata including arxiv_id
                        for doc in documents:
                            doc.metadata["paper_id"] = paper_id  # This is arxiv_id
                            doc.metadata["title"] = result.title
                            doc.metadata["author"] = (
                                member_name  # The lab member who has this paper
                            )
                            doc.metadata["all_authors"] = ", ".join(
                                str(a) for a in result.authors
                            )  # Add all authors
                            doc.metadata["year"] = str(
                                result.published.year
                            )  # Use published year from arxiv
                            doc.metadata["lab"] = lab_name
                            doc.metadata["source"] = "scholar_crawl"

                        # Add to vector store
                        vector_store.add_documents(documents=documents, ids=uuids)

                        papers_added += 1
                        processed_result = True  # Mark as processed

                except requests.exceptions.RequestException as e:
                    errors.append(
                        f"Network error processing paper '{paper_title}' by {member_name}: {str(e)}"
                    )
                    time.sleep(5)  # Wait longer after a network error
                except arxiv.arxiv.UnexpectedEmptyPageError as e:
                    errors.append(
                        f"arXiv API error (empty page) for paper '{paper_title}' (ID: {arxiv_id}): {str(e)}"
                    )
                except arxiv.arxiv.ArxivError as e:
                    errors.append(
                        f"arXiv API error for paper '{paper_title}' (ID: {arxiv_id}): {str(e)}"
                    )
                except sqlite3.Error as e:
                    errors.append(
                        f"Database error for paper '{paper_title}' (ID: {arxiv_id}): {str(e)}"
                    )
                except Exception as e:
                    # Log the full traceback for unexpected errors
                    import traceback

                    tb_str = traceback.format_exc()
                    errors.append(
                        f"Unexpected error processing paper '{paper_title}' by {member_name} (ID: {arxiv_id}): {str(e)}\n{tb_str}"
                    )

        except requests.exceptions.RequestException as e:
            errors.append(
                f"Network error processing scholar page for {member_name}: {str(e)}"
            )
            time.sleep(10)  # Wait longer after a scholar page error
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            errors.append(
                f"Unexpected error processing scholar page for {member_name}: {str(e)}\n{tb_str}"
            )

    conn.commit()
    conn.close()

    result = f"Process completed for lab '{lab_name}'. Added {papers_added} new papers."
    if errors:
        # Log only a summary of errors if there are too many
        error_summary = (
            errors
            if len(errors) <= 5
            else errors[:5] + [f"... ({len(errors) - 5} more errors)"]
        )
        result += f"\nEncountered {len(errors)} errors. Error summary: {json.dumps(error_summary, indent=2)}"

    return result


@function_tool
def check_new_papers(lab_name: str):
    """
    Check if lab members have new papers that haven't been added to the collection yet.
    Returns a detailed report of the crawling results.
    """
    lab_name = lab_name.lower()
    result = crawl_scholar_papers(lab_name)
    # Add a prefix to make it clear this was from check_new_papers
    if isinstance(result, str):
        if "Process completed" in result:
            return f"Checked for new papers: {result}"
        else:
            return f"Failed to check for new papers: {result}"
    return result


@function_tool
def recommend_papers(lab_name: str, time_period_days: int, num_papers: int):
    """
    Recommend papers from arXiv related to the lab's research based on collection similarity.
    Papers are stored in data/recommendation directory and added to a recommendation collection using arXiv ID as the unique identifier.
    Avoids recommending papers already tracked in the main lab collection.

    Note: Installing the 'tzlocal' package is recommended for better timezone handling.
    """
    lab_name = lab_name.lower()

    # Get lab info from database
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    # Get lab research areas
    cursor.execute(
        "SELECT research_area FROM lab_research_areas WHERE lab_name = ?", (lab_name,)
    )
    research_areas = cursor.fetchall()

    if not research_areas:
        conn.close()
        return f"No research areas defined for lab '{lab_name}'"

    # Format research areas for arXiv query
    arxiv_categories = [area[0].lower() for area in research_areas]

    # Create arxiv query with appropriate categories
    category_mapping = {
        "artificial intelligence": "cs.AI",
        "machine learning": "cs.LG",
        "computer vision": "cs.CV",
        "natural language processing": "cs.CL",
        "robotics": "cs.RO",
        "human-computer interaction": "cs.HC",
        "computer graphics": "cs.GR",
        "cryptography": "cs.CR",
        "data mining": "cs.DM",
        "databases": "cs.DB",
        "distributed computing": "cs.DC",
        "information retrieval": "cs.IR",
        "neural networks": "cs.NE",
        "symbolic computation": "cs.SC",
        "software engineering": "cs.SE",
        "systems": "cs.SY",
        "computer science": "cs",
        "quantum computing": "quant-ph",
        "statistics": "stat",
        "mathematics": "math",
        "physics": "physics",
        "neuroscience": "q-bio.NC",
        "cognitive science": "cs.AI",
        "information theory": "cs.IT",
        "deep learning": "cs.LG",
        "reinforcement learning": "cs.LG",
        "multimodal learning": "cs.LG",
        "ai": "cs.AI",
        "nlp": "cs.CL",
        "cv": "cs.CV",
        "ml": "cs.LG",
    }

    arxiv_cats = []
    for area in arxiv_categories:
        area_lower = area.lower()
        mapped_cat = category_mapping.get(area_lower)
        if mapped_cat:
            arxiv_cats.append(mapped_cat)
        elif "." in area_lower:  # Assume it's already an arXiv category like cs.LG
            arxiv_cats.append(area_lower)
        # else: # Optionally, add a default or log a warning for unmapped areas
        #     print(f"Warning: Research area '{area}' not mapped to an arXiv category.")

    if not arxiv_cats:
        conn.close()
        return (
            f"Could not map any research areas {arxiv_categories} to arXiv categories."
        )

    arxiv_cats = list(set(arxiv_cats))
    arxiv_query = " OR ".join(
        [f"cat:{cat}" for cat in arxiv_cats]
    )  # Use OR for broader search across categories

    # Calculate date range for papers
    # Ensure end_date is timezone-aware
    try:
        import tzlocal

        local_tz = tzlocal.get_localzone()
        end_date = datetime.now(local_tz)
        start_date = end_date - timedelta(days=time_period_days)
    except ImportError:
        # If tzlocal is not available, use UTC
        from datetime import timezone

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=time_period_days)
        print("Warning: tzlocal not installed. Using UTC timezone for comparison.")

    # Find papers from arXiv
    client = arxiv.Client(num_retries=5, delay_seconds=5)
    search = arxiv.Search(
        query=arxiv_query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=200,  # Get more initially for filtering and scoring
    )

    papers = []
    arxiv_ids_found = set()

    try:
        for result in client.results(search):
            # Check if paper is within date range
            # Use updated_date if available, otherwise published_date
            paper_date = result.updated if result.updated else result.published
            # Ensure paper_date is timezone-aware for comparison
            if paper_date.tzinfo is None:
                # Attempt to make timezone-aware using local system timezone if possible
                try:
                    import tzlocal

                    local_tz = tzlocal.get_localzone()
                    paper_date = local_tz.localize(paper_date)
                except ImportError:
                    # Fallback: Assume same timezone as start_date (less accurate)
                    paper_date = paper_date.replace(tzinfo=start_date.tzinfo)

            if start_date <= paper_date <= end_date:
                arxiv_id = result.get_short_id()
                if (
                    arxiv_id not in arxiv_ids_found
                ):  # Avoid duplicates from API pagination/updates
                    papers.append(result)
                    arxiv_ids_found.add(arxiv_id)

            # Stop if we have enough candidates or exceed a reasonable limit
            if len(papers) >= 150:
                break
    except arxiv.arxiv.ArxivError as e:
        conn.close()
        return f"Error searching arXiv: {str(e)}"
    except ImportError:
        # Handle case where tzlocal is not installed
        print(
            "Warning: tzlocal not installed. Timezone comparison might be less accurate."
        )
        # Continue without tzlocal if possible
        pass

    if not papers:
        conn.close()
        return f"No papers found on arXiv for query '{arxiv_query}' in the specified time period."

    # Get lab papers collection
    collection_name = f"{lab_name}_papers"
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        # Check if the collection actually has embeddings
        lab_docs = vector_store.get(include=["embeddings"])
        if not lab_docs or not lab_docs.get("ids") or not lab_docs.get("embeddings"):
            conn.close()
            # Be more informative if the collection exists but is empty
            try:
                vector_store.peek()  # Check if collection exists
                return f"Lab collection '{collection_name}' exists but has no papers. Crawl papers first."
            except:
                return f"Lab collection '{collection_name}' not found or is empty. Create it and crawl papers first."

        lab_embeddings = lab_docs["embeddings"]

    except Exception as e:
        conn.close()
        # Provide more specific feedback if the collection doesn't exist yet
        return f"Error accessing lab collection '{collection_name}': {str(e)}. Ensure the lab exists and papers have been crawled."

    # Create recommendation tracking table if it doesn't exist (using arxiv_id as paper_id)
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS recommendation_tracking
             (lab_name text, paper_id text, title text, score float, date_added text, 
             pdf_path text, PRIMARY KEY (lab_name, paper_id))"""
    )

    # Create recommendation collection if it doesn't exist
    recommendation_collection_name = f"recommendation_for_{lab_name}"
    recommendation_store = Chroma(
        collection_name=recommendation_collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )

    # Paper abstracts to calculate similarity
    abstracts = [paper.summary for paper in papers]
    if not abstracts:
        conn.close()
        return "No abstracts found for the retrieved arXiv papers."

    # Calculate embeddings for candidate papers
    try:
        paper_embeddings = embeddings.embed_documents(abstracts)
    except Exception as e:
        conn.close()
        return f"Error generating embeddings for candidate papers: {str(e)}"

    # Calculate similarity scores (cosine similarity)
    similarities = []
    np_lab_embeddings = np.array(lab_embeddings)
    # Handle case where lab_embeddings might be empty or malformed
    if np_lab_embeddings.ndim != 2 or np_lab_embeddings.shape[0] == 0:
        conn.close()
        return f"Lab collection '{collection_name}' embeddings are invalid or empty."

    lab_norms = np.linalg.norm(np_lab_embeddings, axis=1)
    # Prevent division by zero if norms are zero
    lab_norms[lab_norms == 0] = 1e-9

    for paper_embedding in paper_embeddings:
        np_paper_embedding = np.array(paper_embedding)
        paper_norm = np.linalg.norm(np_paper_embedding)

        if paper_norm == 0:
            # Handle zero vector for candidate paper
            max_similarity = 0.0
        else:
            # Calculate cosine similarity efficiently
            sim_vector = np.dot(np_lab_embeddings, np_paper_embedding) / (
                lab_norms * paper_norm
            )
            # Use max similarity instead of average for better relevance signal
            max_similarity = np.max(sim_vector) if sim_vector.size > 0 else 0.0
        similarities.append(max_similarity)

    if not similarities:
        # This case should ideally not happen if abstracts exist and embeddings worked
        conn.close()
        return "Could not calculate similarity scores."

    # Sort papers by similarity score
    scored_papers = sorted(zip(papers, similarities), key=lambda x: x[1], reverse=True)

    # Create recommendation directory if it doesn't exist
    recommendation_dir = os.path.join(DEFAULT_DATA_DIR, "recommendation", lab_name)
    Path(recommendation_dir).mkdir(parents=True, exist_ok=True)

    # Download and add top N papers
    papers_added = 0
    top_papers_result = []
    errors = []
    skipped_papers = 0  # Track papers we had to skip

    # Get existing recommended paper IDs to avoid duplicates
    cursor.execute(
        "SELECT paper_id FROM recommendation_tracking WHERE lab_name = ?", (lab_name,)
    )
    existing_rec_ids = {row[0] for row in cursor.fetchall()}

    # Get existing main lab paper IDs to avoid recommending them
    cursor.execute(
        "SELECT paper_id FROM paper_tracking WHERE lab_name = ?", (lab_name,)
    )
    existing_lab_paper_ids = {row[0] for row in cursor.fetchall()}

    for paper, score in scored_papers:
        if papers_added >= num_papers:
            break  # Stop once we have enough recommendations

        paper_id = paper.get_short_id()  # Use arxiv_id

        # Skip if already recommended OR if it's already in the main lab collection
        if paper_id in existing_rec_ids or paper_id in existing_lab_paper_ids:
            skipped_papers += 1
            continue

        # Use arxiv_id for filename
        safe_arxiv_id = paper_id.replace("/", "_").replace(".", "_")
        filename = f"{safe_arxiv_id}.pdf"
        filepath = os.path.join(recommendation_dir, filename)

        try:
            # Download the PDF
            paper.download_pdf(dirpath=recommendation_dir, filename=filename)

            # Add to tracking database
            current_date = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "INSERT INTO recommendation_tracking (lab_name, paper_id, title, score, date_added, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                (lab_name, paper_id, paper.title, float(score), current_date, filepath),
            )

            # Load and add to recommendation vector store
            loader = PyPDFLoader(filepath)
            documents = loader.load()

            uuids = [str(uuid4()) for _ in range(len(documents))]

            # Add metadata
            for doc in documents:
                doc.metadata["paper_id"] = paper_id  # arxiv_id
                doc.metadata["title"] = paper.title
                doc.metadata["authors"] = ", ".join(
                    str(author) for author in paper.authors
                )
                doc.metadata["score"] = float(score)
                doc.metadata["lab"] = lab_name
                doc.metadata["published_date"] = paper.published.strftime("%Y-%m-%d")
                doc.metadata["source"] = "recommendation"

            # Add to vector store
            recommendation_store.add_documents(documents=documents, ids=uuids)

            # Add to result list
            top_papers_result.append(
                {
                    "arxiv_id": paper_id,
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "score": float(score),
                    "published_date": paper.published.strftime("%Y-%m-%d"),
                    "abstract": (
                        paper.summary[:300] + "..."
                        if len(paper.summary) > 300
                        else paper.summary
                    ),
                }
            )

            papers_added += 1

        except arxiv.arxiv.ArxivError as e:
            errors.append(
                f"arXiv error downloading/processing recommended paper {paper_id}: {str(e)}"
            )
            # Clean up potentially corrupted file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as rm_err:
                    print(f"Error removing corrupted file {filepath}: {rm_err}")
        except sqlite3.Error as e:
            errors.append(
                f"Database error adding recommended paper {paper_id}: {str(e)}"
            )
            if os.path.exists(filepath):  # Clean up downloaded file if DB fails
                try:
                    os.remove(filepath)
                except OSError as rm_err:
                    print(f"Error removing file {filepath} after DB error: {rm_err}")
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            errors.append(
                f"Unexpected error processing recommended paper {paper_id}: {str(e)}\n{tb_str}"
            )
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as rm_err:
                    print(
                        f"Error removing file {filepath} after unexpected error: {rm_err}"
                    )

    conn.commit()
    conn.close()

    # Create a consistent JSON response regardless of whether papers were added
    result = {
        "lab_name": lab_name,
        "papers_added": papers_added,
        "skipped_papers": skipped_papers,
        "recommendations": top_papers_result,
        "status": "success",
    }

    # Add appropriate message based on the outcome
    if not top_papers_result:
        if scored_papers and skipped_papers > 0:
            result["message"] = (
                f"Found {len(scored_papers)} potential papers, but all {skipped_papers} were already recommended or part of the lab collection."
            )
            result["status"] = "no_new_recommendations"
        elif not scored_papers:
            result["message"] = "No relevant papers found that match the criteria."
            result["status"] = "no_matches_found"
        else:
            result["message"] = (
                f"Found potential papers, but failed to add any recommendations. Check errors."
            )
            result["status"] = "error"

    if errors:
        result["errors"] = errors[:5]  # Include a summary of errors

    return json.dumps(result, indent=2)


@function_tool
def generate_paper_summary(
    lab_name: str,
    author_name: str,
    paper_title: str = "",
    arxiv_id_in: str = "",  # Allow specifying arxiv_id directly
    is_latest: bool = False,
    related_papers_count: int = 10,
):
    """
    Generate a comprehensive summary for a paper, considering related papers from the lab's collections.
    Lookup priority: arxiv_id_in > paper_title > is_latest by author_name.
    Uses arXiv ID as the unique identifier internally.
    Uses the complete paper content for the target paper being summarized.
    Uses LaTeX source files (if available) to extract semantic sections from related papers for better context.
    """
    lab_name = lab_name.lower()

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    # Find the target paper's info (arxiv_id, title, pdf_path)
    paper_info = None
    target_arxiv_id = None

    if arxiv_id_in:
        # Lookup by provided arxiv_id
        cursor.execute(
            "SELECT paper_id, title, pdf_path, author FROM paper_tracking WHERE lab_name = ? AND paper_id = ?",
            (lab_name, arxiv_id_in),
        )
        paper_info = cursor.fetchone()
        if paper_info:
            target_arxiv_id = paper_info[0]
        else:
            # Maybe it's in recommendations?
            cursor.execute(
                "SELECT paper_id, title, pdf_path FROM recommendation_tracking WHERE lab_name = ? AND paper_id = ?",
                (lab_name, arxiv_id_in),
            )
            rec_info = cursor.fetchone()
            if rec_info:
                paper_info = (
                    rec_info[0],
                    rec_info[1],
                    rec_info[2],
                    "N/A - Recommended",
                )  # Author not stored here
                target_arxiv_id = rec_info[0]

    elif paper_title:
        # Search by paper title (less reliable) - find the best match's arxiv_id
        # Use `author_name` to narrow down if provided
        query = "SELECT paper_id, title, pdf_path, author FROM paper_tracking WHERE lab_name = ? AND title LIKE ?"
        params = [lab_name, f"%{paper_title}%"]
        if author_name:
            query += " AND author = ?"
            params.append(author_name)
        query += (
            " ORDER BY date_added DESC LIMIT 1"  # Prioritize recent if multiple matches
        )

        cursor.execute(query, tuple(params))
        paper_info = cursor.fetchone()
        if paper_info:
            target_arxiv_id = paper_info[0]
            # Verify author if not used in query
            if not author_name:
                author_name = paper_info[3]

    elif is_latest and author_name:
        # Get the latest paper by author
        cursor.execute(
            "SELECT paper_id, title, pdf_path, author FROM paper_tracking WHERE lab_name = ? AND author = ? ORDER BY date_added DESC LIMIT 1",
            (lab_name, author_name),
        )
        paper_info = cursor.fetchone()
        if paper_info:
            target_arxiv_id = paper_info[0]

    else:
        conn.close()
        return "Please provide either an arxiv_id, a paper title (optionally with author), or set is_latest=True with an author_name."

    if not paper_info or not target_arxiv_id:
        search_criteria = (
            f"arxiv_id '{arxiv_id_in}'"
            if arxiv_id_in
            else (
                f"title '{paper_title}'"
                if paper_title
                else f"latest by '{author_name}'"
            )
        )
        conn.close()
        return f"No paper found matching criteria ({search_criteria}) in lab '{lab_name}' tracked papers or recommendations."

    # We found the paper, extract details
    title, pdf_path = paper_info[1], paper_info[2]
    # Author might be N/A if found in recommendations only
    found_author = paper_info[3] if len(paper_info) > 3 else "N/A"

    # Access lab paper collection
    collection_name = f"{lab_name}_papers"
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        # Verify collection exists
        vector_store.peek()
    except Exception as e:
        # Don't fail completely, maybe only recommendations exist
        vector_store = None
        print(f"Warning: Could not access lab collection '{collection_name}': {str(e)}")

    # Access recommendation collection
    recommendation_collection_name = f"recommendation_for_{lab_name}"
    try:
        recommendation_store = Chroma(
            collection_name=recommendation_collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        # Verify collection exists
        recommendation_store.peek()
    except Exception as e:
        recommendation_store = None
        print(
            f"Warning: Could not access recommendation collection '{recommendation_collection_name}': {str(e)}"
        )

    if not vector_store and not recommendation_store:
        conn.close()
        return f"Neither lab paper collection nor recommendation collection found for lab '{lab_name}'."

    # Load target paper content and attempt to get LaTeX source
    try:
        if not os.path.exists(pdf_path):
            conn.close()
            return f"Error: PDF file not found at path: {pdf_path}"

        # Get the ArXiv API client
        client = arxiv.Client(num_retries=5, delay_seconds=5)

        # First, load the PDF for embedding
        loader = PyPDFLoader(pdf_path)
        target_paper_docs = loader.load()

        # Also get abstract from arXiv API if possible
        abstract = ""
        try:
            # Get paper from ArXiv API
            search = arxiv.Search(id_list=[target_arxiv_id])
            paper = next(client.results(search), None)

            if paper:
                abstract = paper.summary
        except Exception as e:
            print(f"Error getting abstract from arXiv API: {e}")

        # Combine all page content for searching related papers AND for summarization
        full_paper_text = "\n\n".join([doc.page_content for doc in target_paper_docs])

        # Use the entire paper for similarity search
        search_text = full_paper_text

        # Use paper content to find related papers
        lab_related = []
        if vector_store:
            try:
                lab_related = vector_store.similarity_search(
                    search_text, k=related_papers_count
                )
            except Exception as e:
                print(f"Error searching lab collection: {e}")

        recommendation_related = []
        if recommendation_store:
            try:
                recommendation_related = recommendation_store.similarity_search(
                    search_text, k=related_papers_count
                )
            except Exception as e:
                print(f"Error searching recommendation collection: {e}")

        all_related = lab_related + recommendation_related

        # Extract titles and brief content from related papers, filtering out the target paper
        related_info = []
        processed_related_ids = set()

        # Try to get related paper sections from their LaTeX sources
        for doc in all_related:
            related_arxiv_id = doc.metadata.get("paper_id")  # paper_id is arxiv_id

            if (
                related_arxiv_id
                and related_arxiv_id != target_arxiv_id
                and related_arxiv_id not in processed_related_ids
            ):
                related_title = doc.metadata.get("title", "Unknown title")
                related_data = {
                    "title": related_title,
                    "arxiv_id": related_arxiv_id,
                    "content_sample": doc.page_content[:250] + "...",  # Default content
                }

                # Try to get semantic sections for this related paper
                try:
                    # Fetch from ArXiv API
                    rel_search = arxiv.Search(id_list=[related_arxiv_id])
                    rel_paper = next(client.results(rel_search), None)

                    if rel_paper:
                        # We have the related paper - try to get sections from LaTeX
                        rel_intro = ""
                        rel_concl = ""

                        with TemporaryDirectory() as rel_tmpdir:
                            try:
                                rel_source_path = rel_paper.download_source(
                                    dirpath=rel_tmpdir
                                )

                                if rel_source_path and rel_source_path.endswith(
                                    ".tar.gz"
                                ):
                                    with tarfile.open(rel_source_path) as rel_tar:
                                        rel_tex_files = [
                                            f
                                            for f in rel_tar.getnames()
                                            if f.endswith(".tex")
                                        ]

                                        if rel_tex_files:
                                            # Look for main tex file or combine all
                                            rel_content = ""
                                            for rel_tex in rel_tex_files:
                                                try:
                                                    f = rel_tar.extractfile(rel_tex)
                                                    if f:
                                                        content = f.read().decode(
                                                            "utf-8", errors="ignore"
                                                        )
                                                        rel_content += content + "\n\n"
                                                except Exception as e:
                                                    print(
                                                        f"Error reading related tex file {rel_tex}: {e}"
                                                    )

                                            # Clean content
                                            rel_content = re.sub(
                                                r"%.*\n", "\n", rel_content
                                            )

                                            # Extract introduction and conclusion sections
                                            rel_intro_match = re.search(
                                                r"\\section\{(?:Introduction|INTRODUCTION|introduction)\}(.*?)(?:\\section|\\end\{document\}|\\bibliography|\\appendix)",
                                                rel_content,
                                                re.DOTALL,
                                            )
                                            if rel_intro_match:
                                                rel_intro = rel_intro_match.group(
                                                    1
                                                ).strip()
                                                # Limit the size of introduction text
                                                if len(rel_intro) > 500:
                                                    rel_intro = rel_intro[:500] + "..."

                                            rel_concl_match = re.search(
                                                r"\\section\{(?:Conclusion|CONCLUSION|conclusion|Conclusions|CONCLUSIONS|conclusions)\}(.*?)(?:\\section|\\end\{document\}|\\bibliography|\\appendix)",
                                                rel_content,
                                                re.DOTALL,
                                            )
                                            if rel_concl_match:
                                                rel_concl = rel_concl_match.group(
                                                    1
                                                ).strip()
                                                # Limit the size
                                                if len(rel_concl) > 500:
                                                    rel_concl = rel_concl[:500] + "..."

                            except Exception as e:
                                print(
                                    f"Error processing LaTeX for related paper {related_arxiv_id}: {e}"
                                )

                        # If we got sections, use them
                        if rel_intro or rel_concl:
                            section_text = ""
                            if rel_intro:
                                section_text += "Intro: " + rel_intro + "\n"
                            if rel_concl:
                                section_text += "Concl: " + rel_concl

                            # Update the related paper data with sections if we found them
                            if section_text:
                                related_data["content_sample"] = (
                                    section_text[:500] + "..."
                                )

                except Exception as e:
                    print(
                        f"Error getting data for related paper {related_arxiv_id}: {e}"
                    )

                # Add the related paper to our list
                related_info.append(related_data)
                processed_related_ids.add(related_arxiv_id)

        # Prepare the main content for the summarization prompt
        # For the main paper, we'll use the full text to provide complete information
        # If the abstract is available, add it at the beginning for context
        if abstract:
            prompt_main_content = f"Abstract:\n{abstract}\n\n"
            prompt_main_content += f"Full Paper Content:\n{full_paper_text}"
        else:
            prompt_main_content = f"Full Paper Content:\n{full_paper_text}"

        # Check if the content is too long - if so, we need to truncate intelligently
        # A typical approach is to keep beginning, important middle parts, and end
        max_content_length = 25000  # Set a reasonable limit to avoid token overflow
        if len(prompt_main_content) > max_content_length:
            first_part = prompt_main_content[: max_content_length // 3]
            middle_index = len(prompt_main_content) // 2
            middle_part = prompt_main_content[
                middle_index
                - (max_content_length // 6) : middle_index
                + (max_content_length // 6)
            ]
            last_part = prompt_main_content[-max_content_length // 3 :]

            prompt_main_content = f"{first_part}\n\n[...content truncated for length...]\n\n{middle_part}\n\n[...content truncated for length...]\n\n{last_part}"

        # Format summary input with related papers context
        summary_input = f"""
Target Paper: {title} (arXiv:{target_arxiv_id}) by {found_author}

Main Content:
{prompt_main_content}

---
Related Papers Context (Top {related_papers_count}):
"""

        if related_info:
            for i, related in enumerate(related_info[:related_papers_count]):
                summary_input += f"{i+1}. {related['title']} (arXiv:{related['arxiv_id']})\n   Sample: {related['content_sample']}\n\n"
        else:
            summary_input += "No relevant related papers found in the collections.\n"

        # Use OpenAI to generate summary
        from langchain_openai import ChatOpenAI

        # Ensure API key is loaded correctly
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            conn.close()
            return "Error: OPENAI_API_KEY environment variable not set."

        llm = ChatOpenAI(
            model="gpt-4o", api_key=api_key, temperature=0.3
        )  # Lower temperature for factual summary

        prompt = f"""
You are a scientific research assistant AI. Your task is to create a comprehensive, factual, and objective summary of a target academic paper.
Place the paper in the context of the provided related research snippets.

**Instructions:**
1.  **Target Paper Focus:** Primarily summarize the target paper: its core problem, methods, key findings, and contributions.
2.  **Contextualize:** Briefly explain how the target paper relates to the provided related papers (e.g., builds upon, contrasts with, addresses similar problems). Use the provided snippets for context, but do *not* summarize the related papers themselves in detail.
3.  **Structure:** Organize the summary logically (e.g., Introduction/Problem, Methods, Results, Relation to Context, Conclusion/Significance).
4.  **Tone:** Maintain an academic, neutral, and objective tone. Avoid speculation or personal opinions.
5.  **Length:** Aim for approximately 400-800 words.

**Paper Information & Context:**
{summary_input}

---
**Generate the summary now:**
"""

        # Estimate token count and potentially truncate prompt if needed (basic example)
        # Note: This is a rough estimate, actual tokenization varies.
        estimated_tokens = len(prompt.split()) * 1.3
        # print(f"Estimated prompt tokens: {estimated_tokens}")
        if estimated_tokens > 100000:
            conn.close()
            # Implement smarter truncation by reducing the paper content while preserving structure
            max_summary_input_length = 50000
            if len(summary_input) > max_summary_input_length:
                # Keep abstract and beginning intact
                parts = summary_input.split("Main Content:")
                header = parts[0] + "Main Content:\n"

                # Extract abstract if it exists
                abstract_section = ""
                if "Abstract:" in parts[1]:
                    abstract_end = parts[1].find("\n\nFull Paper Content:")
                    if abstract_end > 0:
                        abstract_section = parts[1][
                            : abstract_end + 2
                        ]  # Keep the abstract
                        main_content = parts[1][abstract_end + 2 :]
                    else:
                        main_content = parts[1]
                else:
                    main_content = parts[1]

                # Find the related papers section
                related_start = main_content.find("---\nRelated Papers Context")
                if related_start > 0:
                    paper_content = main_content[:related_start]
                    related_section = main_content[related_start:]
                else:
                    paper_content = main_content
                    related_section = ""

                # Truncate the paper content while preserving structure
                max_paper_length = (
                    max_summary_input_length
                    - len(header)
                    - len(abstract_section)
                    - len(related_section)
                    - 100
                )
                if len(paper_content) > max_paper_length:
                    first_part = paper_content[: max_paper_length // 3]
                    last_part = paper_content[-max_paper_length // 3 :]
                    truncated_content = f"{first_part}\n\n[...content truncated for length...]\n\n{last_part}"
                else:
                    truncated_content = paper_content

                # Rebuild the summary input
                summary_input = (
                    header + abstract_section + truncated_content + related_section
                )

                # Update the prompt
                prompt = f"""
You are a scientific research assistant AI. Your task is to create a comprehensive, factual, and objective summary of a target academic paper.
Place the paper in the context of the provided related research snippets.

**Instructions:**
1.  **Target Paper Focus:** Primarily summarize the target paper: its core problem, methods, key findings, and contributions.
2.  **Contextualize:** Briefly explain how the target paper relates to the provided related papers (e.g., builds upon, contrasts with, addresses similar problems). Use the provided snippets for context, but do *not* summarize the related papers themselves in detail.
3.  **Structure:** Organize the summary logically (e.g., Introduction/Problem, Methods, Results, Relation to Context, Conclusion/Significance).
4.  **Tone:** Maintain an academic, neutral, and objective tone. Avoid speculation or personal opinions.
5.  **Length:** Aim for approximately 400-600 words.
6.  **Note:** The paper content has been partially truncated for length. Focus on the available information to create the best summary possible.

**Paper Information & Context:**
{summary_input}

---
**Generate the summary now:**
"""

        try:
            response = llm.invoke(prompt)
            summary = response.content
        except Exception as e:
            conn.close()
            return f"Error calling OpenAI API for summarization: {str(e)}"

        conn.close()
        return summary

    except FileNotFoundError:
        conn.close()
        return f"Error: PDF file not found at {pdf_path}"
    except Exception as e:
        conn.close()
        import traceback

        tb_str = traceback.format_exc()
        return f"Error generating paper summary for arXiv:{target_arxiv_id}: {str(e)}\n{tb_str}"
