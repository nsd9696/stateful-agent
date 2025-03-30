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
    
    # Implement crawl_scholar_papers functionality directly here
    # Get lab info from database
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Checked for new papers: Lab '{lab_name}' does not exist"

    # Get lab members and their scholar URLs
    cursor.execute(
        "SELECT member_name, scholar_url FROM lab_members WHERE lab_name = ?",
        (lab_name,),
    )
    members = cursor.fetchall()

    if not members:
        conn.close()
        return f"Checked for new papers: No members found for lab '{lab_name}'"

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

    return f"Checked for new papers: {result}"


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
        # Try to access the collection without using peek()
        try:
            # Just try to get one document to verify collection exists
            vector_store.get(include=["documents"], limit=1)
        except Exception as e:
            vector_store = None
            print(f"Warning: Could not access lab collection '{collection_name}': {str(e)}")
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
        # Try to access the collection without using peek()
        try:
            # Just try to get one document to verify collection exists
            recommendation_store.get(include=["documents"], limit=1)
        except Exception as e:
            recommendation_store = None
            print(
                f"Warning: Could not access recommendation collection '{recommendation_collection_name}': {str(e)}"
            )
    except Exception as e:
        recommendation_store = None
        print(
            f"Warning: Could not access recommendation collection '{recommendation_collection_name}': {str(e)}"
        )

    if not vector_store and not recommendation_store:
        conn.close()
        return f"Neither lab paper collection nor recommendation collection found for lab '{lab_name}'."

    # Create directory for recommendation files if it doesn't exist
    recommendation_dir = os.path.join(DEFAULT_DATA_DIR, "recommendation", lab_name)
    Path(recommendation_dir).mkdir(parents=True, exist_ok=True)

    # Process and score the papers - for now just based on recency and matching categories
    # In the future could add more sophisticated relevance scoring
    scored_papers = []
    
    # Check if the papers already exist in the database to avoid duplicates
    existing_paper_ids = set()
    cursor.execute(
        "SELECT paper_id FROM paper_tracking WHERE lab_name = ?",
        (lab_name,),
    )
    for row in cursor.fetchall():
        existing_paper_ids.add(row[0])

    # Score the papers that don't already exist
    for paper in papers:
        arxiv_id = paper.get_short_id()
        if arxiv_id in existing_paper_ids:
            continue  # Skip papers already in the database
            
        # Basic scoring based on recency
        date_score = 1.0  # All papers are within the requested time range
        
        # Calculate overall score (can be expanded with more factors)
        score = date_score
        
        scored_papers.append((paper, score))
    
    # Sort by score (highest first)
    scored_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N papers as requested
    recommended_papers = scored_papers[:num_papers]
    
    if not recommended_papers:
        conn.close()
        return f"No new papers found matching the research areas of lab '{lab_name}' in the last {time_period_days} days."
    
    # Download and add the recommended papers to the recommendation collection
    papers_added = []
    
    for paper, score in recommended_papers:
        arxiv_id = paper.get_short_id()
        
        try:
            # Use arxiv_id for filename
            safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
            filename = f"{safe_arxiv_id}.pdf"
            filepath = os.path.join(recommendation_dir, filename)
            
            # Download the PDF if it doesn't exist already
            if not os.path.exists(filepath):
                paper.download_pdf(dirpath=recommendation_dir, filename=filename)
            
            # Add to tracking database
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Insert with 'recommendation' as author to distinguish from lab member papers
            cursor.execute(
                "INSERT OR IGNORE INTO paper_tracking (lab_name, paper_id, title, author, date_added, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                (lab_name, arxiv_id, paper.title, "recommendation", current_date, filepath),
            )
            
            # Add to vector database if recommendation store exists
            if recommendation_store:
                # Load PDF
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                
                # Generate UUIDs
                uuids = [str(uuid4()) for _ in range(len(documents))]
                
                # Add metadata
                for doc in documents:
                    doc.metadata["paper_id"] = arxiv_id
                    doc.metadata["title"] = paper.title
                    doc.metadata["author"] = "recommendation"
                    doc.metadata["all_authors"] = ", ".join(str(a) for a in paper.authors)
                    doc.metadata["year"] = str(paper.published.year)
                    doc.metadata["lab"] = lab_name
                    doc.metadata["source"] = "recommendation"
                
                # Add to vector store
                recommendation_store.add_documents(documents=documents, ids=uuids)
            
            # Add to result list
            paper_info = {
                "title": paper.title,
                "authors": ", ".join(str(a) for a in paper.authors),
                "arxiv_id": arxiv_id,
                "abstract": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "score": score
            }
            papers_added.append(paper_info)
            
        except Exception as e:
            print(f"Error processing recommended paper {arxiv_id}: {e}")
    
    conn.commit()
    conn.close()
    
    # Format the recommendation response
    if not papers_added:
        return f"No new papers could be added for lab '{lab_name}'. All matching papers are already in the database or couldn't be processed."
    
    result = f"Recommended papers for lab '{lab_name}' (from the last {time_period_days} days):\n\n"
    
    for i, paper in enumerate(papers_added):
        result += f"{i+1}. {paper['title']}\n"
        result += f"   Authors: {paper['authors']}\n"
        result += f"   arXiv: {paper['arxiv_id']} - {paper['url']}\n"
        result += f"   Abstract: {paper['abstract']}\n\n"
    
    return result


@function_tool
def crawl_semantic_scholar(lab_name: str):
    """
    Crawl Semantic Scholar API for lab members and download their arXiv papers.
    Creates a collection for the lab and adds paper embeddings to the vector database.
    
    This function is an alternative to crawl_scholar_papers that uses the official Semantic Scholar API
    instead of scraping Google Scholar, avoiding rate limits and CAPTCHA issues.
    """
    lab_name = lab_name.lower()

    # Not directly calling crawl_semantic_scholar, but using the same implementation
    # Retrieve lab information
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    # Get lab members - we'll need to work with their names instead of Scholar URLs
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
    
    # Base URL for Semantic Scholar API
    semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Optional - get your own API key at https://www.semanticscholar.org/product/api
    # If not provided, will be limited to 100 requests/5min without key
    semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    
    # Headers for API requests
    headers = {
        "Accept": "application/json",
    }
    if semantic_scholar_api_key:
        headers["x-api-key"] = semantic_scholar_api_key
    
    for member_name, _ in members:
        # Use member name to search for author
        print(f"Searching for papers by {member_name} on Semantic Scholar...")
        
        try:
            # First, find the author by name
            author_search_url = f"{semantic_scholar_base_url}/author/search"
            author_params = {
                "query": member_name,
                "fields": "authorId,name,affiliations,paperCount",
                "limit": 3  # Get top 3 matches
            }
            
            author_response = requests.get(
                author_search_url, 
                headers=headers, 
                params=author_params,
                timeout=30
            )
            author_response.raise_for_status()
            author_data = author_response.json()
            
            # Find the most likely author match
            if not author_data.get('data') or len(author_data['data']) == 0:
                errors.append(f"Could not find author {member_name} on Semantic Scholar")
                continue
                
            # Find the best match - could refine this logic based on affiliations if needed
            author = author_data['data'][0]
            author_id = author['authorId']
            
            # Wait to avoid hitting rate limits
            time.sleep(2)
            
            # Get papers by this author
            papers_url = f"{semantic_scholar_base_url}/author/{author_id}/papers"
            papers_params = {
                "fields": "paperId,externalIds,title,abstract,year,authors,url,venue,publicationDate",
                "limit": 100
            }
            
            papers_response = requests.get(
                papers_url, 
                headers=headers, 
                params=papers_params,
                timeout=30
            )
            papers_response.raise_for_status()
            papers_data = papers_response.json()
            
            if not papers_data.get('data'):
                errors.append(f"No papers found for author {member_name} (ID: {author_id})")
                continue
                
            # Process each paper
            print(f"Found {len(papers_data['data'])} papers for {member_name}")
            for i, paper in enumerate(papers_data['data']):
                # Get the arXiv ID if available
                arxiv_id = None
                if paper.get('externalIds') and paper['externalIds'].get('ArXiv'):
                    arxiv_id = paper['externalIds']['ArXiv']
                
                # Skip papers without arXiv ID
                if not arxiv_id:
                    continue
                
                paper_title = paper.get('title', 'Unknown Title')
                print(f"Processing paper: {paper_title} (arXiv:{arxiv_id})")
                
                # Use arxiv_id as the unique paper identifier
                paper_id = arxiv_id
                
                # Check if paper already exists in the database
                cursor.execute(
                    "SELECT * FROM paper_tracking WHERE lab_name = ? AND paper_id = ?",
                    (lab_name, paper_id),
                )
                existing_paper = cursor.fetchone()
                
                if existing_paper:
                    print(f"Paper with arXiv ID {arxiv_id} already exists in the database. Skipping.")
                    continue
                
                # Throttle requests to avoid hitting rate limits
                if i > 0 and i % 5 == 0:
                    print("Pausing briefly to avoid rate limits...")
                    time.sleep(5)
                
                try:
                    # Download the paper using the arxiv API
                    client = arxiv.Client(num_retries=5, delay_seconds=5)
                    search = arxiv.Search(id_list=[arxiv_id])
                    
                    # Get the arxiv paper
                    result = next(client.results(search), None)
                    
                    if not result:
                        errors.append(f"Could not find paper with arXiv ID {arxiv_id} via arXiv API")
                        continue
                    
                    # Use arxiv_id for filename
                    safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                    filename = f"{safe_arxiv_id}.pdf"
                    filepath = os.path.join(lab_data_dir, filename)
                    
                    # Download the PDF
                    result.download_pdf(dirpath=lab_data_dir, filename=filename)
                    
                    # Add to tracking database
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    cursor.execute(
                        "INSERT INTO paper_tracking (lab_name, paper_id, title, author, date_added, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                        (lab_name, paper_id, result.title, member_name, current_date, filepath),
                    )
                    
                    # Load and add to vector database
                    loader = PyPDFLoader(filepath)
                    documents = loader.load()
                    
                    # Generate UUIDs for each document chunk
                    uuids = [str(uuid4()) for _ in range(len(documents))]
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata["paper_id"] = paper_id  # This is arxiv_id
                        doc.metadata["title"] = result.title
                        doc.metadata["author"] = member_name
                        doc.metadata["all_authors"] = ", ".join(
                            str(a) for a in result.authors
                        )
                        doc.metadata["year"] = str(
                            result.published.year
                        )
                        doc.metadata["lab"] = lab_name
                        doc.metadata["source"] = "semantic_scholar_api"
                    
                    # Add to vector store
                    vector_store.add_documents(documents=documents, ids=uuids)
                    
                    papers_added += 1
                    print(f"Successfully added paper '{result.title}' to database.")
                    
                except arxiv.arxiv.ArxivError as e:
                    errors.append(f"arXiv API error for paper '{paper_title}' (ID: {arxiv_id}): {str(e)}")
                    time.sleep(5)  # wait before continuing
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    errors.append(f"Error processing paper '{paper_title}' (ID: {arxiv_id}): {str(e)}\n{tb_str}")
            
            # Wait between authors to avoid hitting rate limits
            time.sleep(10)
            
        except requests.exceptions.RequestException as e:
            errors.append(f"Network error accessing Semantic Scholar API for {member_name}: {str(e)}")
            time.sleep(10)
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            errors.append(f"Unexpected error processing papers for {member_name}: {str(e)}\n{tb_str}")
    
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
def generate_paper_summary(lab_name: str, arxiv_id: str, related_papers_count: int = 3):
    """
    Generate a comprehensive summary of a paper with arXiv ID, including context from related papers.
    
    Args:
        lab_name: The name of the lab that the paper belongs to
        arxiv_id: The arXiv ID of the paper to summarize
        related_papers_count: Number of related papers to include for context (default: 3)
        
    Returns:
        A comprehensive summary of the paper with context from related papers
    """
    lab_name = lab_name.lower()
    
    # Connect to the database
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"
    
    # Check if we have the paper in our database
    target_arxiv_id = arxiv_id.strip()
    cursor.execute(
        "SELECT paper_id, title, author, pdf_path FROM paper_tracking WHERE lab_name = ? AND paper_id = ?",
        (lab_name, target_arxiv_id),
    )
    paper_info = cursor.fetchone()
    
    if not paper_info:
        conn.close()
        return f"Could not find paper with arXiv ID {target_arxiv_id} in lab collection"
    
    # Extract paper info
    _, title, found_author, pdf_path = paper_info
    
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

        # Get lab papers collection to find related papers
        collection_name = f"{lab_name}_papers"
        lab_related = []
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
            lab_related = vector_store.similarity_search(
                search_text, k=related_papers_count
            )
        except Exception as e:
            print(f"Error searching lab collection: {e}")

        # Try to get recommendation collection as well
        recommendation_collection_name = f"recommendation_for_{lab_name}"
        recommendation_related = []
        try:
            recommendation_store = Chroma(
                collection_name=recommendation_collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
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
2.  **Contextualize:** Briefly explain how the target paper relates to the provided related papers.
3.  **Structure:** Organize the summary logically (Introduction/Problem, Methods, Results, Relation to Context, Conclusion/Significance).
4.  **Tone:** Maintain an academic, neutral, and objective tone.
5.  **Length:** Aim for approximately 400-600 words.

**Paper Information & Context:**
{summary_input}

---
**Generate the summary now:**
"""

        try:
            response = llm.invoke(prompt)
            summary = response.content
            conn.close()
            return summary
        except Exception as e:
            conn.close()
            return f"Error calling OpenAI API for summarization: {str(e)}"

    except FileNotFoundError:
        conn.close()
        return f"Error: PDF file not found at {pdf_path}"
    except Exception as e:
        conn.close()
        import traceback
        tb_str = traceback.format_exc()
        return f"Error generating paper summary for arXiv:{target_arxiv_id}: {str(e)}\n{tb_str}"


@function_tool
def check_new_papers_alt(lab_name: str):
    """
    Alternative version of check_new_papers that uses Semantic Scholar API instead of Google Scholar.
    This avoids rate limiting issues when Google Scholar blocks access with CAPTCHA.
    """
    lab_name = lab_name.lower()
    
    # Not directly calling crawl_semantic_scholar, but using the same implementation
    # Retrieve lab information
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Failed to check for new papers: Lab '{lab_name}' does not exist"

    # Execute crawl_semantic_scholar logic
    try:
        # Use external command line approach to avoid FunctionTool call issue
        # Note: This is not a true process call, just to avoid directly calling function object
        print(f"Checking for new papers for '{lab_name}' using Semantic Scholar API...")
        
        # Get lab members
        cursor.execute(
            "SELECT member_name, scholar_url FROM lab_members WHERE lab_name = ?",
            (lab_name,),
        )
        members = cursor.fetchall()

        if not members:
            conn.close()
            return f"Failed to check for new papers: No members found for lab '{lab_name}'"

        # Create paper_tracking table if it doesn't exist
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS paper_tracking
                 (lab_name text, paper_id text, title text, author text, date_added text, 
                 pdf_path text, PRIMARY KEY (lab_name, paper_id))"""
        )

        # Create lab collection if it doesn't exist
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
        
        # Semantic Scholar API base URL
        semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Optional - Get API key at https://www.semanticscholar.org/product/api
        semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        
        # API request headers
        headers = {
            "Accept": "application/json",
        }
        if semantic_scholar_api_key:
            headers["x-api-key"] = semantic_scholar_api_key
        
        for member_name, _ in members:
            print(f"Searching for papers by {member_name} on Semantic Scholar...")
            
            try:
                # First, find the author by name
                author_search_url = f"{semantic_scholar_base_url}/author/search"
                author_params = {
                    "query": member_name,
                    "fields": "authorId,name,affiliations,paperCount",
                    "limit": 3  # Get top 3 matches
                }
                
                author_response = requests.get(
                    author_search_url, 
                    headers=headers, 
                    params=author_params,
                    timeout=30
                )
                author_response.raise_for_status()
                author_data = author_response.json()
                
                # Find the best match
                author = author_data['data'][0]
                author_id = author['authorId']
                
                # Wait to avoid hitting rate limits
                time.sleep(2)
                
                # Get papers by this author
                papers_url = f"{semantic_scholar_base_url}/author/{author_id}/papers"
                papers_params = {
                    "fields": "paperId,externalIds,title,abstract,year,authors,url,venue,publicationDate",
                    "limit": 100
                }
                
                papers_response = requests.get(
                    papers_url, 
                    headers=headers, 
                    params=papers_params,
                    timeout=30
                )
                papers_response.raise_for_status()
                papers_data = papers_response.json()
                
                if not papers_data.get('data'):
                    errors.append(f"No papers found for author {member_name} (ID: {author_id})")
                    continue
                    
                # Process each paper
                print(f"Found {len(papers_data['data'])} papers for {member_name}")
                for i, paper in enumerate(papers_data['data']):
                    # Get arXiv ID if available
                    arxiv_id = None
                    if paper.get('externalIds') and paper['externalIds'].get('ArXiv'):
                        arxiv_id = paper['externalIds']['ArXiv']
                    
                    # Skip papers without arXiv ID
                    if not arxiv_id:
                        continue
                    
                    paper_title = paper.get('title', 'Unknown Title')
                    print(f"Processing paper: {paper_title} (arXiv:{arxiv_id})")
                    
                    # Use arxiv_id as the unique identifier
                    paper_id = arxiv_id
                    
                    # Check if paper already exists in the database
                    cursor.execute(
                        "SELECT * FROM paper_tracking WHERE lab_name = ? AND paper_id = ?",
                        (lab_name, paper_id),
                    )
                    existing_paper = cursor.fetchone()
                    
                    if existing_paper:
                        print(f"Paper with arXiv ID {arxiv_id} already exists in the database. Skipping.")
                        continue
                    
                    # Limit request frequency
                    if i > 0 and i % 5 == 0:
                        print("Pausing briefly to avoid rate limits...")
                        time.sleep(5)
                    
                    try:
                        # Use arxiv API to download paper
                        client = arxiv.Client(num_retries=5, delay_seconds=5)
                        search = arxiv.Search(id_list=[arxiv_id])
                        
                        # Get arxiv paper
                        result = next(client.results(search), None)
                        
                        if not result:
                            errors.append(f"Could not find paper with arXiv ID {arxiv_id} via arXiv API")
                            continue
                        
                        # Use arxiv_id as filename
                        safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                        filename = f"{safe_arxiv_id}.pdf"
                        filepath = os.path.join(lab_data_dir, filename)
                        
                        # Download PDF
                        result.download_pdf(dirpath=lab_data_dir, filename=filename)
                        
                        # Add to tracking database
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        cursor.execute(
                            "INSERT INTO paper_tracking (lab_name, paper_id, title, author, date_added, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                            (lab_name, paper_id, result.title, member_name, current_date, filepath),
                        )
                        
                        # Load and add to vector database
                        loader = PyPDFLoader(filepath)
                        documents = loader.load()
                        
                        # Generate UUIDs for each document chunk
                        uuids = [str(uuid4()) for _ in range(len(documents))]
                        
                        # Add metadata
                        for doc in documents:
                            doc.metadata["paper_id"] = paper_id  # This is arxiv_id
                            doc.metadata["title"] = result.title
                            doc.metadata["author"] = member_name
                            doc.metadata["all_authors"] = ", ".join(
                                str(a) for a in result.authors
                            )
                            doc.metadata["year"] = str(
                                result.published.year
                            )
                            doc.metadata["lab"] = lab_name
                            doc.metadata["source"] = "semantic_scholar_api"
                        
                        # Add to vector store
                        vector_store.add_documents(documents=documents, ids=uuids)
                        
                        papers_added += 1
                        print(f"Successfully added paper '{result.title}' to database.")
                        
                    except arxiv.arxiv.ArxivError as e:
                        errors.append(f"arXiv API error for paper '{paper_title}' (ID: {arxiv_id}): {str(e)}")
                        time.sleep(5)
                    except Exception as e:
                        import traceback
                        tb_str = traceback.format_exc()
                        errors.append(f"Error processing paper '{paper_title}' (ID: {arxiv_id}): {str(e)}\n{tb_str}")
                
                # Wait between different authors to avoid hitting rate limits
                time.sleep(10)
                
            except requests.exceptions.RequestException as e:
                errors.append(f"Network error accessing Semantic Scholar API for {member_name}: {str(e)}")
                time.sleep(10)
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                errors.append(f"Unexpected error processing papers for {member_name}: {str(e)}\n{tb_str}")
        
        conn.commit()
        conn.close()
        
        result = f"Process completed for lab '{lab_name}'. Added {papers_added} new papers."
        if errors:
            # If errors too many, log only summary
            error_summary = (
                errors
                if len(errors) <= 5
                else errors[:5] + [f"... ({len(errors) - 5} more errors)"]
            )
            result += f"\nEncountered {len(errors)} errors. Error summary: {json.dumps(error_summary, indent=2)}"
        
        return f"Checked for new papers (via Semantic Scholar API): {result}"
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        conn.close()
        return f"Failed to check for new papers (via Semantic Scholar API): Error: {str(e)}\n{tb_str}"


@function_tool
def summarize_latest_author_paper(lab_name: str, author_name: str, related_papers_count: int = 3):
    """
    Find and summarize the latest paper by a specific author in a lab.
    
    Args:
        lab_name: The name of the lab
        author_name: The name of the author whose latest paper should be summarized
        related_papers_count: Number of related papers to include for context (default: 3)
        
    Returns:
        A comprehensive summary of the author's latest paper with context from related papers,
        or an error message if no papers are found
    """
    lab_name = lab_name.lower()
    
    # Connect to the database
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()
    
    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"
    
    # Check if the author has papers in this lab
    cursor.execute(
        """SELECT paper_id, title, date_added, pdf_path 
           FROM paper_tracking 
           WHERE lab_name = ? AND author = ? 
           ORDER BY date_added DESC 
           LIMIT 1""",
        (lab_name, author_name),
    )
    
    latest_paper = cursor.fetchone()
    
    if not latest_paper:
        conn.close()
        return f"No papers found for author '{author_name}' in lab '{lab_name}'"
    
    arxiv_id, title, date_added, pdf_path = latest_paper
    
    # Display which paper we're summarizing
    print(f"Summarizing latest paper by {author_name}: '{title}' (arXiv:{arxiv_id}) from {date_added}")
    
    # First, check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at recorded path: {pdf_path}")
        
        # Create the directory path if it doesn't exist
        lab_data_dir = os.path.join(DEFAULT_DATA_DIR, lab_name)
        Path(lab_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Try alternative format for the filename
        safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
        alternative_path = os.path.join(lab_data_dir, f"{safe_arxiv_id}.pdf")
        
        if os.path.exists(alternative_path):
            print(f"Found PDF at alternative path: {alternative_path}")
            pdf_path = alternative_path
        else:
            print(f"Attempting to download the paper from arXiv...")
            
            try:
                # Try to download the PDF using arXiv API
                client = arxiv.Client(num_retries=5, delay_seconds=5)
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(client.results(search), None)
                
                if paper:
                    # Download the PDF
                    filename = f"{safe_arxiv_id}.pdf"
                    pdf_path = os.path.join(lab_data_dir, filename)
                    paper.download_pdf(dirpath=lab_data_dir, filename=filename)
                    
                    # Update the database with the new path
                    cursor.execute(
                        "UPDATE paper_tracking SET pdf_path = ? WHERE lab_name = ? AND paper_id = ?",
                        (pdf_path, lab_name, arxiv_id)
                    )
                    conn.commit()
                    
                    print(f"Successfully downloaded paper to {pdf_path}")
                else:
                    conn.close()
                    return f"Error: Could not find paper with arXiv ID {arxiv_id} on arXiv. The paper might be in the database but is no longer available."
            except Exception as e:
                conn.close()
                return f"Error: Failed to download PDF for arXiv:{arxiv_id}. Please try again later. Details: {str(e)}"
    
    # Implementation from generate_paper_summary:
    try:
        if not os.path.exists(pdf_path):
            conn.close()
            return f"Error: PDF file still not found after recovery attempts. Path: {pdf_path}"

        # Get the ArXiv API client
        client = arxiv.Client(num_retries=5, delay_seconds=5)

        # First, load the PDF for embedding
        try:
            loader = PyPDFLoader(pdf_path)
            target_paper_docs = loader.load()
            
            if not target_paper_docs:
                conn.close()
                return f"Error: The PDF file at {pdf_path} appears to be empty or corrupted."
                
        except Exception as pdf_error:
            conn.close()
            return f"Error: Could not load the PDF file. The file may be corrupted. Details: {str(pdf_error)}"

        # Also get abstract from arXiv API if possible
        abstract = ""
        try:
            # Get paper from ArXiv API
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search), None)

            if paper:
                abstract = paper.summary
        except Exception as e:
            print(f"Warning: Error getting abstract from arXiv API: {e}")
            # Continue without abstract rather than failing completely

        # Combine all page content for searching related papers AND for summarization
        full_paper_text = "\n\n".join([doc.page_content for doc in target_paper_docs])

        # Use the entire paper for similarity search
        search_text = full_paper_text

        # Get lab papers collection to find related papers
        collection_name = f"{lab_name}_papers"
        lab_related = []
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
            lab_related = vector_store.similarity_search(
                search_text, k=related_papers_count
            )
        except Exception as e:
            print(f"Warning: Error searching lab collection: {e}")
            # Continue without lab related papers

        # Try to get recommendation collection as well
        recommendation_collection_name = f"recommendation_for_{lab_name}"
        recommendation_related = []
        try:
            recommendation_store = Chroma(
                collection_name=recommendation_collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
            recommendation_related = recommendation_store.similarity_search(
                search_text, k=related_papers_count
            )
        except Exception as e:
            print(f"Warning: Error searching recommendation collection: {e}")
            # Continue without recommendation related papers

        all_related = lab_related + recommendation_related

        # Extract titles and brief content from related papers, filtering out the target paper
        related_info = []
        processed_related_ids = set()

        # Try to get related paper sections from their LaTeX sources
        for doc in all_related:
            related_arxiv_id = doc.metadata.get("paper_id")  # paper_id is arxiv_id

            if (
                related_arxiv_id
                and related_arxiv_id != arxiv_id
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
Target Paper: {title} (arXiv:{arxiv_id}) by {author_name}

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
2.  **Contextualize:** Briefly explain how the target paper relates to the provided related papers.
3.  **Structure:** Organize the summary logically (Introduction/Problem, Methods, Results, Relation to Context, Conclusion/Significance).
4.  **Tone:** Maintain an academic, neutral, and objective tone.
5.  **Length:** Aim for approximately 400-600 words.

**Paper Information & Context:**
{summary_input}

---
**Generate the summary now:**
"""

        try:
            response = llm.invoke(prompt)
            summary = response.content
            conn.close()
            return summary
        except Exception as e:
            conn.close()
            return f"Error calling OpenAI API for summarization: {str(e)}"

    except FileNotFoundError as fnf_error:
        conn.close()
        return f"Error: PDF file not found at {pdf_path}. Specific error: {str(fnf_error)}"
    except Exception as e:
        conn.close()
        import traceback
        tb_str = traceback.format_exc()
        return f"Error generating paper summary for arXiv:{arxiv_id}: {str(e)}\n{tb_str}"
