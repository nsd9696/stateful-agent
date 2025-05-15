import os
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from ratelimit import limits, sleep_and_retry
import hashlib
import json
import PyPDF2
import io
import tempfile
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
import openai
import random
from hyperpocket.tool import function_tool

load_dotenv()

# Rate limiting configuration: 100 calls per 5 minutes as per Semantic Scholar guidelines
CALLS_PER_PERIOD = 100
PERIOD_IN_SECONDS = 300

# OpenAI rate limit handling
OPENAI_RATE_LIMIT_RETRIES = 3
OPENAI_RETRY_DELAY = 2  # seconds

class PaperSearchRequest(BaseModel):
    research_area: str
    author_name: Optional[str] = None
    limit: int = 10
    year_from: Optional[int] = None
    sort_by: str = "relevance"  # "relevance" or "year"
    download_pdfs: bool = True  # Whether to download PDFs for papers

class PaperSearchResponse(BaseModel):
    papers: List[dict] = Field(default_factory=list)
    total_results: int
    query_info: dict = Field(default_factory=dict)

class PaperSummaryRequest(BaseModel):
    paper_id: str
    max_length: int = 500
    include_citations: bool = True
    include_visuals: bool = True
    include_key_points: bool = True
    include_methodology: bool = True
    include_results: bool = True

class PaperSummaryResponse(BaseModel):
    title: str
    authors: List[str]
    year: int
    summary: str
    citations: Optional[int] = None
    url: Optional[str] = None
    pdf_path: Optional[str] = None
    key_points: Optional[List[str]] = None
    methodology: Optional[str] = None
    results: Optional[str] = None
    visual_elements: Optional[List[Dict[str, str]]] = None

@function_tool
def scrape_papers(
    research_area: str,
    author_name: Optional[str] = None,
    limit: int = 10,
    year_from: Optional[int] = None,
    sort_by: str = "relevance",
    download_pdfs: bool = True
) -> PaperSearchResponse:
    """
    Scrape papers from Semantic Scholar based on research area and optionally author name.
    Rate limited to 100 calls per 5 minutes.
    
    Args:
            research_area: The research area/topic to search for
            author_name: Optional author name to filter by
            limit: Maximum number of papers to return (default: 10)
            year_from: Optional year to start searching from
            sort_by: How to sort results ("relevance" or "year")
            download_pdfs: Whether to download PDFs for papers (default: True)
    
    Returns:
        PaperSearchResponse: List of papers and metadata
        
    Example:
        >>> response = scrape_papers(
        ...     research_area="transformer architecture",
        ...     author_name="Vaswani",
        ...     limit=5,
        ...     year_from=2020
        ... )
    """
    request = PaperSearchRequest(
        research_area=research_area,
        author_name=author_name,
        limit=limit,
        year_from=year_from,
        sort_by=sort_by,
        download_pdfs=download_pdfs
    )
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Get API key from environment
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    
    # Construct query
    query = request.research_area
    if request.author_name:
        query += f" author:{request.author_name}"
    
    # Build parameters
    params = {
        "query": query,
        "limit": min(request.limit, 100),  # Cap at 100 papers per request
        "fields": "title,authors,abstract,year,venue,citationCount,influentialCitationCount,url,openAccessPdf,paperId",
        "sort": request.sort_by
    }
    
    if request.year_from:
        params["year"] = f"{request.year_from}-"
    
    headers = {
        "Accept": "application/json"
    }
    
    # Add API key if available
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        # Add exponential backoff for retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        data = response.json()
        
        # Process and filter results
        papers = []
        for paper in data.get("data", []):
            paper_id = paper.get("paperId")
            pdf_url = paper.get("openAccessPdf", {}).get("url")
            pdf_path = None
            
            # Download PDF if requested and available
            if request.download_pdfs and pdf_url and paper_id:
                pdf_path = download_paper_pdf(pdf_url, paper_id, request.research_area)
            
            processed_paper = {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "authors": [author.get("name") for author in paper.get("authors", [])],
                "year": paper.get("year"),
                "abstract": paper.get("abstract"),
                "venue": paper.get("venue"),
                "citations": paper.get("citationCount"),
                "influential_citations": paper.get("influentialCitationCount"),
                "url": paper.get("url"),
                "pdf_url": pdf_url,
                "pdf_path": pdf_path
            }
            papers.append(processed_paper)
        
        # Save papers if any were found
        if papers:
            save_papers_to_db(papers, request.research_area)
        
        return PaperSearchResponse(
            papers=papers,
            total_results=data.get("total", 0),
            query_info={
                "research_area": request.research_area,
                "author": request.author_name,
                "year_from": request.year_from,
                "sort_by": request.sort_by
            }
        )
        
    except requests.exceptions.RequestException as e:
        return PaperSearchResponse(
            papers=[],
            total_results=0,
            query_info={
                "error": f"Failed to fetch papers: {str(e)}",
                "research_area": request.research_area,
                "author": request.author_name
            }
        )

def download_paper_pdf(pdf_url: str, paper_id: str, research_area: str) -> Optional[str]:
    """
    Download a paper PDF and save it to the filesystem.
    
    Args:
        pdf_url: URL of the PDF to download
        paper_id: Unique identifier for the paper
        research_area: Research area for organizing files
        
    Returns:
        Optional[str]: Path to the downloaded PDF, or None if download failed
    """
    try:
        # Create base directory if it doesn't exist
        base_dir = os.path.join("data", "paper_pdfs")
        os.makedirs(base_dir, exist_ok=True)
        
        # Create research area directory (sanitized)
        area_dir = os.path.join(base_dir, sanitize_filename(research_area))
        os.makedirs(area_dir, exist_ok=True)
        
        # Create filename using paper ID
        filename = f"{paper_id}.pdf"
        filepath = os.path.join(area_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            return filepath
        
        # Download the PDF
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        # Save the PDF
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath
        
    except Exception as e:
        print(f"Error downloading PDF for paper {paper_id}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a paper by its ID from the saved papers.
    
    Args:
        paper_id: Unique identifier for the paper
        
    Returns:
        Optional[Dict[str, Any]]: Paper data if found, None otherwise
    """
    try:
        # Search in all research area directories
        base_dir = os.path.join("data", "scraped_papers")
        if not os.path.exists(base_dir):
            return None
            
        for area_dir in os.listdir(base_dir):
            area_path = os.path.join(base_dir, area_dir)
            if not os.path.isdir(area_path):
                continue
                
            for filename in os.listdir(area_path):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(area_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for paper in data.get("papers", []):
                    if paper.get("paper_id") == paper_id:
                        return paper
                        
        return None
        
    except Exception as e:
        print(f"Error retrieving paper {paper_id}: {str(e)}")
        return None

def extract_figures_from_pdf(pdf_path: str) -> List[Tuple[str, bytes]]:
    """
    Extract figures from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples containing figure captions and image data
    """
    figures = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract text to find figure captions
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            # Find figure captions using regex
            figure_captions = re.findall(r'Figure \d+[.:] .*?(?=\n\n|\Z)', text, re.DOTALL)
            
            # For each page, try to extract images
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                
                # Check if page has images
                if '/XObject' in page['/Resources']:
                    x_objects = page['/Resources']['/XObject'].get_object()
                    
                    for obj in x_objects:
                        if x_objects[obj]['/Subtype'] == '/Image':
                            try:
                                # Get image data
                                image = x_objects[obj]
                                image_data = image._data
                                
                                # Find matching caption (simplified approach)
                                caption = "Figure"
                                for cap in figure_captions:
                                    if f"Figure {len(figures)+1}" in cap:
                                        caption = cap
                                        break
                                
                                figures.append((caption, image_data))
                            except Exception as e:
                                print(f"Error extracting image: {str(e)}")
                                
        return figures
    except Exception as e:
        print(f"Error extracting figures from PDF {pdf_path}: {str(e)}")
        return []

def create_visualization_from_data(pdf_text: str) -> Optional[bytes]:
    """
    Create a visualization from data found in the PDF text.
    
    Args:
        pdf_text: Text extracted from the PDF
        
    Returns:
        Optional[bytes]: Image data if visualization was created, None otherwise
    """
    try:
        # Look for numerical data in the text
        # This is a simplified approach - in a real implementation, you'd use more sophisticated parsing
        data_pattern = r'(\d+\.\d+)'
        numbers = re.findall(data_pattern, pdf_text)
        
        if len(numbers) >= 5:
            # Convert to float
            data = [float(num) for num in numbers[:10]]  # Limit to 10 numbers
            
            # Create a simple bar chart
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(data)), data)
            plt.title('Data from Paper')
            plt.xlabel('Index')
            plt.ylabel('Value')
            
            # Save to bytes
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            
            return img_buffer.getvalue()
        
        return None
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

def extract_key_points_from_text(text: str) -> List[str]:
    """
    Extract key points from text.
    
    Args:
        text: Text to extract key points from
        
    Returns:
        List of key points
    """
    key_points = []
    
    # Look for bullet points or numbered lists
    bullet_pattern = r'[•\-\*]\s+(.*?)(?=\n[•\-\*]|\n\n|\Z)'
    numbered_pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\n\n|\Z)'
    
    bullet_points = re.findall(bullet_pattern, text, re.DOTALL)
    numbered_points = re.findall(numbered_pattern, text, re.DOTALL)
    
    key_points.extend(bullet_points)
    key_points.extend(numbered_points)
    
    # If no bullet points found, try to extract sentences that might be key points
    if not key_points:
        # Look for sentences that start with keywords
        keyword_pattern = r'(?:Key|Important|Significant|Main|Primary|Critical|Essential|Notable|Remarkable|Noteworthy).*?[.:]\s+(.*?)(?=\n\n|\Z)'
        keyword_points = re.findall(keyword_pattern, text, re.DOTALL | re.IGNORECASE)
        key_points.extend(keyword_points)
    
    # Clean up and limit to 5 points
    key_points = [point.strip() for point in key_points if len(point.strip()) > 10]
    return key_points[:5]

def extract_methodology_from_text(text: str) -> Optional[str]:
    """
    Extract methodology section from text.
    
    Args:
        text: Text to extract methodology from
        
    Returns:
        Methodology text if found, None otherwise
    """
    # Look for methodology section
    methodology_pattern = r'(?:Methodology|Methods|Approach|Implementation|Experimental Setup|Proposed Method).*?\n(.*?)(?=\n\n|\Z)'
    match = re.search(methodology_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        methodology = match.group(1).strip()
        # Limit length
        if len(methodology) > 300:
            methodology = methodology[:297] + "..."
        return methodology
    
    return None

def extract_results_from_text(text: str) -> Optional[str]:
    """
    Extract results section from text.
    
    Args:
        text: Text to extract results from
        
    Returns:
        Results text if found, None otherwise
    """
    # Look for results section
    results_pattern = r'(?:Results|Findings|Outcomes|Performance|Evaluation).*?\n(.*?)(?=\n\n|\Z)'
    match = re.search(results_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        results = match.group(1).strip()
        # Limit length
        if len(results) > 300:
            results = results[:297] + "..."
        return results
    
    return None

def save_papers_to_db(papers: List[dict], research_area: str) -> bool:
    """
    Save scraped papers as JSON files under data/scraped_papers directory.
    Papers are organized by research area and date of scraping.
    
    Args:
        papers: List of paper dictionaries containing paper metadata
        research_area: The research area/topic these papers belong to
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create base directory if it doesn't exist
        base_dir = os.path.join("data", "scraped_papers")
        os.makedirs(base_dir, exist_ok=True)
        
        # Create research area directory (sanitized)
        area_dir = os.path.join(base_dir, sanitize_filename(research_area))
        os.makedirs(area_dir, exist_ok=True)
        
        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save papers with metadata
        save_data = {
            "research_area": research_area,
            "scrape_date": datetime.now().isoformat(),
            "paper_count": len(papers),
            "papers": papers
        }
        
        # Generate unique filename using timestamp and content hash
        content_hash = hashlib.md5(json.dumps(save_data).encode()).hexdigest()[:8]
        filename = f"papers_{timestamp}_{content_hash}.json"
        
        filepath = os.path.join(area_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        print(f"Error saving papers: {str(e)}")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Convert research area string to valid directory name.
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Convert to lowercase and replace spaces with underscores
    filename = filename.lower().replace(' ', '_')
    
    return filename

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {str(e)}", e.doc, e.pos)