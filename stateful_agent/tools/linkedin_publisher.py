import os
from typing import Literal, Optional, Tuple, Dict, Any, List
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
import base64
import time
import random
from langchain_openai import ChatOpenAI
from .paper_scraper import (
    PaperSearchRequest, 
    PaperSearchResponse, 
    PaperSummaryRequest, 
    PaperSummaryResponse,
    scrape_papers,
    get_paper_by_id,
    extract_text_from_pdf,
    extract_key_points_from_text,
    extract_methodology_from_text,
    extract_results_from_text,
    extract_figures_from_pdf,
    create_visualization_from_data
)
load_dotenv()

OPENAI_RATE_LIMIT_RETRIES = 3
OPENAI_RETRY_DELAY = 2

class LinkedInPostRequest(BaseModel):
    """Request model for LinkedIn post content"""
    content: str
    pdf_data: Optional[bytes] = None
    visibility: Literal["PUBLIC", "CONNECTIONS"] = "PUBLIC"

class LinkedInResponse(BaseModel):
    """Response model for LinkedIn API calls"""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    post_id: Optional[str] = None

def generate_linkedin_post_with_chatgpt(paper_data: Dict[str, Any], pdf_text: str = "") -> str:
    """
    Generate a thoughtful LinkedIn post about a research paper using ChatGPT.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY_SUMMARIZER")
        if not api_key:
            print("OpenAI API key not found. Using fallback method.")
            return format_linkedin_post_basic(paper_data)
        for attempt in range(OPENAI_RATE_LIMIT_RETRIES):
            try:
                llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
                title = paper_data.get("title", "Untitled Paper")
                authors = ", ".join(paper_data.get("authors", []))
                year = paper_data.get("year", "")
                abstract = paper_data.get("abstract", "")
                citations = paper_data.get("citations", 0)
                url = paper_data.get("url", "")
                prompt = (
                    "The LinkedIn post should:\n"
                    "1. Start with an attention-grabbing introduction\n"
                    "2. Highlight the key contributions and findings\n"
                    "3. Explain why this research is important\n"
                    "4. Include relevant hashtags\n"
                    "5. Be engaging and accessible to a general audience\n"
                    "6. Include emojis to make it visually appealing\n"
                    "7. End with a call to action or thought-provoking question\n\n"
                    "Format the post with proper line breaks and structure.\n"
                )

                prompt += (
                    "----\n\n"
                    "**Paper details:**\n"
                    "- **Title:** {title}\n"
                    "- **Authors:** {authors}\n"
                    "- **Year:** {year}\n"
                    "- **Citations:** {citations}\n"
                    "- **URL:** {url}\n\n"
                    "**Abstract (for reference):**\n"
                    "{abstract}\n\n"
                ).format(
                    title=title,
                    authors=authors,
                    year=year,
                    citations=citations,
                    url=url,
                    abstract=abstract
                )

                if pdf_text:
                    prompt += (
                        "Additional content from the paper (first 2000 chars):\n"
                        "{pdf_text[:2000]}\n"
                    ).format(pdf_text=pdf_text)

                prompt += (
                    "Please generate the LinkedIn post now."
                )

                response = llm.invoke(prompt)
                post = response.content
                if len(post) > 3000:
                    post = post[:2997] + "..."
                return post
            except Exception as e:
                if attempt < OPENAI_RATE_LIMIT_RETRIES - 1:
                    time.sleep(OPENAI_RETRY_DELAY)
                else:
                    print(f"OpenAI rate limit or error after {OPENAI_RATE_LIMIT_RETRIES} retries. Using fallback method. Error: {str(e)}")
                    return format_linkedin_post_basic(paper_data)
        return format_linkedin_post_basic(paper_data)
    except Exception as e:
        print(f"Error in generate_linkedin_post_with_chatgpt: {str(e)}")
        return format_linkedin_post_basic(paper_data)

def format_linkedin_post_basic(paper_data: Dict[str, Any]) -> str:
    """
    Format a basic LinkedIn post about a paper (fallback method).
    """
    title = paper_data.get("title", "Untitled Paper")
    authors = ", ".join(paper_data.get("authors", []))
    year = paper_data.get("year", "")
    abstract = paper_data.get("abstract", "")
    citations = paper_data.get("citations", 0)
    url = paper_data.get("url", "")
    post = f"🧠 {title} 🧠\n\n"
    post += f"By {authors} ({year})\n\n"
    post += f"{abstract}\n\n"
    if citations:
        post += f"📚 Citations: {citations}\n\n"
    if url:
        post += f"🔗 Read the full paper: {url}\n\n"
    post += "#AI #MachineLearning #Research #AcademicPaper #Innovation"
    if len(post) > 3000:
        post = post[:2997] + "..."
    return post

def summarize_paper_for_linkedin(request: 'PaperSummaryRequest') -> 'PaperSummaryResponse':
    """
    Generate a comprehensive summary of a paper suitable for a LinkedIn post.
    """
    paper = get_paper_by_id(request.paper_id)
    if not paper:
        return PaperSummaryResponse(
            title="Paper not found",
            authors=[],
            year=0,
            summary="The requested paper could not be found.",
            citations=None,
            url=None,
            pdf_path=None,
            key_points=None,
            methodology=None,
            results=None,
            visual_elements=None
        )
    pdf_text = ""
    if paper.get("pdf_path") and os.path.exists(paper["pdf_path"]):
        pdf_text = extract_text_from_pdf(paper["pdf_path"])
    linkedin_post = generate_linkedin_post_with_chatgpt(paper, pdf_text)
    key_points = None
    if request.include_key_points and pdf_text:
        key_points = extract_key_points_from_text(pdf_text)
    methodology = None
    if request.include_methodology and pdf_text:
        methodology = extract_methodology_from_text(pdf_text)
    results = None
    if request.include_results and pdf_text:
        results = extract_results_from_text(pdf_text)
    visual_elements = None
    if request.include_visuals and paper.get("pdf_path") and os.path.exists(paper["pdf_path"]):
        figures = extract_figures_from_pdf(paper["pdf_path"])
        if not figures and pdf_text:
            viz_data = create_visualization_from_data(pdf_text)
            if viz_data:
                figures.append(("Data Visualization", viz_data))
        if figures:
            visual_elements = []
            for caption, image_data in figures[:3]:
                try:
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    visual_elements.append({
                        "caption": caption,
                        "image_data": base64_data
                    })
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
    return PaperSummaryResponse(
        title=paper.get("title", "Untitled"),
        authors=paper.get("authors", []),
        year=paper.get("year", 0),
        summary=linkedin_post,
        citations=paper.get("citations") if request.include_citations else None,
        url=paper.get("url"),
        pdf_path=paper.get("pdf_path"),
        key_points=key_points,
        methodology=methodology,
        results=results,
        visual_elements=visual_elements
    )

def prepare_pdf_for_linkedin(pdf_path: str) -> Optional[bytes]:
    """
    Prepare a PDF file for LinkedIn attachment.
    """
    try:
        if not os.path.exists(pdf_path):
            return None
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        return pdf_data
    except Exception as e:
        print(f"Error preparing PDF for LinkedIn: {str(e)}")
        return None

def create_linkedin_post_from_paper(paper_id: str, max_length: int = 3000) -> Tuple[str, Optional[bytes]]:
    """
    Create a complete LinkedIn post from a paper, including summary and PDF attachment.
    """
    summary_request = PaperSummaryRequest(
        paper_id=paper_id,
        max_length=max_length,
        include_citations=True,
        include_visuals=True,
        include_key_points=True,
        include_methodology=True,
        include_results=True
    )
    summary = summarize_paper_for_linkedin(summary_request)
    pdf_data = None
    if summary.pdf_path:
        pdf_data = prepare_pdf_for_linkedin(summary.pdf_path)
    return summary.summary, pdf_data

def publish_linkedin_post(request: LinkedInPostRequest) -> LinkedInResponse:
    """
    Publish a post to LinkedIn with optional PDF attachment.
    
    Args:
        request: LinkedInPostRequest containing:
            content: The content of the post
            pdf_data: Optional PDF data to attach to the post
            visibility: Post visibility setting ("PUBLIC" or "CONNECTIONS")
    
    Returns:
        LinkedInResponse: Response from LinkedIn API
    """
    try:
        # Get access token from environment
        access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        if not access_token:
            return LinkedInResponse(
                success=False,
                message="LinkedIn access token not found in environment variables"
            )
        
        # Get user profile ID
        profile_id = os.getenv('LINKEDIN_USER_ID')
        if not profile_id:
            return LinkedInResponse(
                success=False,
                message="LinkedIn profile ID not found in environment variables"
            )
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        # Prepare post data
        post_data = {
            'author': f'urn:li:person:{profile_id}',
            'lifecycleState': 'PUBLISHED',
            'specificContent': {
                'com.linkedin.ugc.ShareContent': {
                    'shareCommentary': {
                        'text': request.content
                    },
                    'shareMediaCategory': 'NONE'
                }
            },
            'visibility': {
                'com.linkedin.ugc.MemberNetworkVisibility': request.visibility
            }
        }
        
        # If PDF data is provided, prepare media upload
        if request.pdf_data:
            # First, register the media upload
            media_headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            
            media_data = {
                'registerUploadRequest': {
                    'recipes': ['urn:li:digitalmediaRecipe:feedshare-image'],
                    'owner': f'urn:li:person:{profile_id}',
                    'serviceRelationships': [{
                        'relationshipType': 'OWNER',
                        'identifier': 'urn:li:userGeneratedContent'
                    }]
                }
            }
            
            # Register media upload
            media_response = requests.post(
                'https://api.linkedin.com/v2/assets?action=registerUpload',
                headers=media_headers,
                json=media_data
            )
            
            if media_response.status_code == 200:
                media_data = media_response.json()
                asset = media_data['value']['asset']
                upload_url = media_data['value']['uploadUrl']
                
                # Upload the PDF
                upload_headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/pdf'
                }
                
                upload_response = requests.put(
                    upload_url,
                    headers=upload_headers,
                    data=request.pdf_data
                )
                
                if upload_response.status_code == 201:
                    # Update post data to include the media
                    post_data['specificContent']['com.linkedin.ugc.ShareContent']['shareMediaCategory'] = 'DOCUMENT'
                    post_data['specificContent']['com.linkedin.ugc.ShareContent']['media'] = [{
                        'status': 'READY',
                        'description': {
                            'text': 'Research Paper PDF'
                        },
                        'media': asset,
                        'title': {
                            'text': 'Research Paper'
                        }
                    }]
        
        # Make the API request
        response = requests.post(
            'https://api.linkedin.com/v2/ugcPosts',
        headers=headers,
            json=post_data
        )
        
        # Check if the request was successful
        if response.status_code == 201:
            # Extract post ID from response headers
            post_id = response.headers.get('x-restli-id')
            
            return LinkedInResponse(
                success=True,
                message="Post published successfully",
                data={"status_code": response.status_code},
                post_id=post_id
            )
        else:
            # Handle error response
            error_message = "Failed to publish post"
            try:
                error_data = response.json()
                if 'message' in error_data:
                    error_message = error_data['message']
            except:
                pass
                
            return LinkedInResponse(
                success=False,
                message=error_message,
                data={"status_code": response.status_code}
            )
            
    except Exception as e:
        return LinkedInResponse(
            success=False,
            message=f"Error publishing post: {str(e)}"
        )

def publish_paper_to_linkedin(paper_id: str, max_length: int = 3000) -> LinkedInResponse:
    """
    Publish a paper summary to LinkedIn with PDF attachment.
    
    Args:
        paper_id: ID of the paper to publish
        max_length: Maximum length of the post in characters
        
    Returns:
        LinkedInResponse: Response from LinkedIn API
    """
    # Create LinkedIn post content and get PDF data
    post_content, pdf_data = create_linkedin_post_from_paper(paper_id, max_length)
    
    # Create request
    request = LinkedInPostRequest(
        content=post_content,
        pdf_data=pdf_data
    )
    
    # Publish to LinkedIn
    return publish_linkedin_post(request)

def search_and_publish_paper(research_area: str, title_keywords: Optional[str] = None, 
                            author_name: Optional[str] = None, limit: int = 1) -> LinkedInResponse:
    """
    Search for a paper and publish it to LinkedIn.
    
    Args:
        research_area: Research area to search for
        title_keywords: Optional keywords to match in paper title
        author_name: Optional author name to filter by
        limit: Maximum number of papers to search for
        
    Returns:
        LinkedInResponse: Response from LinkedIn API
    """
    # Search for papers
    search_request = PaperSearchRequest(
        research_area=research_area,
        author_name=author_name,
        limit=limit,
        download_pdfs=True
    )
    
    search_response = scrape_papers(search_request)
    
    if not search_response.papers:
        return LinkedInResponse(
            success=False,
            message=f"No papers found for research area: {research_area}"
        )
    
    # Filter by title keywords if provided
    papers = search_response.papers
    if title_keywords:
        papers = [p for p in papers if title_keywords.lower() in p.get("title", "").lower()]
        
    if not papers:
        return LinkedInResponse(
            success=False,
            message=f"No papers found matching title keywords: {title_keywords}"
        )
    
    # Publish the first matching paper
    paper_id = papers[0].get("paper_id")
    return publish_paper_to_linkedin(paper_id)
