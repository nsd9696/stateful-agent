import os
from typing import Literal, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from .paper_scraper import (
    PaperSearchRequest, 
    PaperSearchResponse, 
    PaperSummaryRequest, 
    PaperSummaryResponse,
    create_linkedin_post_from_paper,
    scrape_papers
)

load_dotenv()

class LinkedInPostRequest(BaseModel):
    """Request model for LinkedIn post content"""
    content: str
    pdf_data: Optional[bytes] = None
    visibility: Literal["PUBLIC", "CONNECTIONS"] = "PUBLIC"

class LinkedInResponse(BaseModel):
    """Response model for LinkedIn API calls"""
    success: bool
    message: str
    data: Dict[str, Any] = {}
    post_id: Optional[str] = None

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
