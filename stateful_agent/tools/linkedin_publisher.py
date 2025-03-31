import os
from typing import Literal
from dotenv import load_dotenv
import requests
from pydantic import BaseModel

load_dotenv()

class LinkedInPostRequest(BaseModel):
    commentary: str
    visibility: Literal["PUBLIC", "CONNECTIONS"] = "PUBLIC"

class LinkedInResponse(BaseModel):
    status: str
    message: str
    data: dict

class LinkedInPublisher:
    """
    LinkedIn Publisher tool for posting content to LinkedIn.
    """
    def __init__(self):
        self.user_id = os.getenv('LINKEDIN_USER_ID')
        self.access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        
        if not self.user_id or not self.access_token:
            raise ValueError("LinkedIn credentials not found in environment variables")
        
        self.base_url = "https://api.linkedin.com/rest/posts"

def publish_linkedin_post(request: LinkedInPostRequest) -> LinkedInResponse:
    """
    Publish a post to LinkedIn.
    
    Args:
        request: LinkedInPostRequest containing:
            commentary: The content of the post
            visibility: Post visibility setting ("PUBLIC" or "CONNECTIONS")
    
    Returns:
        LinkedInResponse: Response from LinkedIn API
    """
    publisher = LinkedInPublisher()
    
    headers = {
        "Authorization": f"Bearer {publisher.access_token}",
        "LinkedIn-Version": "202306",
        "Content-Type": "application/json"
    }
    
    payload = {
        "author": f"urn:li:person:{publisher.user_id}",
        "commentary": request.commentary,
        "visibility": request.visibility,
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": []
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False
    }
    
    response = requests.post(
        publisher.base_url,
        headers=headers,
        json=payload
    )
    
    if not response.ok:
        return LinkedInResponse(
            status="error",
            message=f"Failed to publish LinkedIn post: {response.text}",
            data={}
        )
    
    return LinkedInResponse(
        status="success",
        message="Post published successfully",
        data=response.json()
    )