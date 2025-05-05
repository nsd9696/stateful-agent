from typing import List, Optional, Dict, Any
import os
import json
import logging
import time
import requests
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

# Configure logging
log_path = os.path.expanduser('~/Desktop/rednote_error.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG)

class PaperData(BaseModel):
    """Model for paper data used in Rednote publishing"""
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(..., description="Abstract of the paper")
    authors: str = Field(..., description="Authors of the paper")
    publication: str = Field(..., description="Publication venue of the paper")
    url: Optional[str] = Field(None, description="URL to the paper")
    pdf_url: Optional[str] = Field(None, description="URL to the PDF version of the paper")

class RednoteResponse(BaseModel):
    """Response model for Rednote API calls"""
    success: bool
    message: str
    data: Dict[str, Any] = {}
    post_id: Optional[str] = None

def create_rednote_content_with_gpt(paper_data: PaperData) -> str:
    """Create engaging content for Rednote using GPT.
    
    Args:
        paper_data: PaperData object containing paper information
        
    Returns:
        str: Formatted content suitable for Rednote posting
    """
    # Use the appropriate API key for content generation
    content_api_key = os.getenv("OPENAI_API_KEY_SUMMARIZER")
    if not content_api_key:
        raise ValueError("OPENAI_API_KEY_SUMMARIZER environment variable not set")
        
    llm = ChatOpenAI(model="gpt-4", api_key=content_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a content creator for Xiaohongshu (Rednote), a Chinese social media platform.
        Create engaging content about academic papers that will appeal to a general audience.
        The content should be informative, easy to understand, and include relevant hashtags.
        
        Format the content in Chinese with the following structure:
        1. Catchy title
        2. Brief introduction
        3. Key findings
        4. Why it matters
        5. Hashtags
        
        Keep the tone conversational and engaging."""),
        ("user", """Create a Rednote post about this paper:
        Title: {title}
        Abstract: {abstract}
        Authors: {authors}
        Publication: {publication}""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "title": paper_data.title,
        "abstract": paper_data.abstract,
        "authors": paper_data.authors,
        "publication": paper_data.publication
    })
    
    return response.content

async def publish_rednote_post(paper_data: PaperData, images: list = None) -> RednoteResponse:
    """Publish a post to Rednote.
    
    Args:
        paper_data: PaperData object containing paper information
        images: Optional list of image paths to include
        
    Returns:
        RednoteResponse: Response from Rednote API
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Navigate to Rednote and perform login
            await page.goto("https://creator.xiaohongshu.com/login")
            
            # QR code login
            qr_code = await page.wait_for_selector(".qrcode-img img")
            if not qr_code:
                return RednoteResponse(success=False, message="QR code not found")
            
            qr_code_url = await qr_code.get_attribute("src")
            if not qr_code_url:
                return RednoteResponse(success=False, message="QR code URL not found")
            
            response = requests.get(qr_code_url)
            if response.status_code != 200:
                return RednoteResponse(success=False, message="Failed to download QR code")
            
            qr_code_path = "qr_code.png"
            with open(qr_code_path, "wb") as f:
                f.write(response.content)
            
            logging.info("Please scan the QR code to login")
            os.system(f"open {qr_code_path}")
            
            # Wait for login
            await page.wait_for_url("**/creator/home", timeout=60000)
            os.remove(qr_code_path)
            
            # Create new post
            await page.click(".btn.el-tooltip__trigger.el-tooltip__trigger")
            upload_tab = await page.wait_for_selector(".creator-tab:has(span.title:text('上传图文'))", timeout=5000)
            if upload_tab:
                await upload_tab.click()
                await asyncio.sleep(1)
            
            # Upload media if provided
            if images:
                async with page.expect_file_chooser() as fc_info:
                    await page.click(".upload-input")
                file_chooser = await fc_info.value
                await file_chooser.set_files(images)
                await asyncio.sleep(1)
            
            # Enter title and content
            title_input = await page.wait_for_selector(".d-text", timeout=5000)
            if not title_input:
                return RednoteResponse(success=False, message="Could not find title input field")
            await title_input.fill(paper_data.title)
            
            content_input = await page.wait_for_selector(".ql-editor.ql-blank", timeout=5000)
            if not content_input:
                return RednoteResponse(success=False, message="Could not find content input field")
            await content_input.fill(paper_data.abstract)
            
            # Publish
            await asyncio.sleep(1)
            publish_button = await page.wait_for_selector(".d-button-content:has-text('发布')", timeout=5000)
            if not publish_button:
                return RednoteResponse(success=False, message="Could not find publish button")
            await publish_button.click()
            
            # Get post ID from URL
            await asyncio.sleep(5)
            post_id = page.url.split('/')[-1]
            
            await browser.close()
            return RednoteResponse(success=True, message="Post published successfully", post_id=post_id)
            
    except Exception as e:
        logging.error(f"Error publishing to Rednote: {str(e)}")
        return RednoteResponse(success=False, message=f"Error publishing post: {str(e)}")

async def publish_paper_to_rednote(paper_data: PaperData, images: list[str] = None) -> RednoteResponse:
    """Publish a paper summary to Rednote.
    
    Args:
        paper_data: PaperData object containing paper information
        images: Optional list of image paths to include
        
    Returns:
        RednoteResponse: Response from Rednote API
    """
    content = create_rednote_content_with_gpt(paper_data)
    paper_data.abstract = content  # Update the abstract with the generated content
    return await publish_rednote_post(paper_data, images)

async def search_and_publish_paper_rednote(research_area: str, title_keywords: Optional[str] = None,
                                         author_name: Optional[str] = None, limit: int = 1) -> RednoteResponse:
    """Search for papers and publish the first matching one to Rednote.
    
    Args:
        research_area: Research area to search in
        title_keywords: Optional keywords to search in paper titles
        author_name: Optional author name to filter by
        limit: Maximum number of papers to consider
        
    Returns:
        RednoteResponse: Response from Rednote API
    """
    # Implementation would use paper_crawler functions to search for papers
    # and then publish the first matching one
    return RednoteResponse(success=False, message="Not implemented yet") 