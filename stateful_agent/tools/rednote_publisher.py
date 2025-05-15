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
from playwright.sync_api import sync_playwright, BrowserContext
from PyQt6.QtWidgets import QMessageBox, QLabel, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
import base64
import tempfile
import webbrowser
from pdf2image import convert_from_path
import io
from PIL import Image
from openai import OpenAI
from hyperpocket.tool import function_tool
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

# Configure logging
log_path = os.path.expanduser('~/Desktop/rednote_error.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # Add console handler
    ]
)

# Constants for file paths
TOKEN_FILE = os.path.expanduser('~/.xhs_system/xiaohongshu_token.json')
COOKIES_FILE = os.path.expanduser('~/.xhs_system/xiaohongshu_cookies.json')

# Ensure directories exist
os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
os.makedirs(os.path.dirname(COOKIES_FILE), exist_ok=True)

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

def _load_token() -> Optional[str]:
    """从文件加载token"""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
                if token_data.get('expire_time', 0) > time.time():
                    return token_data.get('token')
        except Exception as e:
            logging.debug(f"加载token失败: {str(e)}")
    return None

def _save_token(token: str) -> None:
    """保存token到文件"""
    token_data = {
        'token': token,
        'expire_time': time.time() + 30 * 24 * 3600  # 30 days
    }
    try:
        with open(TOKEN_FILE, 'w') as f:
            json.dump(token_data, f)
        logging.info("Token saved successfully")
    except Exception as e:
        logging.error(f"保存token失败: {str(e)}")

async def _load_cookies(context: BrowserContext) -> None:
    """从文件加载cookies"""
    if os.path.exists(COOKIES_FILE):
        try:
            with open(COOKIES_FILE, 'r') as f:
                cookies = json.load(f)
                for cookie in cookies:
                    if 'domain' not in cookie:
                        cookie['domain'] = '.xiaohongshu.com'
                    if 'path' not in cookie:
                        cookie['path'] = '/'
                await context.add_cookies(cookies)
                logging.info("Cookies loaded successfully")
        except Exception as e:
            logging.error(f"加载cookies失败: {str(e)}")

async def _save_cookies(context: BrowserContext) -> None:
    """保存cookies到文件"""
    try:
        cookies = await context.cookies()
        with open(COOKIES_FILE, 'w') as f:
            json.dump(cookies, f)
        logging.info("Cookies saved successfully")
    except Exception as e:
        logging.error(f"保存cookies失败: {str(e)}")


def create_rednote_content_with_gpt(paper_data: PaperData) -> tuple[str, str]:
    """Create engaging content for Rednote using GPT.
        
    Returns:
        tuple[str, str]: A tuple containing (title, content)
    """
    content_api_key = os.getenv("OPENAI_API_KEY_SUMMARIZER")
    if not content_api_key:
        raise ValueError("OPENAI_API_KEY_SUMMARIZER environment variable not set")
    llm = ChatOpenAI(model="gpt-4", api_key=content_api_key)
    # 更随意一点的小红书笔记模板
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是小红书上的内容达人，专门把学术论文变成超好懂又有趣的小笔记~

    ⚠️ 标题要求（必须严格遵守）：
    - 必须控制在20字以内（包括标点符号）
    - 标题示例（注意字数）：
    ✅ "AI绘画惊艳全网！"（8字）
    ✅ "震惊！AI也懂写诗了"（9字）
    ✅ "让科研效率翻倍的神器"（11字）
    ❌ "人工智能和机器学习如何改变我们的生活"（19字，太长）

    笔记格式：
    1. 标题（与正文空行分隔）

    2. 开场（1–2句）  
    – 像跟朋友聊天，用个小问题或惊叹句吸引注意。

    3. 关键发现（2–3句）  
    – 用简单的话说出 2 点最牛创新／结论，专业术语一定要顺带小科普。

    4. 为什么有用（1–2句）  
    – 贴近生活或行业，说说这项研究怎么改变未来。

    5. 结尾互动（1句）  
    – 抛个问题或话题，让大家评论，“你怎么看？”  

    6. 标签（≥3个，单行）  
    – 每个标签前加"#"，一行写完，比如：
    #人工智能 #深度学习 #学术笔记

    ⚡ 全文控制在100–130字，段落空行，语言轻松、有温度~

    ⚠️ 帖子要包含以下信息：论文标题、作者、期刊/会议、链接（url）
    """),
        ("user", """
    请根据以下信息，写一篇符合上面风格的小红书笔记：

    - 论文标题：{title}  
    - 摘要：{abstract}  
    - 作者：{authors}  
    - 期刊/会议：{publication}  
    - 链接：{url}

    ⚠️ 再次提醒：标题必须控制在20字以内，包括标点符号！
    生成中文回复
    """)
    ])

    chain = prompt | llm
    response = chain.invoke({
        "title": paper_data.title,
        "abstract": paper_data.abstract,
        "authors": paper_data.authors,
        "publication": paper_data.publication,
        "url": paper_data.url or "暂无链接"
    })
    
    # Split response into title and content
    content_parts = response.content.strip().split('\n\n', 1)
    if len(content_parts) != 2:
        logging.warning("Response format unexpected, trying to extract title and content")
        lines = response.content.strip().split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
    else:
        title = content_parts[0].strip()
        content = content_parts[1].strip()
    
    # Verify title length
    if len(title) > 20:
        logging.warning(f"Generated title exceeds 20 characters: {title}")
        title = title[:17] + "..."
    
    return title, content

async def publish_rednote_post_async(paper_data: PaperData) -> RednoteResponse:
    """Publish a post to Rednote (async version)."""
    try:
        # Generate image for the paper first
        generated_image = await generate_paper_image(paper_data)
        if not generated_image:
            return RednoteResponse(success=False, message="Failed to generate paper image")
        
        # Generate content
        try:
            title, content = create_rednote_content_with_gpt(paper_data)
        except Exception as e:
            logging.error(f"Failed to generate content: {str(e)}")
            return RednoteResponse(success=False, message="Failed to generate content")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            # First try cookies
            await _load_cookies(context)
            await page.goto("https://creator.xiaohongshu.com/new/home", wait_until="networkidle")
            
            # If cookies didn't work, try QR code login
            if page.url != "https://creator.xiaohongshu.com/new/home":
                logging.info("Cookie login failed, trying QR code login")
                await page.goto("https://creator.xiaohongshu.com/login")
                await asyncio.sleep(1)
                
                qr_code_button = await page.wait_for_selector(".css-wemwzq")
                if not qr_code_button:
                    await browser.close()
                    return RednoteResponse(success=False, message="QR code button not found")
                await qr_code_button.click()
                
                try:
                    # Wait for login and redirect to home page
                    await page.wait_for_url("https://creator.xiaohongshu.com/new/home", timeout=60000)
                    logging.info("Successfully logged in with QR code")
                    
                    # Save cookies for future use
                    await _save_cookies(context)
                    _save_token("dummy_token")
                except Exception as e:
                    logging.error(f"QR code login failed: {str(e)}")
                    await browser.close()
                    return RednoteResponse(success=False, message="QR code login failed")

            # At this point we should be on the home page
            if page.url != "https://creator.xiaohongshu.com/new/home":
                await browser.close()
                return RednoteResponse(success=False, message="Failed to reach home page after login attempts")

            logging.info("Successfully logged in, creating new post")

            try:
                # Create new post
                create_button = await page.wait_for_selector(".btn.el-tooltip__trigger.el-tooltip__trigger", timeout=50000)
                if not create_button:
                    raise Exception("Could not find create post button")
                await create_button.click()
                
                upload_tab = await page.wait_for_selector(".creator-tab:has(span.title:text('上传图文'))", timeout=50000)
                if not upload_tab:
                    raise Exception("Could not find upload tab")
                await upload_tab.click()
                await asyncio.sleep(1)

                # Upload the generated image
                async with page.expect_file_chooser() as fc_info:
                    await page.click(".upload-input")
                file_chooser = await fc_info.value
                await file_chooser.set_files([generated_image])
                await asyncio.sleep(2)  # Wait longer for image upload

                # Enter title and content separately
                title_input = await page.wait_for_selector(".d-text", timeout=50000)
                if not title_input:
                    raise Exception("Could not find title input field")
                await title_input.fill(title)

                content_input = await page.wait_for_selector(".ql-editor.ql-blank", timeout=50000)
                if not content_input:
                    raise Exception("Could not find content input field")
                await content_input.fill(content)

                # Ensure content is entered before publishing
                await asyncio.sleep(2)

                # Find and click the publish button
                publish_button = await page.wait_for_selector(".d-button-content:has-text('发布')", timeout=50000)
                if not publish_button:
                    raise Exception("Could not find publish button")
                
                # Make sure the button is visible and clickable
                await publish_button.wait_for_element_state('visible')
                await publish_button.scroll_into_view_if_needed()
                await asyncio.sleep(1)
                
                # Click the publish button
                await publish_button.click()
                
                await asyncio.sleep(5)

                await browser.close()
                return RednoteResponse(success=True, message="Post published successfully")

            except Exception as e:
                logging.error(f"Error during post creation: {str(e)}")
                await browser.close()
                return RednoteResponse(success=False, message=f"Failed to create post: {str(e)}")

    except Exception as e:
        logging.error(f"Error publishing to Rednote: {str(e)}")
        return RednoteResponse(success=False, message=f"Error publishing post: {str(e)}")

@function_tool
async def publish_paper_to_rednote(
    title: str,
    abstract: str,
    authors: str,
    publication: str,
    url: str = None,
    pdf_url: str = None,
    images: str = None  # Comma-separated image paths
) -> RednoteResponse:
    """Publish a paper summary to Rednote as a tool-compatible function. Images should be a comma-separated string of file paths."""
    paper_data = PaperData(
        title=title,
        abstract=abstract,
        authors=authors,
        publication=publication,
        url=url,
        pdf_url=pdf_url
    )
    return await publish_rednote_post_async(paper_data)

async def generate_paper_image(paper_data: PaperData, output_dir: str = None) -> str:
    """
    Generate an image for the paper using GPT-4's DALL-E capabilities.
    
    Args:
        paper_data: PaperData object containing paper information
        output_dir: Directory to save the generated image. If None, creates a temp directory.
        
    Returns:
        Path to the generated image
    """
    if output_dir is None:
        # Create a temp directory in the workspace
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "generated_images")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First, get a prompt for DALL-E from GPT-4
        content_api_key = os.getenv("OPENAI_API_KEY_SUMMARIZER")
        if not content_api_key:
            raise ValueError("OPENAI_API_KEY_SUMMARIZER environment variable not set")
        
        llm = ChatOpenAI(model="gpt-4", api_key=content_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
        You're an expert at crafting DALL·E prompts.
        Make a vivid, detailed prompt (≤400 chars) for a social-media-ready image of an academic paper's core ideas. Imagine a beautiful scientist is actively researching this topic—let your creativity shine! Include:

        1. Main subject/theme  
        2. Style (e.g., digital art, scientific illustration)  
        3. Mood/atmosphere  
        4. Specific elements (e.g., lab equipment, data visuals, and the beautiful scientist)
        """),
            ("user", """
        Create a DALL·E prompt for this paper:  
        Title: {title}  
        Abstract: {abstract}
        """)
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "title": paper_data.title,
            "abstract": paper_data.abstract
        })
        
        dalle_prompt = response.content.strip()
        logging.info(f"Generated DALL-E prompt: {dalle_prompt}")
        
        # Generate image using DALL-E
        client = OpenAI(api_key=content_api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Download and save the image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_path = os.path.join(output_dir, f"{paper_data.title[:50]}_generated.png")
        
        with open(image_path, 'wb') as f:
            f.write(image_response.content)
        
        logging.info(f"Generated and saved image: {image_path}")
        return image_path
            
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return None