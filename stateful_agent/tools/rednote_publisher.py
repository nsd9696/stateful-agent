from typing import List, Optional, Dict, Any
import os
import json
import logging
import asyncio
from playwright.async_api import async_playwright
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_path = os.path.expanduser('~/Desktop/rednote_error.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG)

class RednotePublisher:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.token = None
        self.token_file = None
        self.cookies_file = None

    async def initialize(self):
        """Initialize the browser and set up necessary configurations"""
        if self.playwright is not None:
            return

        try:
            logging.info("Initializing Playwright...")
            self.playwright = await async_playwright().start()

            # Configure browser launch arguments
            launch_args = {
                'headless': False,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-infobars',
                    '--start-maximized',
                    '--ignore-certificate-errors',
                    '--ignore-ssl-errors'
                ]
            }

            # Launch browser
            self.browser = await self.playwright.chromium.launch(**launch_args)
            self.context = await self.browser.new_context(permissions=['geolocation'])
            self.page = await self.context.new_page()

            # Set up token and cookies file paths
            home_dir = os.path.expanduser('~')
            app_dir = os.path.join(home_dir, '.rednote_system')
            if not os.path.exists(app_dir):
                os.makedirs(app_dir)

            self.token_file = os.path.join(app_dir, "rednote_token.json")
            self.cookies_file = os.path.join(app_dir, "rednote_cookies.json")
            self.token = self._load_token()
            await self._load_cookies()

            logging.info("Browser initialized successfully")

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            await self.close(force=True)
            raise

    def _load_token(self) -> Optional[str]:
        """Load token from file"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                    if token_data.get('expire_time', 0) > time.time():
                        return token_data.get('token')
            except Exception as e:
                logging.error(f"Error loading token: {str(e)}")
        return None

    def _save_token(self, token: str):
        """Save token to file"""
        token_data = {
            'token': token,
            'expire_time': time.time() + 30 * 24 * 3600  # 30 days
        }
        try:
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
        except Exception as e:
            logging.error(f"Error saving token: {str(e)}")

    async def _load_cookies(self):
        """Load cookies from file"""
        if os.path.exists(self.cookies_file):
            try:
                with open(self.cookies_file, 'r') as f:
                    cookies = json.load(f)
                    for cookie in cookies:
                        if 'domain' not in cookie:
                            cookie['domain'] = '.xiaohongshu.com'
                        if 'path' not in cookie:
                            cookie['path'] = '/'
                    await self.context.add_cookies(cookies)
            except Exception as e:
                logging.error(f"Error loading cookies: {str(e)}")

    async def _save_cookies(self):
        """Save cookies to file"""
        try:
            cookies = await self.context.cookies()
            with open(self.cookies_file, 'w') as f:
                json.dump(cookies, f)
        except Exception as e:
            logging.error(f"Error saving cookies: {str(e)}")

    async def login(self):
        """Login to Xiaohongshu"""
        await self.ensure_browser()

        # Check if already logged in
        await self.page.goto("https://creator.xiaohongshu.com/new/home", wait_until="networkidle")
        current_url = self.page.url

        if "creator.xiaohongshu.com/new/home" in current_url:
            logging.info("Already logged in")
            return

        # Try cookie login if token exists
        if self.token:
            logging.info("Attempting cookie login")
            await self.page.goto("https://creator.xiaohongshu.com/login", wait_until="networkidle")
            await self.context.clear_cookies()
            await self._load_cookies()
            await self.page.reload(wait_until="networkidle")

            current_url = self.page.url
            if "login" not in current_url:
                logging.info("Cookie login successful")
                self.token = self._load_token()
                await self._save_cookies()
                return

        # QR code login
        logging.info("Starting QR code login")
        await self.page.goto("https://creator.xiaohongshu.com/login")
        await asyncio.sleep(1)

        qr_code = await self.page.wait_for_selector(".qrcode-img img")
        if not qr_code:
            raise Exception("QR code not found")

        qr_code_url = await qr_code.get_attribute("src")
        if not qr_code_url:
            raise Exception("QR code URL not found")

        response = requests.get(qr_code_url)
        if response.status_code != 200:
            raise Exception("Failed to download QR code")

        qr_code_path = "qr_code.png"
        with open(qr_code_path, "wb") as f:
            f.write(response.content)

        logging.info("Please scan the QR code to login")
        os.system(f"open {qr_code_path}")

        try:
            await self.page.wait_for_url("**/creator/home", timeout=60000)
            logging.info("QR code login successful")
            self.token = self._load_token()
            await self._save_cookies()
            os.remove(qr_code_path)
        except Exception as e:
            logging.error(f"QR code login failed: {str(e)}")
            raise

    async def post_article(self, title: str, content: str, images: Optional[List[str]] = None):
        """Post an article to Xiaohongshu"""
        await self.ensure_browser()

        try:
            # Click publish button
            await self.page.click(".btn.el-tooltip__trigger.el-tooltip__trigger")

            # Determine content type
            is_video = False
            if images and len(images) > 0:
                first_file = images[0]
                if isinstance(first_file, str):
                    if 'video' in first_file.lower() or first_file.lower().endswith('.mp4'):
                        is_video = True
                        logging.info("Detected video content")

            # Select appropriate upload tab
            if is_video:
                upload_tab = await self.page.wait_for_selector(".creator-tab:has(span.title:text('上传视频'))", timeout=5000)
            else:
                upload_tab = await self.page.wait_for_selector(".creator-tab:has(span.title:text('上传图文'))", timeout=5000)

            if upload_tab:
                await upload_tab.click()
                await asyncio.sleep(1)
            else:
                raise Exception(f"Could not find {'video' if is_video else 'image'} upload tab")

            # Upload media if provided
            if images:
                async with self.page.expect_file_chooser() as fc_info:
                    await self.page.click(".upload-input")
                file_chooser = await fc_info.value
                await file_chooser.set_files(images)
                await asyncio.sleep(1)

            # Enter title
            title_input = await self.page.wait_for_selector(".d-text", timeout=5000)
            if not title_input:
                raise Exception("Could not find title input field")
            await title_input.fill(title)

            # Enter content
            content_input = await self.page.wait_for_selector(".ql-editor.ql-blank", timeout=5000)
            if not content_input:
                raise Exception("Could not find content input field")
            await content_input.fill(content)

            # Publish
            await asyncio.sleep(1)
            publish_button = await self.page.wait_for_selector(".d-button-content:has-text('发布')", timeout=5000)
            if not publish_button:
                raise Exception("Could not find publish button")
            await publish_button.click()
            logging.info("Article published successfully")

        except Exception as e:
            logging.error(f"Error publishing article: {str(e)}")
            raise

    async def ensure_browser(self):
        """Ensure browser is initialized"""
        if not self.playwright:
            await self.initialize()

    async def close(self, force: bool = False):
        """Close browser and cleanup"""
        try:
            if force:
                if self.context:
                    await self.context.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()
                self.playwright = None
                self.browser = None
                self.context = None
                self.page = None
        except Exception as e:
            logging.error(f"Error closing browser: {str(e)}") 