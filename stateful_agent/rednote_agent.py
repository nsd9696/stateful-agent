import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from tools.rednote_publisher import RednotePublisher
from tools.paper_scraper import generate_linkedin_post_with_chatgpt

# Load environment variables
load_dotenv()

class RednoteAgent:
    def __init__(self):
        self.publisher = RednotePublisher()
        self.initialized = False

    async def initialize(self):
        """Initialize the Rednote publisher"""
        if not self.initialized:
            await self.publisher.initialize()
            self.initialized = True

    async def post_paper(self, paper_data: Dict[str, Any], images: Optional[List[str]] = None) -> bool:
        """
        Post a paper to Xiaohongshu
        
        Args:
            paper_data: Dictionary containing paper information
            images: Optional list of image paths to include in the post
            
        Returns:
            bool: True if post was successful, False otherwise
        """
        try:
            # Ensure publisher is initialized
            await self.initialize()

            # Generate post content using the existing function
            post_content = await generate_linkedin_post_with_chatgpt(paper_data)
            if not post_content:
                print("Failed to generate post content")
                return False

            # Post the article
            await self.publisher.post_article(
                title=paper_data.get('title', 'Research Paper'),
                content=post_content,
                images=images
            )

            print("Successfully posted to Xiaohongshu")
            return True

        except Exception as e:
            print(f"Error posting to Xiaohongshu: {str(e)}")
            return False

    async def search_and_post_paper(self, research_area: str, title_keywords: Optional[str] = None,
                                  author_name: Optional[str] = None, limit: int = 1) -> bool:
        """
        Search for a paper and post it to Xiaohongshu
        
        Args:
            research_area: Research area to search for
            title_keywords: Optional keywords to match in paper title
            author_name: Optional author name to filter by
            limit: Maximum number of papers to search for
            
        Returns:
            bool: True if post was successful, False otherwise
        """
        try:
            # Search for papers
            from tools.paper_scraper import scrape_papers, PaperSearchRequest
            search_request = PaperSearchRequest(
                research_area=research_area,
                author_name=author_name,
                limit=limit,
                download_pdfs=True
            )
            
            search_response = scrape_papers(search_request)
            
            if not search_response.papers:
                print(f"No papers found for research area: {research_area}")
                return False
            
            # Filter by title keywords if provided
            papers = search_response.papers
            if title_keywords:
                papers = [p for p in papers if title_keywords.lower() in p.get("title", "").lower()]
                
            if not papers:
                print(f"No papers found matching title keywords: {title_keywords}")
                return False
            
            # Post the first matching paper
            paper_data = papers[0]
            return await self.post_paper(paper_data)

        except Exception as e:
            print(f"Error searching and posting paper: {str(e)}")
            return False

    async def close(self):
        """Close the Rednote publisher"""
        try:
            await self.publisher.close(force=True)
            self.initialized = False
        except Exception as e:
            print(f"Error closing Rednote publisher: {str(e)}")

# Example usage
async def main():
    # Initialize the agent
    agent = RednoteAgent()
    await agent.initialize()

    # Example paper data
    paper_data = {
        'title': 'Example Paper Title',
        'abstract': 'This is an example paper abstract...',
        'authors': ['Author 1', 'Author 2'],
        'url': 'https://example.com/paper',
        'published_date': '2024-01-01'
    }

    # Post the paper
    success = await agent.post_paper(paper_data)
    if success:
        print("Paper posted successfully")
    else:
        print("Failed to post paper")

    # Example of searching and posting
    success = await agent.search_and_post_paper(
        research_area="machine learning",
        title_keywords="deep learning",
        limit=1
    )
    if success:
        print("Paper found and posted successfully")
    else:
        print("Failed to find and post paper")

    # Close the agent
    await agent.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 