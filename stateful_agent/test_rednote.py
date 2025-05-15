import asyncio
import os
import json
from tools.rednote_publisher import PaperData, publish_paper_to_rednote

async def test_rednote_posting():
    # Load paper data from JSON file
    json_path = "data/scraped_papers/llm_post-training/papers_20250406_151410_47e37109.json"
    with open(json_path, 'r') as f:
        paper_data = json.load(f)
    
    # Get the first paper from the list
    paper = paper_data['papers'][0]
    
    print(f"Testing paper: {paper['title']}")
    
    # Path to the image in the same directory as this test file
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TesseraQ: Ultra Low-Bit LLM Post-Training Quantiza_generated.png")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Create PaperData object with the existing image
    test_paper = PaperData(
        title=paper['title'],
        abstract=paper['abstract'],
        authors=", ".join(paper['authors']),
        publication=paper['venue'],
        images=[image_path]
    )

    print(f"Testing paper publishing for: {paper['title']}")
    print(f"Using existing image: {image_path}")
    
    result = await publish_paper_to_rednote(test_paper)
    print(f"Publishing result: {result}")

if __name__ == "__main__":
    asyncio.run(test_rednote_posting()) 