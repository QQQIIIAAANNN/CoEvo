"""Tools for web search and email functionality via n8n workflows."""
from mcp.server.fastmcp import FastMCP, Context, Image
import logging
from typing import Dict, Any, Optional, Union, List
import json
import requests
import uuid
from datetime import datetime
import base64
from io import BytesIO
import time
from PIL import Image as PILImage
import traceback

# Configure logging
logger = logging.getLogger("UtilityTools")

class UtilityTools:
    """Collection of utility tools that interface with n8n workflows."""
    
    def __init__(self, app):
        self.app = app
        self._register_tools()
        
        # Configuration for n8n webhooks
        self.web_search_webhook_url = "https://run8n.xyz/webhook/webSearchAgent"
        self.email_webhook_url = "https://run8n.xyz/webhook/gmailAgent"
        self.auth_token = "abc123secretvalue"  # This should be in environment variables in production
    
    def _register_tools(self):
        """Register all utility tools with the MCP server."""
        self.app.tool()(self.web_search)
        self.app.tool()(self.email_tool)
    
    def _generate_session_id(self):
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000))[-3:]

    def _download_image(self, url, max_size=800, jpeg_quality=80):
        """Download and process an image from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Open image from bytes
            img = PILImage.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Calculate new dimensions while maintaining aspect ratio
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # Resize if needed
            if ratio < 1:
                img = img.resize(new_size, PILImage.Resampling.LANCZOS)
            
            # Save as JPEG to BytesIO
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                "data": image_base64,
                "format": "jpeg",
                "width": img.width,
                "height": img.height
            }
            
        except Exception as e:
            logger.error(f"Error downloading/processing image from {url}: {str(e)}")
            return None
    
    def _parse_search_response(self, response_data: Dict[str, Any], download_images: bool = False) -> Dict[str, Any]:
        """Parse the search response from n8n webhook."""
        try:
            # Extract data from response
            output = json.loads(response_data.get("output", "{}"))
            
            # Create result dictionary
            result = {
                "summary": output.get("shortSummary", "No summary available"),
                "report": output.get("searchResultReport", "No detailed report available"),
                "sources": output.get("sources", []),
                "image_urls": output.get("imageUrl", []),
                "images": []  # Will be populated if download_images is True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing search response: {str(e)}")
            return {"status": "error", "message": f"Failed to parse response: {str(e)}"}
    
    def web_search(self, ctx: Context, user_intent: str, download_images: bool = False) -> str:
        """Perform a web search via n8n webhook.
        
        Args:
            user_intent: The search query or intent
            download_images: Whether to download and process images
        
        Returns:
            Search results in markdown format
        """
        try:
            session_id = self._generate_session_id()
            
            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "userIntent": user_intent,
                "downloadImages": download_images,
                "sessionId": session_id
            }
            
            response = requests.get(
                self.web_search_webhook_url,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # If images need to be downloaded
            if download_images and "imageUrls" in result:
                images = []
                for url in result["imageUrls"]:
                    image_data = self._download_image(url)
                    if image_data:
                        images.append(image_data)
                result["images"] = images
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def email_tool(self, ctx: Context, user_intent: str) -> str:
        """Search and interact with emails via n8n webhook.
        
        Args:
            user_intent: The email search query or intent
        
        Returns:
            Email results in markdown format
        """
        try:
            session_id = self._generate_session_id()
            
            headers = {
                "Authorization": self.auth_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "userIntent": user_intent,
                "sessionId": session_id
            }
            
            response = requests.get(
                self.email_webhook_url,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            
            # Return the markdown response directly
            return response.text
            
        except Exception as e:
            error_msg = f"Error searching emails: {str(e)}"
            logger.error(error_msg)
            return error_msg 