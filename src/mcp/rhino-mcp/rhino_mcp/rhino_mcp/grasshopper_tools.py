"""Tools for interacting with Grasshopper through socket connection."""
from mcp.server.fastmcp import FastMCP, Context, Image
import logging
from typing import Dict, Any, List, Optional, Union
import json
import socket
import time
import base64
import io
from PIL import Image as PILImage
import requests
import re
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable insecure HTTPS warnings
urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
logger = logging.getLogger("GrasshopperTools")

# Add a preprocessing function for LLM inputs
def preprocess_llm_input(input_str: str) -> str:
    """
    Preprocess a potentially malformed JSON string from an LLM.
    This handles common issues before attempting JSON parsing.

    Args:
        input_str: Raw string from LLM that may contain malformed JSON

    Returns:
        Preprocessed string that should be easier to parse
    """
    if not isinstance(input_str, str):
        return input_str

    # Replace backtick delimiters with proper double quotes for the entire JSON object
    if input_str.strip().startswith('`{') and input_str.strip().endswith('}`'):
        input_str = input_str.strip()[1:-1]  # Remove the outer backticks

    # Handle backtick-delimited field names and string values
    # This is a basic approach - first convert all standalone backtick pairs to double quotes
    result = ""
    in_string = False
    last_char = None
    i = 0
    
    while i < len(input_str):
        char = input_str[i]
        
        # Handle backtick as quote
        if char == '`' and (last_char is None or last_char != '\\'):
            in_string = not in_string
            result += '"'
        else:
            result += char
            
        last_char = char
        i += 1

    # Fix boolean values
    result = re.sub(r':\s*True\b', ': true', result)
    result = re.sub(r':\s*False\b', ': false', result)
    result = re.sub(r':\s*None\b', ': null', result)
    
    return result

def extract_payload_fields(raw_input: str) -> Dict[str, Any]:
    """
    Extract fields from a payload that might be malformed.
    Works with raw LLM output directly.
    
    Args:
        raw_input: Raw string input from LLM
        
    Returns:
        Dictionary of extracted fields
    """
    if not isinstance(raw_input, str):
        return {}
    
    # First attempt: try the standard JSON sanitizer
    payload = sanitize_json(raw_input)
    if payload:
        return payload
    
    # Second attempt: special handling for backtick-delimited code
    if '`code`' in raw_input or '"code"' in raw_input:
        # Find the code section
        code_match = re.search(r'[`"]code[`"]\s*:\s*[`"](.*?)[`"](?=\s*,|\s*\})', raw_input, re.DOTALL)
        instance_guid_match = re.search(r'[`"]instance_guid[`"]\s*:\s*[`"](.*?)[`"]', raw_input)
        message_match = re.search(r'[`"]message_to_user[`"]\s*:\s*[`"](.*?)[`"]', raw_input)
        
        result = {}
        
        if instance_guid_match:
            result["instance_guid"] = instance_guid_match.group(1)
            
        if code_match:
            result["code"] = code_match.group(1)
            
        if message_match:
            result["message_to_user"] = message_match.group(1)
            
        return result
        
    return {}

# Update the sanitize_json function to use the preprocessor
def sanitize_json(json_str_or_dict: Union[str, Dict]) -> Dict[str, Any]:
    """
    Sanitize and validate JSON input, which might come from an LLM.
    
    Args:
        json_str_or_dict: Either a JSON string or dictionary that might need sanitizing
        
    Returns:
        A properly formatted dictionary
    """
    # If it's already a dictionary, return it
    if isinstance(json_str_or_dict, dict):
        return json_str_or_dict.copy()
    
    # If it's a string, try to fix common issues
    if isinstance(json_str_or_dict, str):
        # Apply preprocessing for LLM input
        json_str = preprocess_llm_input(json_str_or_dict)
        
        # Remove markdown JSON code block markers if present
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        
        # Try to parse the JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after preprocessing: {e}")
            logger.error(f"Preprocessed JSON string: {json_str}")
            
            # Try another approach - remove all newlines from outside code sections
            try:
                # Find code sections
                if '"code"' in json_str:
                    parts = []
                    last_end = 0
                    
                    # Find all code sections
                    for match in re.finditer(r'"code"\s*:\s*"(.*?)"(?=\s*,|\s*\})', json_str, re.DOTALL):
                        # Add the part before code with newlines removed
                        before_code = json_str[last_end:match.start()]
                        before_code = re.sub(r'\s+', ' ', before_code)
                        parts.append(before_code)
                        
                        # Add the code section as is
                        code_section = match.group(0)
                        parts.append(code_section)
                        
                        last_end = match.end()
                    
                    # Add the remaining part
                    remaining = json_str[last_end:]
                    remaining = re.sub(r'\s+', ' ', remaining)
                    parts.append(remaining)
                    
                    # Combine all parts
                    json_str = ''.join(parts)
                    
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON with alternative method")
            
            # Return empty dict as fallback
            return {}
    
    # If it's neither a dict nor string, return empty dict
    return {}

class GrasshopperConnection:
    def __init__(self, host='localhost', port=9999):  # Using port 9999 to match gh_socket_server.py
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = 30.0  # 30 second timeout
    
    def check_server_available(self) -> bool:
        """Check if the Grasshopper server is running and available.
        
        Returns:
            bool: True if the server is available, False otherwise
        """
        try:
            response = requests.get(self.base_url, timeout=2.0)
            response.raise_for_status()
            logger.info("Grasshopper server is available at {0}".format(self.base_url))
            return True
        except Exception as e:
            logger.warning("Grasshopper server is not available: {0}".format(str(e)))
            return False
    
    def connect(self):
        """Connect to the Grasshopper script's HTTP server"""
        # Check if server is available
        if not self.check_server_available():
            raise Exception("Grasshopper server not available at {0}. Make sure the GHPython component is running and the toggle is set to True.".format(self.base_url))
        logger.info("Connected to Grasshopper server")
    
    def disconnect(self):
        """No need to disconnect for HTTP connections"""
        pass
    
    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to the Grasshopper script and wait for response"""
        try:
            data = {
                "type": command_type,
                **(params or {})
            }

            logger.info(f"Sending command to Grasshopper server: type={command_type}")
            
            # Use a session to handle connection properly
            with requests.Session() as session:
                response = session.post(
                    self.base_url,
                    json=data,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'},
                    stream=True
                )
                response.raise_for_status()
                
                # Read the response content and return it directly
                return response.json()
                    
        except requests.exceptions.RequestException as req_err:
            error_content = ""
            if hasattr(req_err, 'response') and req_err.response is not None:
                try:
                    error_content = req_err.response.text
                except:
                    pass
                
            error_msg = f"HTTP request error: {str(req_err)}. Response: {error_content}"
            logger.error(error_msg)
            return {"status": "error", "result": error_msg}
            
        except Exception as e:
            error_msg = f"Error communicating with Grasshopper script: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "result": error_msg}

# Global connection instance
_grasshopper_connection = None

def get_grasshopper_connection() -> GrasshopperConnection:
    """Get or create the Grasshopper connection"""
    global _grasshopper_connection
    if _grasshopper_connection is None:
        _grasshopper_connection = GrasshopperConnection()
    return _grasshopper_connection

class GrasshopperTools:
    """Collection of tools for interacting with Grasshopper."""
    
    def __init__(self, app):
        self.app = app
        self._register_tools()
    
    def _register_tools(self):
        """Register all Grasshopper tools with the MCP server."""
        # self.app.tool()(self.is_server_available)
        self.app.tool()(self.execute_code_in_gh)
        self.app.tool()(self.get_gh_context)
        self.app.tool()(self.get_objects)
        self.app.tool()(self.get_selected)
        self.app.tool()(self.update_script)
        self.app.tool()(self.update_script_with_code_reference)
        self.app.tool()(self.expire_and_get_info)
    
    # def is_server_available(self, ctx: Context) -> bool:
    #     """Grasshopper: Check if the Grasshopper server is available.
        
    #     This is a quick check to see if the Grasshopper socket server is running
    #     and available for connections.
        
    #     Returns:
    #         bool: True if the server is available, False otherwise
    #     """
    #     try:
    #         connection = get_grasshopper_connection()
    #         return connection.check_server_available()
    #     except Exception as e:
    #         logger.error("Error checking Grasshopper server availability: {0}".format(str(e)))
    #         return False
    
    def execute_code_in_gh(self, ctx: Context, code: str) -> str:
        """Grasshopper: Execute arbitrary Python code in Grasshopper.
        
        IMPORTANT: 
        - Uses IronPython 2.7 - no f-strings or modern Python features
        - Always include ALL required imports in your code
        - Use 'result = value' to return data (don't use return statements)

        Example - Adding components to canvas:
        ```python
        import scriptcontext as sc
        import clr
        import Rhino
        import System.Drawing as sd
        import Grasshopper
        import Grasshopper.Kernel.Special as GHSpecial

        doc = ghenv.Component.OnPingDocument()
        
        # Create and position a Pipeline
        pipe = GHSpecial.GH_GeometryPipeline()
        if pipe.Attributes is None: pipe.CreateAttributes()
        pipe.Attributes.Pivot = sd.PointF(100, 100)
        doc.AddObject(pipe, False)

        # Create and connect a Panel
        pan = GHSpecial.GH_Panel()
        if pan.Attributes is None: pan.CreateAttributes()
        pan.Attributes.Pivot = sd.PointF(300, 100)
        doc.AddObject(pan, False)
        pan.AddSource(pipe)
        
        result = "Components created successfully"
        ```
        
        You can also provide the code as part of a JSON object with a "code" field.
        
        Args:
            code: The Python code to execute, or a JSON object with a "code" field
        
        Returns:
            The result of the code execution
        """
        try:
            # Check if the input might be a JSON payload
            if isinstance(code, str) and (
                code.strip().startswith('{') or 
                code.strip().startswith('`{') or
                '`code`' in code or 
                '"code"' in code
            ):
                # Try direct extraction for speed and reliability
                payload = extract_payload_fields(code)
                if payload and "code" in payload:
                    code = payload["code"]
            
            # Validate that we have code to execute
            if not code or not isinstance(code, str):
                return "Error: No valid code provided. Please provide Python code to execute."
            
            # Make sure code ends with a result variable if it doesn't have one
            if "result =" not in code and "result=" not in code:
                # Extract the last line if it starts with "return"
                lines = code.strip().split("\n")
                if lines and lines[-1].strip().startswith("return "):
                    return_value = lines[-1].strip()[7:].strip() # Remove "return " prefix
                    # Replace the return with a result assignment
                    lines[-1] = "result = " + return_value
                    code = "\n".join(lines)
                else:
                    # Append a default result if no return or result is present
                    code += "\n\n# Auto-added result assignment\nresult = \"Code executed successfully\""
            
            logger.info(f"Sending code execution request to Grasshopper")
            connection = get_grasshopper_connection()
            
            result = connection.send_command("execute_code", {
                "code": code
            })
            
            # Simply return result info with error prefix if needed
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return result.get("result", "Code executed successfully")
                
        except Exception as e:
            return f"Error executing code: {str(e)}"

    def get_gh_context(self, ctx: Context, simplified: bool = False) -> str:
        """Grasshopper: Get current Grasshopper document state and definition graph, sorted by execution order.
        
        Returns a JSON string containing:
        - Component graph (connections between components)
        - Component info (guid, name, type)
        - Component properties and parameters
        
        Args:
            simplified: When true, returns minimal component info without detailed properties
        
        Returns:
            JSON string with grasshopper definition graph
        """
        try:
            logger.info("Getting Grasshopper context with simplified={0}".format(simplified))
            connection = get_grasshopper_connection()
            result = connection.send_command("get_context", {
                "simplified": simplified
            })
            
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return json.dumps(result.get("result", {}), indent=2)
                
        except Exception as e:
            return f"Error getting context: {str(e)}"

    def get_objects(self, ctx: Context, instance_guids: List[str], simplified: bool = False, context_depth: int = 0) -> str:
        """Grasshopper: Get information about specific components by their GUIDs.
        
        Args:
            instance_guids: List of component GUIDs to retrieve
            simplified: When true, returns minimal component info
            context_depth: How many levels of connected components to include (0-3), try to keep it small
        
        Returns:
            JSON string with component information and optional context
        """
        try:
            logger.info("Getting objects with GUIDs: {0}".format(instance_guids))
            connection = get_grasshopper_connection()
            result = connection.send_command("get_objects", {
                "instance_guids": instance_guids,
                "simplified": simplified,
                "context_depth": context_depth
            })
            
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return json.dumps(result.get("result", {}), indent=2)
                
        except Exception as e:
            return f"Error getting objects: {str(e)}"

    def get_selected(self, ctx: Context, simplified: bool = False, context_depth: int = 0) -> str:
        """Grasshopper: Get information about currently selected components.
        
        Args:
            simplified: When true, returns minimal component info
            context_depth: How many levels of connected components to include (0-3)
        
        Returns:
            JSON string with selected component information and optional context
        """
        try:
            logger.info("Getting selected components")
            connection = get_grasshopper_connection()
            result = connection.send_command("get_selected", {
                "simplified": simplified,
                "context_depth": context_depth
            })
            
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return json.dumps(result.get("result", {}), indent=2)
                
        except Exception as e:
            return f"Error getting selected components: {str(e)}"

    def update_script(self, ctx: Context, instance_guid: str = None, code: str = None, description: str = None, 
                     message_to_user: str = None, param_definitions: List[Dict[str, Any]] = None) -> str:
        """Grasshopper: Update a script component with new code, description, user feedback message, and optionally redefine its parameters.
        
        IMPORTANT NOTES:
        0. the output param "output" is reserved for the "message_to_user", name output params with a meaningful name if you create new ones
        1. The code must be valid Python 3 / IronPython code (no f-strings!)
        2. When updating existing code:
           - If NOT changing parameters, ensure to keep the same input/output variable names!
           - Know their datatypes and access methods (list, datatree, item) before modifying
           - The script may be part of a larger definition - maintain input/output structure
        3. When changing input and outputparameters:
           - You must provide ALL input/output parameters, even existing ones you want to keep
           - The component will be completely reconfigured with the new parameter set
           - Existing connections may be lost if parameter names change
        
        Example:
        ```json
        {
            "instance_guid": "a1b2c3d4-e5f6-4a5b-9c8d-7e6f5d4c3b2a",
            "code": "import Rhino.Geometry as rg\\n\\n# Create circle from radius\\norigin = rg.Point3d(0, 0, 0)\\ncircle = rg.Circle(origin, radius)\\n\\n# Set outputs\\nresult = circle\\ncircle_center = circle.Center\\ncircle_area = circle.Area",
            "description": "Creates a circle and outputs its geometry, center point, and area",
            "message_to_user": "Circle component updated with new outputs for center point and area calculation",
            "param_definitions": [
                {
                    "type": "input",
                    "name": "radius",
                    "access": "item",
                    "typehint": "float",
                    "description": "Circle radius",
                    "optional": false,
                    "default": 1.0
                },
                {
                    "type": "output",
                    "name": "circle",
                    "description": "Generated circle geometry"
                },
                {
                    "type": "output",
                    "name": "center",
                    "description": "Center point of the circle"
                },
                {
                    "type": "output",
                    "name": "output",
                    "description": "Used to display messages to the user"
                }
            ]
        }
        ```
        
        Args:
            instance_guid: The GUID of the script component to update
            code: Optional new Python code for the component
            description: Optional new description for the component
            message_to_user: Optional feedback message that should include a change summary and/or suggestions
            param_definitions: Optional list of parameter definitions. If provided, ALL parameters will be redefined.
                Each definition must be a dictionary with:
                Required keys:
                    - "type": "input" or "output"
                    - "name": Parameter name (string)
                Optional keys for inputs:
                    - "access": "item", "list", or "tree" (default "list")
                    - "typehint": e.g. "str", "int", "float", "bool" (determines parameter type)
                    - "description": Parameter description
                    - "optional": bool, default True
                    - "default": Default value (not persistent)
        
        Returns:
            Success status with summary of which elements were updated
        """
        try:
            # Log initial input for debugging
            if isinstance(instance_guid, str) and len(instance_guid) > 200:
                logger.info(f"Received long payload as instance_guid parameter: first 100 chars: {instance_guid[:100]}...")
            else:
                logger.info(f"Initial parameters: instance_guid={instance_guid}, code length={len(code) if code else 0}, "
                          f"description={'provided' if description else 'None'}, "
                          f"message_to_user={'provided' if message_to_user else 'None'}, "
                          f"param_definitions={'provided' if param_definitions else 'None'}")
            
            # Check if the first argument is a string that looks like a JSON payload
            if isinstance(instance_guid, str) and (
                instance_guid.strip().startswith('{') or 
                instance_guid.strip().startswith('`{') or 
                '`instance_guid`' in instance_guid or 
                '"instance_guid"' in instance_guid
            ):
                logger.info("Detected JSON-like payload in instance_guid parameter, extracting fields")
                # More robust extraction for complex payloads
                payload = extract_payload_fields(instance_guid)
                if payload and "instance_guid" in payload:
                    # Log what was extracted
                    logger.info(f"Extracted fields from payload: {sorted(payload.keys())}")
                    
                    instance_guid = payload.get("instance_guid")
                    code = payload.get("code", code)
                    description = payload.get("description", description)
                    message_to_user = payload.get("message_to_user", message_to_user)
                    param_definitions = payload.get("param_definitions", param_definitions)
                    
                    logger.info(f"After extraction: instance_guid={instance_guid}, code length={len(code) if code else 0}")
                else:
                    logger.warning("Failed to extract instance_guid from payload")
            
            # Ensure we have a valid instance_guid
            if not instance_guid:
                logger.error("No instance_guid provided")
                return "Error: No instance_guid provided. Please specify the GUID of the script component to update."
            
            logger.info(f"Updating script component {instance_guid}")
            logger.info(f"Parameter details: code={bool(code)}, description={bool(description)}, "
                      f"message_to_user={bool(message_to_user)}, param_definitions type={type(param_definitions) if param_definitions else None}")
                      
            connection = get_grasshopper_connection()
            
            # Sanitize param_definitions if provided
            if param_definitions is not None and isinstance(param_definitions, list):
                # Create new sanitized list
                sanitized_params = []
                for param in param_definitions:
                    if isinstance(param, dict):
                        sanitized_params.append(param.copy())
                    else:
                        # Try to parse if it's a string
                        try:
                            if isinstance(param, str):
                                param_dict = json.loads(preprocess_llm_input(param))
                                sanitized_params.append(param_dict)
                        except:
                            logger.warning(f"Could not parse parameter definition: {param}")
                
                param_definitions = sanitized_params
            
            # Prepare the command payload - log it before sending
            command_payload = {
                "instance_guid": instance_guid,
                "code": code,
                "description": description,
                "message_to_user": message_to_user,
                "param_definitions": param_definitions
            }
            
            logger.info(f"Sending command with payload keys: {sorted(command_payload.keys())}")
            if code:
                logger.info(f"Code snippet (first 50 chars): {code[:50]}...")
            
            # Always use "update_script" as the command type
            result = connection.send_command("update_script", command_payload)
            
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return json.dumps(result.get("result", {}), indent=2)
                
        except Exception as e:
            return f"Error updating script: {str(e)}"

    def update_script_with_code_reference(self, ctx: Context, instance_guid: str = None, file_path: str = None, 
                                        param_definitions: List[Dict[str, Any]] = None, description: str = None, 
                                        name: str = None, force_code_reference: bool = False) -> str:
        """Grasshopper: Update a script component to use code from an external Python file.
        This tool allows you to modify a GHPython script component to use code from an external Python file 
        instead of embedded code. This enables better code organization, version control, and reuse across 
        multiple components. Moreove, you can add and remove input/ output paramters.
        
        important notes:
        1. Only use when working in/with curser or another IDE
        2. First, check the grasshopper script component using  "get_objects" tool
        3. Second, check if a python file is already referenced by the component AND if it exists in the cursor project
            ALWAYS add the component instance_guid to the file name (e.g. cirler_creator_a1b2c3d4-e5f6-4a5b-9c8d-7e6f5d4c3b2a.py)
        4. write code im the file and save it, update the file path with this tool
        5. Once referenced, future updates on the code file will automatically be reflected in the component (no need to use this tool)
        6. you can use get_objects tool to get potential error messages from the component for debugging (runtimeMessages)

        Args:
            instance_guid: The GUID of the target GHPython component to modify.
            file_path: Path to the external Python file that contains the code.
            param_definitions: List of dictionaries defining input/output parameters.
            description: New description for the component.
            name: New nickname for the component.
            force_code_reference: When True, converts/sets a component to use referenced code mode.

        Returns:
            Success status with summary of which elements were updated and component instance_guid
        
        Example:
        ```json
        {
            "instance_guid": "a1b2c3d4-e5f6-4a5b-9c8d-7e6f5d4c3b2a",
            "file_path": "/scripts/cirler_creator_a1b2c3d4-e5f6-4a5b-9c8d-7e6f5d4c3b2a.py"
            "name":"CircleTool"
            "description": "Creates a circle and outputs its geometry, center point, and area",
            "message_to_user": "Circle, add one radius slider as input",
            force_code_reference = True, 
            "param_definitions": [
                {
                    "type": "input",
                    "name": "radius",
                    "access": "item",
                    "typehint": "float",
                    "description": "Circle radius",
                    "optional": false,
                    "default": 1.0
                },
                {
                    "type": "output",
                    "name": "circle",
                    "description": "Generated circle geometry"
                }
            ]
        }
        ```
        """
        try:
            # Ensure we have a valid instance_guid
            if not instance_guid:
                return "Error: No instance_guid provided. Please specify the GUID of the script component to update."
            
            connection = get_grasshopper_connection()
            
            # Prepare the command payload
            command_payload = {
                "instance_guid": instance_guid,
                "file_path": file_path,
                "param_definitions": param_definitions,
                "description": description,
                "name": name,
                "force_code_reference": force_code_reference
            }
            
            # Send command and get result
            result = connection.send_command("update_script_with_code_reference", command_payload)
            
            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            return json.dumps(result.get("result", {}), indent=2)
                
        except Exception as e:
            return f"Error updating script with code reference: {str(e)}"

    def expire_and_get_info(self, ctx: Context, instance_guid: str) -> str:
        """Grasshopper: Expire a specific component and get its updated information.

        This is useful after updating a component's code, especially via a referenced file,
        to force a recompute and retrieve the latest state, including potential errors or messages.

        Args:
            instance_guid: The GUID of the component to expire and query.

        Returns:
            JSON string with the component's updated information after expiration.
        """
        try:
            if not instance_guid:
                return "Error: No instance_guid provided. Please specify the GUID of the component to expire."

            logger.info(f"Expiring component and getting info for GUID: {instance_guid}")
            connection = get_grasshopper_connection()
            result = connection.send_command("expire_component", {
                "instance_guid": instance_guid
            })

            if result.get("status") == "error":
                return f"Error: {result.get('result', 'Unknown error')}"
            # The server side already returns component info after expiring
            return json.dumps(result.get("result", {}), indent=2)

        except Exception as e:
            return f"Error expiring component: {str(e)}"
