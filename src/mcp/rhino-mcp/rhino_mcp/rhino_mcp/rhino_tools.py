"""Tools for interacting with Rhino through socket connection."""
from mcp.server.fastmcp import FastMCP, Context, Image
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional, Union
import json
import socket
import time
import base64
import io
from PIL import Image as PILImage
import os


# Configure logging
logger = logging.getLogger("RhinoTools")

class RhinoConnection:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.socket = None
        self.timeout = 30.0  # 30 second timeout
        self.buffer_size = 14485760  # 10MB buffer size for handling large images
    
    def connect(self):
        """Connect to the Rhino script's socket server"""
        if self.socket is None:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))
                logger.info("Connected to Rhino script")
            except Exception as e:
                logger.error("Failed to connect to Rhino script: {0}".format(str(e)))
                self.disconnect()
                raise
    
    def disconnect(self):
        """Disconnect from the Rhino script"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to the Rhino script and wait for response"""
        if self.socket is None:
            self.connect()
        
        try:
            # Prepare command
            command = {
                "type": command_type,
                "params": params or {}
            }
            
            # Send command
            command_json = json.dumps(command)
            logger.info("Sending command: {0}".format(command_json))
            self.socket.sendall(command_json.encode('utf-8'))
            
            # Receive response with timeout and larger buffer
            buffer = b''
            start_time = time.time()
            
            while True:
                try:
                    # Check timeout
                    if time.time() - start_time > self.timeout:
                        raise Exception("Response timeout after {0} seconds".format(self.timeout))
                    
                    # Receive data
                    data = self.socket.recv(self.buffer_size)
                    if not data:
                        break
                        
                    buffer += data
                    logger.debug("Received {0} bytes of data".format(len(data)))
                    
                    # Try to parse JSON
                    try:
                        response = json.loads(buffer.decode('utf-8'))
                        logger.info("Received complete response: {0}".format(response))
                        
                        # Check for error response
                        if response.get("status") == "error":
                            raise Exception(response.get("message", "Unknown error from Rhino"))
                            
                        return response
                    except json.JSONDecodeError:
                        # If we have a complete response, it should be valid JSON
                        if len(buffer) > 0:
                            continue
                        else:
                            raise Exception("Invalid JSON response from Rhino")
                            
                except socket.timeout:
                    raise Exception("Socket timeout while receiving response")
                    
            raise Exception("Connection closed by Rhino script")
            
        except Exception as e:
            logger.error("Error communicating with Rhino script: {0}".format(str(e)))
            self.disconnect()  # Disconnect on error to force reconnection
            raise

# Global connection instance
_rhino_connection = None

def get_rhino_connection() -> RhinoConnection:
    """Get or create the Rhino connection"""
    global _rhino_connection
    if _rhino_connection is None:
        _rhino_connection = RhinoConnection()
    return _rhino_connection

class RhinoTools:
    """Collection of tools for interacting with Rhino."""
    
    def __init__(self, app):
        self.app = app
        self._register_tools()
    
    def _register_tools(self):
        """Register all Rhino tools with the MCP server."""
        self.app.tool()(self.get_scene_info)
        self.app.tool()(self.get_layers)
        self.app.tool()(self.get_scene_objects_with_metadata)
        self.app.tool()(self.capture_viewport)
        self.app.tool()(self.execute_rhino_code)
        #NEW
        self.app.tool()(self.zoom_to_target)
        self.app.tool()(self.set_view_projection)
        # --- 註冊相機目標工具 ---
        self.app.tool()(self.set_view_camera_target)
        # --- 結束註冊 ---
        self.app.tool()(self.capture_focused_view)
        # --- 註冊群組工具 ---
        self.app.tool()(self.add_objects_to_group)
    
    def get_scene_info(self, ctx: Context) -> str:
        """Get basic information about the current Rhino scene.
        
        This is a lightweight function that returns basic scene information:
        - List of all layers with basic information about the layer and 5 sample objects with their metadata 
        - No metadata or detailed properties
        - Use this for quick scene overview or when you only need basic object information
        
        Returns:
            JSON string containing basic scene information
        """
        try:
            connection = get_rhino_connection()
            result = connection.send_command("get_scene_info")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error getting scene info from Rhino: {0}".format(str(e)))
            return "Error getting scene info: {0}".format(str(e))

    def get_layers(self, ctx: Context) -> str:
        """Get list of layers in Rhino"""
        try:
            connection = get_rhino_connection()
            result = connection.send_command("get_layers")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error getting layers from Rhino: {0}".format(str(e)))
            return "Error getting layers: {0}".format(str(e))

    def get_scene_objects_with_metadata(self, ctx: Context, filters: Optional[Dict[str, Any]] = None, metadata_fields: Optional[List[str]] = None) -> str:
        """Get detailed information about objects in the scene with their metadata.
        
        This is a CORE FUNCTION for scene context awareness. It provides:
        1. Full metadata for each object we created via this mcp connection including:
           - short_id (DDHHMMSS format), can be dispalyed in the viewport when using capture_viewport, can help yo uto visually identify the a object and find it with this function
           - created_at timestamp
           - layer  - layer path
           - type - geometry type 
           - bbox - the bounding box as lsit of points
           - name - the name you assigned 
           - description - description yo uasigned 
        
        2. Advanced filtering capabilities:
           - layer: Filter by layer name (supports wildcards, e.g., "Layer*")
           - name: Filter by object name (supports wildcards, e.g., "Cube*")
           - short_id: Filter by exact short ID match
        
        3. Field selection:
           - Can specify which metadata fields to return
           - Useful for reducing response size when only certain fields are needed
        
        Args:
            filters: Optional dictionary of filters to apply
            metadata_fields: Optional list of specific metadata fields to return
        
        Returns:
            JSON string containing filtered objects with their metadata
        """
        try:
            connection = get_rhino_connection()
            result = connection.send_command("get_objects_with_metadata", {
                "filters": filters or {},
                "metadata_fields": metadata_fields
            })
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error getting objects with metadata: {0}".format(str(e)))
            return "Error getting objects with metadata: {0}".format(str(e))

    def capture_viewport(self, ctx: Context, layer: Optional[str] = None, show_annotations: bool = False, max_size: int = 1600) -> str:
        """Capture the current viewport, save it on the Rhino side, and return the file path.

        Args:
            layer: Optional layer name to filter annotations
            show_annotations: Whether to show object annotations, this will display the short_id of the object in the viewport you can use the short_id to select specific objects with the get_objects_with_metadata function
            max_size: Maximum dimension (width or height) of the captured image.

        Returns:
            A string containing the absolute path to the saved image file on the Rhino machine,
            or an error message string if capture failed.
        """
        try:
            connection = get_rhino_connection()
            # --- 【修改點】定義目標保存目錄 ---
            # 注意：這裡使用了絕對路徑。確保 LangGraph 進程有權限寫入此目錄。
            # 使用 r"" 來處理 Windows 路徑中的反斜杠
            target_save_dir = r"D:\MA system\LangGraph\output\cache\model_cache"

            # --- 【修改點】在發送的參數中加入 save_directory ---
            result = connection.send_command("capture_viewport", {
                "layer": layer,
                "show_annotations": show_annotations,
                "max_size": max_size,
                "save_directory": target_save_dir # Pass the target dir
            })
            logger.info("Received response from Rhino for capture_viewport: %s", result)

            # --- Processing logic remains the same ---
            if result.get("type") == "file_path":
                file_path = result.get("path")
                if file_path and isinstance(file_path, str):
                    if os.path.normpath(file_path).startswith(os.path.normpath(target_save_dir)):
                         logger.info("capture_viewport successful, returning file path in target directory: %s", file_path)
                    else:
                         logger.warning("capture_viewport successful, but file path '%s' is not in the expected directory '%s'. Returning path anyway.", file_path, target_save_dir)
                    return file_path
                else:
                    error_msg = "Rhino returned file_path type but path is missing or invalid."
                    logger.error(error_msg)
                    return f"[Error]: {error_msg}"
            elif result.get("type") == "text":
                 error_msg = result.get("text", "Unknown error text from Rhino capture.")
                 logger.error("Rhino returned error text for capture_viewport: %s", error_msg)
                 return f"[Error]: {error_msg}"
            else:
                 error_msg = "Received unexpected response type from Rhino capture_viewport: {0}".format(result.get('type', 'N/A'))
                 logger.error(error_msg)
                 return f"[Error]: {error_msg}"

        except Exception as e:
            logger.error("Exception during capture_viewport tool execution: %s", str(e))
            return "[Error]: Exception during capture_viewport tool execution: {0}".format(str(e))

    def execute_rhino_code(self, ctx: Context, code: str) -> str:
        """Execute arbitrary Python code in Rhino.
        
        IMPORTANT NOTES FOR CODE EXECUTION:
        0. DONT FORGET NO f-strings! No f-strings, No f-strings!
        1. This is Rhino 8 with IronPython 2.7 - no f-strings or modern Python features
        3. When creating objects, ALWAYS call add_object_metadata(name, description) after creation
        4. For user interaction, you can use RhinoCommon syntax (selected_objects = rs.GetObjects("Please select some objects") etc.) prompted the suer what to do 
           but prefer automated solutions unless user interaction is specifically requested
        
        The add_object_metadata() function is provided in the code context and must be called
        after creating any object. It adds standardized metadata including:
        - name (provided by you)
        - description (provided by you)
        The metadata helps you to identify and select objects later in the scene and stay organised.

        Common Syntax Errors to Avoid:
        2. No walrus operator (:=)
        3. No type hints
        4. No modern Python features (match/case, etc.)
        5. No list/dict comprehensions with multiple for clauses
        6. No assignment expressions in if/while conditions

        Example of proper object creation:
        <<<python
        # Create geometry
        cube_id = rs.AddBox(rs.WorldXYPlane(), 5, 5, 5)
            # Add metadata - ALWAYS do this after creating an object
        add_object_metadata(cube_id, "My Cube", "A test cube created via MCP")
        >>>

        References:
        AddBox(corners)
            Adds a box-shaped polysurface to the document.
        Parameters:
            corners ([point, point, …, point]) – 8 points defining the corners of the box in counter-clockwise order, starting with the bottom rectangle.
        Example:
            import rhinoscriptsyntax as rs
            box = rs.GetBox()
            if box:
                rs.AddBox(box)

        AddSphere(center_or_plane, radius)
            Adds a spherical surface to the document.
        Parameters:
            center_or_plane (point | plane): center point of the sphere; if a plane is given, its origin is used.
            radius (number): sphere radius in current model units.
        Example:
            import rhinoscriptsyntax as rs
            radius = 2
            center = rs.GetPoint("Center of sphere")
            if center:
                rs.AddSphere(center, radius)

        DONT FORGET NO f-strings! No f-strings, No f-strings!
        """
        try:
            code_template = """
import rhinoscriptsyntax as rs
import scriptcontext as sc
import json
import time
from datetime import datetime

def add_object_metadata(obj_id, name=None, description=None):
    \"\"\"Add standardized metadata to an object\"\"\"
    try:
        # Generate short ID
        short_id = datetime.now().strftime("%d%H%M%S")
        
        # Get bounding box
        bbox = rs.BoundingBox(obj_id)
        bbox_data = [[p.X, p.Y, p.Z] for p in bbox] if bbox else []
        
        # Get object type
        obj = sc.doc.Objects.Find(obj_id)
        obj_type = obj.Geometry.GetType().Name if obj else "Unknown"
        
        # Standard metadata
        metadata = {
            "short_id": short_id,
            "created_at": time.time(),
            "layer": rs.ObjectLayer(obj_id),
            "type": obj_type,
            "bbox": bbox_data
        }
        
        # User-provided metadata
        if name:
            rs.ObjectName(obj_id, name)
            metadata["name"] = name
        else:
            auto_name = "{0}_{1}".format(obj_type, short_id)
            rs.ObjectName(obj_id, auto_name)
            metadata["name"] = auto_name
            
        if description:
            metadata["description"] = description
            
        # Store metadata as user text
        user_text_data = metadata.copy()
        user_text_data["bbox"] = json.dumps(bbox_data)
        
        for key, value in user_text_data.items():
            rs.SetUserText(obj_id, key, str(value))
            
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

""" + code
            logger.info("Sending code execution request to Rhino")
            connection = get_rhino_connection()
            result = connection.send_command("execute_code", {"code": code_template})
            
            logger.info("Received response from Rhino: {0}".format(result))
            
            # Simplified error handling
            if result.get("status") == "error":
                error_msg = "Error: {0}".format(result.get("message", "Unknown error"))
                logger.error("Code execution error: {0}".format(error_msg))
                return error_msg
            else:
                response = result.get("result", "Code executed successfully")
                logger.info("Code execution successful: {0}".format(response))
                return response
                
        except Exception as e:
            error_msg = "Error executing code: {0}".format(str(e))
            logger.error(error_msg)
            return error_msg

    #NEW
    def zoom_to_target(self, ctx: Context, 
                       view: Optional[str] = None, 
                       object_ids: Optional[List[str]] = None, 
                       bounding_box: Optional[List[List[float]]] = None,
                       all_views: bool = False) -> str:
        """Zooms a Rhino view to focus on specific targets.

        You can specify ONE target type:
        - object_ids: A list of object ID strings (Guids) to zoom to.
        - bounding_box: An axis-aligned bounding box defined by 8 corner points [[x,y,z], [x,y,z], ...].
        
        If neither 'object_ids' nor 'bounding_box' is provided, the view will zoom to the extents of all visible objects.

        Args:
            view: Optional name or ID of the view to modify. Defaults to the active view in Rhino.
            object_ids: Optional list of object ID strings to zoom the selection to.
            bounding_box: Optional list of 8 points defining a bounding box to zoom to.
            all_views: If True, attempts to zoom all views instead of just the target view (default False). Note that targeting specific objects or bounding box usually makes sense only for a single view unless 'all_views' is True AND the target is extents.

        Returns:
            JSON string indicating success or failure.
        """
        try:
            connection = get_rhino_connection()
            params = {
                "view": view,
                "object_ids": object_ids,
                "bounding_box": bounding_box,
                "all_views": all_views
            }
            # Remove None values to avoid sending unnecessary keys
            params = {k: v for k, v in params.items() if v is not None} 

            result = connection.send_command("zoom_to_target", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error calling zoom_to_target: {0}".format(str(e)))
            return json.dumps({"status": "error", "message": "MCP Error: {0}".format(str(e))}, indent=2)

    def set_view_projection(self, ctx: Context, 
                            projection_type: str, 
                            view: Optional[str] = None, 
                            lens_angle: Optional[float] = 50.0) -> str:
        """Sets the projection mode of a Rhino view.

        Args:
            projection_type: The desired projection mode. Must be one of 'parallel', 'perspective', or 'two_point'.
            view: Optional name or ID of the view to modify. Defaults to the active view in Rhino.
            lens_angle: Optional lens angle (in degrees) for 'perspective' and 'two_point' projections. Defaults to 50.0.

        Returns:
            JSON string indicating success or failure, including the previous projection mode.
        """
        if projection_type not in ['parallel', 'perspective', 'two_point']:
            return json.dumps({"status": "error", "message": "Invalid projection_type. Use 'parallel', 'perspective', or 'two_point'."}, indent=2)
            
        try:
            connection = get_rhino_connection()
            params = {
                "projection_type": projection_type,
                "view": view,
                "lens_angle": lens_angle
            }
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            result = connection.send_command("set_view_projection", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error calling set_view_projection: {0}".format(str(e)))
            return json.dumps({"status": "error", "message": "MCP Error: {0}".format(str(e))}, indent=2)

    # --- 新增組合工具函式 ---
    def capture_focused_view(self, ctx: Context,
                             # Zoom parameters
                             object_ids: Optional[List[str]] = None,
                             bounding_box: Optional[List[List[float]]] = None,
                             # Projection parameters
                             projection_type: Optional[str] = None, # e.g., 'perspective', 'parallel', 'two_point'
                             lens_angle: Optional[float] = 50.0,
                             # --- 新增相機參數 ---
                             camera_position: Optional[List[float]] = None, # [x, y, z]
                             target_position: Optional[List[float]] = None, # [x, y, z]
                             # --- 結束新增 ---
                             # Capture parameters
                             layer: Optional[str] = None, # For annotations
                             show_annotations: bool = False,
                             max_size: int = 1600,
                             # General parameter
                             view: Optional[str] = None # Target view for all operations
                            ) -> str:
        """Sets view projection, camera target, zooms to a target, and captures the viewport.

        Executes operations in the following order:
        1. Set Projection (if `projection_type` is provided)
        2. Set Camera Target (if `camera_position` and `target_position` are provided)
        3. Zoom (if `object_ids` or `bounding_box` is provided; skipped if camera target was set)
        4. Capture Viewport

        Args:
            object_ids: Optional list of object ID strings (Guids) to zoom to.
            bounding_box: Optional axis-aligned bounding box (8 corner points) to zoom to.
                        If neither object_ids nor bounding_box is given, zooms to extents (unless camera target is set).
            projection_type: Optional desired projection ('parallel', 'perspective', 'two_point').
            lens_angle: Lens angle for 'perspective' or 'two_point' projections.
            camera_position: Optional [x, y, z] coordinates for the camera location. Requires target_position.
            target_position: Optional [x, y, z] coordinates for the target point. Requires camera_position.
            layer: Optional layer name to filter annotations shown in the capture.
            max_size: Maximum dimension (width or height) of the captured image.
            view: Optional name or ID of the view for projection, camera, and zoom. Capture targets the active view.

        Returns:
            A string containing the absolute path to the saved image file on the Rhino machine,
            or an error message string if any step failed.
        """
        try:
            connection = get_rhino_connection()
            target_save_dir = r"D:\MA system\LangGraph\output\cache\model_cache" # Define save dir here as well

            params = {
                "view": view,
                "object_ids": object_ids,
                "bounding_box": bounding_box,
                "projection_type": projection_type,
                "lens_angle": lens_angle,
                # --- 傳遞相機參數 ---
                "camera_position": camera_position,
                "target_position": target_position,
                # --- 結束傳遞 ---
                "layer": layer,
                # "show_annotations": show_annotations,
                "max_size": max_size,
                "save_directory": target_save_dir # Pass the save dir to Rhino
            }
            # Remove None values before sending
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Sending command: capture_focused_view with params: %s", params)
            result = connection.send_command("capture_focused_view", params)
            logger.info("Received response from Rhino for capture_focused_view: %s", result)

            # --- Process the result exactly like capture_viewport ---
            if result.get("type") == "file_path":
                file_path = result.get("path")
                if file_path and isinstance(file_path, str):
                    if os.path.normpath(file_path).startswith(os.path.normpath(target_save_dir)):
                         logger.info("capture_focused_view successful, returning file path: %s", file_path)
                    else:
                         logger.warning("capture_focused_view successful, but path '%s' not in expected dir '%s'.", file_path, target_save_dir)
                    return file_path
                else:
                    error_msg = "Rhino returned file_path type but path is missing or invalid."
                    logger.error(error_msg)
                    return f"[Error]: {error_msg}"
            elif result.get("type") == "text":
                 error_msg = result.get("text", "Unknown error text from Rhino focused capture.")
                 logger.error("Rhino returned error text for capture_focused_view: %s", error_msg)
                 return f"[Error]: {error_msg}"
            else:
                 error_msg = "Received unexpected response type from Rhino capture_focused_view: {0}".format(result.get('type', 'N/A'))
                 logger.error(error_msg)
                 return f"[Error]: {error_msg}"

        except Exception as e:
            logger.error("Exception during capture_focused_view tool execution: %s", str(e))
            return "[Error]: Exception during capture_focused_view tool execution: {0}".format(str(e))
    # --- 結束新增 ---

    # --- 添加群組管理工具函數 ---
    def add_objects_to_group(self, ctx: Context,
                           object_ids: List[str],
                           group_name: Optional[str] = None,
                           create_new: bool = True,
                           delete_existing: bool = False) -> str:
        """將物件添加到新的或現有的群組中。

        群組可以幫助組織多個相關物件，使選擇和視圖操作更簡單。例如，您可以將一個設計
        方案的所有部件放入一個群組，然後在capture_focused_view中通過群組方式選擇全部物件。

        Args:
            object_ids: 要添加到群組的物件ID列表。
            group_name: 要使用或創建的群組名稱。如果不提供，會自動生成一個時間戳名稱。
            create_new: 如果群組已存在，是否創建一個新的同名群組（會添加時間戳）。
            delete_existing: 如果群組已存在，是否先刪除現有群組。這個選項優先於create_new。

        Returns:
            JSON字串，包含操作的結果信息，如成功狀態、群組名稱、添加的物件數量等。
        """
        try:
            connection = get_rhino_connection()
            params = {
                "object_ids": object_ids,
                "group_name": group_name,
                "create_new": create_new,
                "delete_existing": delete_existing
            }
            # 移除None值
            params = {k: v for k, v in params.items() if v is not None}
            
            result = connection.send_command("add_objects_to_group", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error calling add_objects_to_group: {0}".format(str(e)))
            return json.dumps({"status": "error", "message": "MCP Error: {0}".format(str(e))}, indent=2)
    # --- 結束新增 ---

    # --- 新增設定相機目標工具 ---
    def set_view_camera_target(self, ctx: Context,
                               camera_position: List[float],
                               target_position: List[float],
                               view: Optional[str] = None) -> str:
        """Sets the camera location and target point for a Rhino view.

        Args:
            camera_position: The [x, y, z] coordinates for the camera location.
            target_position: The [x, y, z] coordinates for the target point.
            view: Optional name or ID of the view to modify. Defaults to the active view.

        Returns:
            JSON string indicating success or failure, potentially including the previous state.
        """
        if not isinstance(camera_position, list) or len(camera_position) != 3 or \
           not isinstance(target_position, list) or len(target_position) != 3:
            return json.dumps({"status": "error", "message": "Invalid input: camera_position and target_position must be lists of 3 numbers [x, y, z]."}, indent=2)

        try:
            connection = get_rhino_connection()
            params = {
                "view": view,
                "camera_position": camera_position,
                "target_position": target_position
            }
            params = {k: v for k, v in params.items() if v is not None}

            result = connection.send_command("set_view_camera_target", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Error calling set_view_camera_target: {0}".format(str(e)))
            return json.dumps({"status": "error", "message": "MCP Error: {0}".format(str(e))}, indent=2)
    # --- 結束新增 ---