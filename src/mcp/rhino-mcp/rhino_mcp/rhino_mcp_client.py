"""
Rhino MCP - Rhino-side Script
Handles communication with external MCP server and executes Rhino commands.
"""

import socket
import threading
import json
import time
import System
import Rhino
import scriptcontext as sc
import rhinoscriptsyntax as rs
import os
import platform
import traceback
import sys
import base64
from System.Drawing import Bitmap
from System.Drawing.Imaging import ImageFormat
from System.IO import MemoryStream
from datetime import datetime
import tempfile

# Configuration
HOST = 'localhost'
PORT = 9876

# Add constant for annotation layer
ANNOTATION_LAYER = "MCP_Annotations"

VALID_METADATA_FIELDS = {
    'required': ['id', 'name', 'type', 'layer'],
    'optional': [
        'short_id',      # Short identifier (DDHHMMSS format)
        'created_at',    # Timestamp of creation
        'bbox',          # Bounding box coordinates
        'description',   # Object description
        'user_text'      # All user text key-value pairs
    ]
}

def get_log_dir():
    """Get the appropriate log directory based on the platform"""
    home_dir = os.path.expanduser("~")
    
    # Platform-specific log directory
    if platform.system() == "Darwin":  # macOS
        log_dir = os.path.join(home_dir, "Library", "Application Support", "RhinoMCP", "logs")
    elif platform.system() == "Windows":
        log_dir = os.path.join(home_dir, "AppData", "Local", "RhinoMCP", "logs")
    else:  # Linux and others
        log_dir = os.path.join(home_dir, ".rhino_mcp", "logs")
    
    return log_dir

def log_message(message):
    """Log a message to both Rhino's command line and log file"""
    # Print to Rhino's command line
    Rhino.RhinoApp.WriteLine(message)
    
    # Log to file
    try:
        log_dir = get_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "rhino_mcp.log")
        
        # Log platform info on first run
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("=== RhinoMCP Log ===\n")
                f.write("Platform: {0}\n".format(platform.system()))
                f.write("Python Version: {0}\n".format(sys.version))
                f.write("Rhino Version: {0}\n".format(Rhino.RhinoApp.Version))
                f.write("==================\n\n")
        
        with open(log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{0}] {1}\n".format(timestamp, message))
    except Exception as e:
        Rhino.RhinoApp.WriteLine("Failed to write to log file: {0}".format(str(e)))

class RhinoMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.server_thread = None
    
    def start(self):
        if self.running:
            log_message("Server is already running")
            return
            
        self.running = True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            log_message("RhinoMCP server started on {0}:{1}".format(self.host, self.port))
        except Exception as e:
            log_message("Failed to start server: {0}".format(str(e)))
            self.stop()
            
    def stop(self):
        self.running = False
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        
        log_message("RhinoMCP server stopped")
    
    def _server_loop(self):
        """Main server loop that accepts connections"""
        while self.running:
            try:
                client, addr = self.socket.accept()
                log_message("Client connected from {0}:{1}".format(addr[0], addr[1]))
                
                # Handle client in a new thread
                client_thread = threading.Thread(target=self._handle_client, args=(client,))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    log_message("Error accepting connection: {0}".format(str(e)))
                    time.sleep(0.5)
    
    def _handle_client(self, client):
        """Handle a client connection"""
        try:
            # Set socket buffer size
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 14485760)  # 10MB
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 14485760)  # 10MB
            
            while self.running:
                # Receive command with larger buffer
                data = client.recv(14485760)  # 10MB buffer
                if not data:
                    log_message("Client disconnected")
                    break
                    
                try:
                    command = json.loads(data.decode('utf-8'))
                    log_message("Received command: {0}".format(command))
                    
                    # Create a closure to capture the client connection
                    def execute_wrapper():
                        try:
                            response = self.execute_command(command)
                            response_json = json.dumps(response)
                            # Split large responses into chunks if needed
                            chunk_size = 14485760  # 10MB chunks
                            response_bytes = response_json.encode('utf-8')
                            for i in range(0, len(response_bytes), chunk_size):
                                chunk = response_bytes[i:i + chunk_size]
                                client.sendall(chunk)
                            log_message("Response sent successfully")
                        except Exception as e:
                            log_message("Error executing command: {0}".format(str(e)))
                            traceback.print_exc()
                            error_response = {
                                "status": "error",
                                "message": str(e)
                            }
                            try:
                                client.sendall(json.dumps(error_response).encode('utf-8'))
                            except Exception as e:
                                log_message("Failed to send error response: {0}".format(str(e)))
                                return False  # Signal connection should be closed
                        return True  # Signal connection should stay open
                    
                    # Use RhinoApp.Idle event for IronPython 2.7 compatibility
                    def idle_handler(sender, e):
                        if not execute_wrapper():
                            # If execute_wrapper returns False, close the connection
                            try:
                                client.close()
                            except:
                                pass
                        # Remove the handler after execution
                        Rhino.RhinoApp.Idle -= idle_handler
                    
                    Rhino.RhinoApp.Idle += idle_handler
                    
                except ValueError as e:
                    # Handle JSON decode error (IronPython 2.7)
                    log_message("Invalid JSON received: {0}".format(str(e)))
                    error_response = {
                        "status": "error",
                        "message": "Invalid JSON format"
                    }
                    try:
                        client.sendall(json.dumps(error_response).encode('utf-8'))
                    except:
                        break  # Close connection on send error
                
        except Exception as e:
            log_message("Error handling client: {0}".format(str(e)))
            traceback.print_exc()
        finally:
            try:
                client.close()
            except:
                pass
    
    def execute_command(self, command):
        """Execute a command received from the client"""
        try:
            command_type = command.get("type")
            params = command.get("params", {})
            
            if command_type == "get_scene_info":
                return self._get_scene_info(params)
            elif command_type == "create_cube":
                return self._create_cube(params)
            elif command_type == "get_layers":
                return self._get_layers()
            elif command_type == "execute_code":
                return self._execute_code(params)
            elif command_type == "get_objects_with_metadata":
                return self._get_objects_with_metadata(params)
            elif command_type == "capture_viewport":
                return self._capture_viewport(params)
            elif command_type == "add_metadata":
                return self._add_object_metadata(
                    params.get("object_id"), 
                    params.get("name"), 
                    params.get("description")
                )
            ##NEW
            elif command_type == "zoom_to_target":
                return self._zoom_to_target(params)
            elif command_type == "set_view_projection":
                return self._set_view_projection(params)
            elif command_type == "capture_focused_view":
                return self._capture_focused_view(params)
            elif command_type == "set_view_camera_target":
                return self._set_view_camera_target(params)
            elif command_type == "add_objects_to_group":
                return self._add_objects_to_group(params)
            else:
                return {"status": "error", "message": "Unknown command type"}
                
        except Exception as e:
            log_message("Error executing command: {0}".format(str(e)))
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def _get_scene_info(self, params=None):
        """Get simplified scene information focusing on layers and example objects"""
        try:
            doc = sc.doc
            if not doc:
                return {
                    "status": "error",
                    "message": "No active document"
                }
            
            log_message("Getting simplified scene info...")
            layers_info = []
            
            for layer in doc.Layers:
                layer_objects = [obj for obj in doc.Objects if obj.Attributes.LayerIndex == layer.Index]
                example_objects = []
                
                for obj in layer_objects[:5]:  # Limit to 5 example objects per layer
                    try:
                        # Convert NameValueCollection to dictionary
                        user_strings = {}
                        if obj.Attributes.GetUserStrings():
                            for key in obj.Attributes.GetUserStrings():
                                user_strings[key] = obj.Attributes.GetUserString(key)
                        
                        obj_info = {
                            "id": str(obj.Id),
                            "name": obj.Name or "Unnamed",
                            "type": obj.Geometry.GetType().Name if obj.Geometry else "Unknown",
                            "metadata": user_strings  # Now using the converted dictionary
                        }
                        example_objects.append(obj_info)
                    except Exception as e:
                        log_message("Error processing object: {0}".format(str(e)))
                        continue
                
                layer_info = {
                    "full_path": layer.FullPath,
                    "object_count": len(layer_objects),
                    "is_visible": layer.IsVisible,
                    "is_locked": layer.IsLocked,
                    "example_objects": example_objects
                }
                layers_info.append(layer_info)
            
            response = {
                "status": "success",
                "layers": layers_info
            }
            
            log_message("Simplified scene info collected successfully")
            return response
            
        except Exception as e:
            log_message("Error getting simplified scene info: {0}".format(str(e)))
            return {
                "status": "error",
                "message": str(e),
                "layers": []
            }
    
    def _create_cube(self, params):
        """Create a cube in the scene"""
        try:
            size = float(params.get("size", 1.0))
            location = params.get("location", [0, 0, 0])
            name = params.get("name", "Cube")
            
            # Create cube using RhinoCommon
            box = Rhino.Geometry.Box(
                Rhino.Geometry.Plane.WorldXY,
                Rhino.Geometry.Interval(0, size),
                Rhino.Geometry.Interval(0, size),
                Rhino.Geometry.Interval(0, size)
            )
            
            # Move to specified location
            transform = Rhino.Geometry.Transform.Translation(
                location[0] - box.Center.X,
                location[1] - box.Center.Y,
                location[2] - box.Center.Z
            )
            box.Transform(transform)
            
            # Add to document
            id = sc.doc.Objects.AddBox(box)
            if id != System.Guid.Empty:
                obj = sc.doc.Objects.Find(id)
                if obj:
                    obj.Name = name
                    sc.doc.Views.Redraw()
                    return {
                        "status": "success",
                        "message": "Created cube with size {0}".format(size),
                        "id": str(id)
                    }
            
            return {"status": "error", "message": "Failed to create cube"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _get_layers(self):
        """Get information about all layers"""
        try:
            doc = sc.doc
            layers_data = [] # Renamed variable for clarity

            # Iterate through layers using RhinoCommon is fine
            for layer in doc.Layers:
                # --- Correction Start ---
                object_count = 0
                try:
                    # Use layer's full path for ObjectsByLayer to handle nested layers correctly
                    layer_full_path = layer.FullPath
                    # Use rs.ObjectsByLayer to get objects on this specific layer
                    objects_on_layer = rs.ObjectsByLayer(layer_full_path)
                    # Calculate count
                    if objects_on_layer:
                        object_count = len(objects_on_layer)
                except Exception as e:
                    # Log error if counting fails for a specific layer
                    print("Error counting objects for layer '{0}': {1}".format(layer.FullPath, str(e)))
                    object_count = -1 # Indicate error or inability to count
                # --- Correction End ---

                layers_data.append({
                    "id": layer.Index, # Keep using layer.Index for ID
                    "name": layer.FullPath, # Use FullPath for clarity, as Name might not be unique
                    "object_count": object_count, # Use the calculated count
                    "is_visible": layer.IsVisible,
                    "is_locked": layer.IsLocked,
                    # You might want to add layer color etc. here if needed
                    # "color": layer.Color.ToArgb()
                })

            return {
                "status": "success",
                "layers": layers_data # Return the corrected list
            }
        except Exception as e:
            # General error handling for the function
            print("Error in _get_layers function: {0}".format(str(e))) # Added print for server log
            return {"status": "error", "message": "Failed to retrieve layers: {0}".format(str(e))}
    
    def _execute_code(self, params):
        """Execute arbitrary Python code"""
        try:
            code = params.get("code", "")
            if not code:
                return {"status": "error", "message": "No code provided"}
            
            log_message("Executing code: {0}".format(code))
            
            # Create a new scope for code execution
            local_dict = {}
            
            try:
                # Execute the code
                exec(code, globals(), local_dict)
                
                # Get result from local_dict or use a default message
                result = local_dict.get("result", "Code executed successfully")
                log_message("Code execution completed. Result: {0}".format(result))
                
                response = {
                    "status": "success",
                    "result": str(result),
                    "variables": {k: str(v) for k, v in local_dict.items() if not k.startswith('__')}
                }
                
                log_message("Sending response: {0}".format(json.dumps(response)))
                return response
                
            except Exception as e:
                hint = "Did you use f-string formatting? You have to use IronPython here that doesn't support this."
                error_response = {
                    "status": "error",
                    "message": "{0} {1}".format(hint, str(e)),
                }
                log_message("Error: {0}".format(error_response))
                return error_response
                
        except Exception as e:
            hint = "Did you use f-string formatting? You have to use IronPython here that doesn't support this."
            error_response = {
                "status": "error",
                "message": "{0} {1}".format(hint, str(e)),
            }
            log_message("System error: {0}".format(error_response))
            return error_response

    def _add_object_metadata(self, obj_id, name=None, description=None):
        """Add standardized metadata to an object"""
        try:
            import json
            import time
            from datetime import datetime
            
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
                # Auto-generate name if none provided
                auto_name = "{0}_{1}".format(obj_type, short_id)
                rs.ObjectName(obj_id, auto_name)
                metadata["name"] = auto_name
                
            if description:
                metadata["description"] = description
                
            # Store metadata as user text (convert bbox to string for storage)
            user_text_data = metadata.copy()
            user_text_data["bbox"] = json.dumps(bbox_data)
            
            # Add all metadata as user text
            for key, value in user_text_data.items():
                rs.SetUserText(obj_id, key, str(value))
                
            return {"status": "success"}
        except Exception as e:
            log_message("Error adding metadata: " + str(e))
            return {"status": "error", "message": str(e)}

    def _get_objects_with_metadata(self, params):
        """Get objects with their metadata, with optional filtering"""
        try:
            import re
            import json
            
            filters = params.get("filters", {})
            metadata_fields = params.get("metadata_fields")
            layer_filter = filters.get("layer")
            name_filter = filters.get("name")
            id_filter = filters.get("short_id")
            
            # Validate metadata fields
            all_fields = VALID_METADATA_FIELDS['required'] + VALID_METADATA_FIELDS['optional']
            if metadata_fields:
                invalid_fields = [f for f in metadata_fields if f not in all_fields]
                if invalid_fields:
                    return {
                        "status": "error",
                        "message": "Invalid metadata fields: " + ", ".join(invalid_fields),
                        "available_fields": all_fields
                    }
            
            objects = []
            
            for obj in sc.doc.Objects:
                obj_id = obj.Id
                
                # Apply filters
                if layer_filter:
                    layer = rs.ObjectLayer(obj_id)
                    pattern = "^" + layer_filter.replace("*", ".*") + "$"
                    if not re.match(pattern, layer, re.IGNORECASE):
                        continue
                    
                if name_filter:
                    name = obj.Name or ""
                    pattern = "^" + name_filter.replace("*", ".*") + "$"
                    if not re.match(pattern, name, re.IGNORECASE):
                        continue
                    
                if id_filter:
                    short_id = rs.GetUserText(obj_id, "short_id") or ""
                    if short_id != id_filter:
                        continue
                    
                # Build base object data with required fields
                obj_data = {
                    "id": str(obj_id),
                    "name": obj.Name or "Unnamed",
                    "type": obj.Geometry.GetType().Name,
                    "layer": rs.ObjectLayer(obj_id)
                }
                
                # Get user text data and parse stored values
                stored_data = {}
                for key in rs.GetUserText(obj_id):
                    value = rs.GetUserText(obj_id, key)
                    if key == "bbox":
                        try:
                            value = json.loads(value)
                        except:
                            value = []
                    elif key == "created_at":
                        try:
                            value = float(value)
                        except:
                            value = 0
                    stored_data[key] = value
                
                # Build metadata based on requested fields
                if metadata_fields:
                    metadata = {k: stored_data[k] for k in metadata_fields if k in stored_data}
                else:
                    metadata = {k: v for k, v in stored_data.items() 
                              if k not in VALID_METADATA_FIELDS['required']}
                
                # Only include user_text if specifically requested
                if not metadata_fields or 'user_text' in metadata_fields:
                    user_text = {k: v for k, v in stored_data.items() 
                               if k not in metadata}
                    if user_text:
                        obj_data["user_text"] = user_text
                
                # Add metadata if we have any
                if metadata:
                    obj_data["metadata"] = metadata
                    
                objects.append(obj_data)
            
            return {
                "status": "success",
                "count": len(objects),
                "objects": objects,
                "available_fields": all_fields
            }
            
        except Exception as e:
            log_message("Error filtering objects: " + str(e))
            return {
                "status": "error",
                "message": str(e),
                "available_fields": all_fields
            }

    def _capture_viewport(self, params):
        """Capture viewport, save to a temporary file, and return the file path."""
        log_message("Executing capture_viewport...")
        view = None
        bitmap = None
        resized_bitmap = None
        original_layer = None
        temp_dots = []
        save_path = None # Initialize save_path

        try:
            layer_name = params.get("layer")
            show_annotations = params.get("show_annotations", False)
            max_size = params.get("max_size", 1600)  # Default max dimension
            original_layer = rs.CurrentLayer() # Store original layer safely

            if show_annotations:
                # Ensure annotation layer exists and is current
                if not rs.IsLayer(ANNOTATION_LAYER):
                    rs.AddLayer(ANNOTATION_LAYER, color=(255, 0, 0))
                rs.CurrentLayer(ANNOTATION_LAYER)

                # Create temporary text dots for each object
                for obj in sc.doc.Objects:
                    if layer_name and rs.ObjectLayer(obj.Id) != layer_name:
                        continue

                    bbox = rs.BoundingBox(obj.Id)
                    if bbox:
                        pt = bbox[1]  # Use top corner of bounding box
                        short_id = rs.GetUserText(obj.Id, "short_id")
                        if not short_id:
                            short_id = datetime.now().strftime("%d%H%M%S")
                            rs.SetUserText(obj.Id, "short_id", short_id)

                        name = rs.ObjectName(obj.Id) or "Unnamed"
                        text = "{0}\n{1}".format(name, short_id)

                        dot_id = rs.AddTextDot(text, pt)
                        rs.TextDotHeight(dot_id, 8)
                        temp_dots.append(dot_id)

            view = sc.doc.Views.ActiveView
            if view is None:
                log_message("Error capturing viewport: No active Rhino view found.")
                # Return error dictionary directly
                return {
                    "type": "text",
                    "text": "Error capturing viewport: No active Rhino view found. Please ensure a viewport is open and active."
                }

            # Capture to bitmap
            bitmap = view.CaptureToBitmap()
            if bitmap is None:
                log_message("Error capturing viewport: CaptureToBitmap returned None.")
                # Return error dictionary directly
                return {
                    "type": "text",
                    "text": "Error capturing viewport: Failed to capture bitmap from the active view. The view might be invalid or Rhino might be busy."
                }

            # Calculate new dimensions while maintaining aspect ratio
            width, height = bitmap.Width, bitmap.Height
            if width <= 0 or height <= 0:
                 log_message("Error capturing viewport: Invalid bitmap dimensions ({0}x{1}).".format(width, height))
                 return { "type": "text", "text": "Error capturing viewport: Captured bitmap has invalid dimensions."}

            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            # Ensure minimum dimension is 1
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            # Ensure dimensions are even for rendering compatibility
            if new_width % 2 != 0:
                new_width -= 1
            if new_height % 2 != 0:
                new_height -= 1
            
            # Ensure minimum dimension is 2 after making them even (to avoid 0 or 1)
            # And ensure they are still even if original was 1 and became 0 then 2.
            new_width = max(2, (new_width // 2) * 2)
            new_height = max(2, (new_height // 2) * 2)


            # Create resized bitmap
            resized_bitmap = Bitmap(bitmap, new_width, new_height)

            # --- 【修改點】Save to temporary file ---
            # temp_dir = tempfile.gettempdir() # Get system temporary directory
            # # Ensure our sub-directory exists (optional, but good practice)
            # capture_dir = os.path.join(temp_dir, "rhino_mcp_captures")

            # --- 【修改點】Save to absolute path ---
            capture_dir = r"D:\MA system\LangGraph\output\cache\model_cache"  # 設定絕對路徑

            # 確保資料夾存在
            if not os.path.exists(capture_dir):
                try:
                    os.makedirs(capture_dir)
                except OSError as e:
                    log_message("Error creating capture directory '{0}': {1}. Using system temp directory.".format(capture_dir, e))
                    capture_dir = tempfile.gettempdir()  # 如果失敗，就用系統暫存目錄作為備援

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = System.Guid.NewGuid().ToString("N")[:8] # Use System.Guid for uniqueness
            file_name = "Rhino_{0}_{1}.jpeg".format(timestamp, unique_id) # Save as JPEG
            save_path = os.path.join(capture_dir, file_name)

            try:
                log_message("Attempting to save capture to: {0}".format(save_path))
                resized_bitmap.Save(save_path, ImageFormat.Jpeg)
                log_message("Capture saved successfully to: {0}".format(save_path))
            except Exception as save_e:
                 log_message("Error saving resized bitmap to '{0}': {1}".format(save_path, save_e))
                 # Return error dictionary
                 return { "type": "text", "text": "Error capturing viewport: Failed to save image file: {0}".format(save_e)}

            # --- Return file path ---
            return {
                "type": "file_path",
                "path": save_path
            }

        except Exception as e:
            log_message("Error during capture viewport processing: {0}".format(e))
            traceback.print_exc() # Log full traceback for debugging
            # Return generic error dictionary
            return {
                "type": "text",
                "text": "Error capturing viewport: An unexpected error occurred: {0}".format(e)
            }
        finally:
            # Clean up resources in finally block
            if temp_dots:
                try:
                    rs.DeleteObjects(temp_dots)
                except Exception as del_e:
                    log_message("Error deleting temp dots: {0}".format(del_e))
            if original_layer is not None: # Check if it was assigned
                 try:
                    rs.CurrentLayer(original_layer)
                 except Exception as layer_e:
                    log_message("Error restoring original layer: {0}".format(layer_e))
            if bitmap and hasattr(bitmap, 'Dispose'):
                 try:
                    bitmap.Dispose()
                 except: pass # Ignore dispose errors
            if resized_bitmap and hasattr(resized_bitmap, 'Dispose'):
                 try:
                     resized_bitmap.Dispose()
                 except: pass # Ignore dispose errors

    #NEW
    def _zoom_to_target(self, params):
        """Zooms the specified view to a target (selected objects, bounding box, or extents)."""
        try:
            view_name = params.get("view") # Optional: specify view name/id, defaults to active view
            object_ids_str = params.get("object_ids") # List of Guid strings
            bounding_box_data = params.get("bounding_box") # List of 8 point lists [ [x,y,z], ... ]
            zoom_all = params.get("all_views", False) # Zoom all views or just the specified/active one

            target_view = view_name if view_name else rs.CurrentView()
            if not rs.IsView(target_view):
                 return {"status": "error", "message": "View '{0}' not found.".format(target_view)}

            if object_ids_str and isinstance(object_ids_str, list):
                # Convert string Guids back to System.Guid
                object_ids = [System.Guid(id_str) for id_str in object_ids_str if rs.IsObject(System.Guid(id_str))]
                if not object_ids:
                     return {"status": "warning", "message": "No valid object IDs provided or found."}

                rs.UnselectAllObjects()
                selected_count = rs.SelectObjects(object_ids)
                if selected_count > 0:
                    rs.ZoomSelected(view=target_view, all=zoom_all)
                    log_message("Zoomed selected {0} objects in view '{1}'.".format(selected_count, target_view))
                    rs.UnselectAllObjects() # Deselect after zooming
                    return {"status": "success", "message": "Zoomed to {0} selected objects.".format(selected_count)}
                else:
                    return {"status": "warning", "message": "Provided object IDs could not be selected."}

            elif bounding_box_data and isinstance(bounding_box_data, list) and len(bounding_box_data) == 8:
                # Convert list of lists to list of Rhino.Geometry.Point3d
                try:
                    bbox_points = [Rhino.Geometry.Point3d(pt[0], pt[1], pt[2]) for pt in bounding_box_data]
                    # Create a BoundingBox object (though rs.ZoomBoundingBox uses the 8 points directly)
                    # bbox = Rhino.Geometry.BoundingBox(bbox_points) # This is not needed for rs.ZoomBoundingBox
                    rs.ZoomBoundingBox(bounding_box=bbox_points, view=target_view, all=zoom_all)
                    log_message("Zoomed to bounding box in view '{0}'.".format(target_view))
                    return {"status": "success", "message": "Zoomed to specified bounding box."}
                except Exception as e:
                    log_message("Error processing bounding box data: {0}".format(e))
                    return {"status": "error", "message": "Invalid bounding box data format."}
            else:
                # Default to Zoom Extents if no specific target is given
                rs.ZoomExtents(view=target_view, all=zoom_all)
                log_message("Zoomed extents in view '{0}'.".format(target_view))
                return {"status": "success", "message": "Zoomed to extents."}

        except Exception as e:
            log_message("Error in _zoom_to_target: {0}".format(str(e)))
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _set_view_projection(self, params):
        """Sets the projection mode for the specified view."""
        try:
            view_name = params.get("view") # Optional: specify view name/id, defaults to active view
            projection_type = params.get("projection_type", "").lower() # 'parallel', 'perspective', 'two_point'
            lens_angle = params.get("lens_angle", 20.0) # Default lens angle for perspective modes

            target_view = view_name if view_name else rs.CurrentView()
            if not rs.IsView(target_view):
                 return {"status": "error", "message": "View '{0}' not found.".format(target_view)}

            # Use rs.ViewProjection which is simpler
            projection_mode = None
            if projection_type == "parallel":
                projection_mode = 1
            elif projection_type == "perspective":
                projection_mode = 2
            elif projection_type == "two_point":
                projection_mode = 3
            else:
                 return {"status": "error", "message": "Invalid projection_type. Use 'parallel', 'perspective', or 'two_point'."}

            # Set the projection
            previous_mode = rs.ViewProjection(target_view, mode=projection_mode)

            # Adjust lens angle if it's a perspective view (mode 2 or 3) and angle was provided
            if projection_mode in [2, 3] and "lens_angle" in params:
                try:
                    current_lens = rs.ViewCameraLens(view=target_view)
                    if abs(current_lens - float(lens_angle)) > 0.1: # Only set if different
                        rs.ViewCameraLens(view=target_view, lens_length=float(lens_angle))
                        log_message("Set lens angle to {0} for view '{1}'.".format(lens_angle, target_view))
                except Exception as lens_e:
                     log_message("Could not set lens angle for view '{0}': {1}".format(target_view, lens_e))


            log_message("Set view '{0}' projection to '{1}'.".format(target_view, projection_type))
            sc.doc.Views.Redraw() # Ensure view updates
            return {"status": "success", "message": "View projection set to {0}.".format(projection_type), "previous_mode": previous_mode}

        except Exception as e:
            log_message("Error in _set_view_projection: {0}".format(str(e)))
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    # --- 新增設定相機目標函數 ---
    def _set_view_camera_target(self, params):
        """Sets the camera location and target point for the specified view."""
        try:
            view_name = params.get("view") # Optional: specify view name/id, defaults to active view
            camera_pos_data = params.get("camera_position") # [x, y, z]
            target_pos_data = params.get("target_position") # [x, y, z]

            if not camera_pos_data or not target_pos_data:
                return {"status": "error", "message": "Both camera_position and target_position must be provided."}

            target_view_name = view_name if view_name else rs.CurrentView()
            if not rs.IsView(target_view_name):
                 return {"status": "error", "message": "View '{0}' not found.".format(target_view_name)}

            try:
                # Convert list to Point3d
                camera_point = Rhino.Geometry.Point3d(float(camera_pos_data[0]), float(camera_pos_data[1]), float(camera_pos_data[2]))
                target_point = Rhino.Geometry.Point3d(float(target_pos_data[0]), float(target_pos_data[1]), float(target_pos_data[2]))
            except (IndexError, ValueError, TypeError) as e:
                 log_message("Error converting camera/target positions: {0}".format(e))
                 return {"status": "error", "message": "Invalid camera_position or target_position format. Must be [x, y, z]."}

            # --- MODIFICATION: Use RhinoCommon for more robust camera control ---
            view = sc.doc.Views.Find(target_view_name, False)
            if not view:
                return {"status": "error", "message": "Could not find view object for '{0}'".format(target_view_name)}

            viewport = view.ActiveViewport
            viewport.SetCameraLocations(target_point, camera_point)
            viewport.CameraUp = Rhino.Geometry.Vector3d(0, 0, 1)
            # --- END MODIFICATION ---

            log_message("Set camera/target for view '{0}'. Camera: {1}, Target: {2}".format(target_view_name, camera_point, target_point))
            sc.doc.Views.Redraw() # Ensure view updates

            return {"status": "success", "message": "View camera and target set."}

        except Exception as e:
            log_message("Error in _set_view_camera_target: {0}".format(str(e)))
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    # --- 結束新增 ---

    def _capture_focused_view(self, params):
        """Combines setting view projection, camera target, zooming, and capturing the viewport."""
        log_message("Executing capture_focused_view...")
        view_name = params.get("view") # Optional target view
        original_display_mode = None # To store the original display mode
        # original_projection_info = None # 保留，以備將來恢復
        # original_camera_info = None # 保留，以備將來恢復

        try:
            target_view_name = view_name if view_name else rs.CurrentView()
            if not rs.IsView(target_view_name):
                return {"type": "text", "text": "Error: View '{0}' not found.".format(target_view_name)}

            # --- 步驟 1: 設定視角投影 (如果需要) ---
            projection_type = params.get("projection_type")
            if projection_type:
                projection_params = {
                    "view": target_view_name,
                    "projection_type": projection_type,
                    "lens_angle": params.get("lens_angle") # Pass lens angle if provided
                }
                projection_params = {k: v for k, v in projection_params.items() if v is not None}
                projection_result = self._set_view_projection(projection_params)
                if projection_result.get("status") != "success":
                    return {"type": "text", "text": "Error setting projection: {0}".format(projection_result.get("message"))}
                log_message("Projection set for capture.")

            # --- 步驟 2: 設定相機目標 (如果需要) ---
            camera_pos = params.get("camera_position")
            target_pos = params.get("target_position")
            if camera_pos and target_pos:
                cam_target_params = {
                    "view": target_view_name,
                    "camera_position": camera_pos,
                    "target_position": target_pos
                }
                # 不需要過濾 None，因為前面已經檢查過兩者都存在
                cam_target_result = self._set_view_camera_target(cam_target_params)
                if cam_target_result.get("status") != "success":
                    # 返回錯誤，因為相機設定失敗會嚴重影響結果
                    return {"type": "text", "text": "Error setting camera target: {0}".format(cam_target_result.get("message"))}
                log_message("Camera target set for capture.")


            # --- 步驟 3: 縮放視圖 (如果需要) ---
            object_ids_str = params.get("object_ids")
            bounding_box_data = params.get("bounding_box")
            if object_ids_str or bounding_box_data:
                zoom_params = {
                    "view": target_view_name,
                    "object_ids": object_ids_str,
                    "bounding_box": bounding_box_data,
                    "all_views": False # Force zoom only in the target view
                }
                zoom_params = {k: v for k, v in zoom_params.items() if v is not None and k != 'all_views'}
                zoom_result = self._zoom_to_target(zoom_params)
                if zoom_result.get("status") != "success":
                     log_message("Warning during zoom: {0}. Proceeding with capture.".format(zoom_result.get("message")))
                     # 考慮是否返回錯誤
                     # return {"type": "text", "text": "Error zooming: {0}".format(zoom_result.get("message"))}
                else:
                    log_message("Zoom applied for capture.")
            else:
                # 如果沒有指定縮放目標，但設定了相機，則不執行 ZoomExtents，否則會覆蓋相機設定
                if not (camera_pos and target_pos):
                    rs.ZoomExtents(view=target_view_name)
                    log_message("Zoomed extents applied for capture (no specific target).")


            # --- 步驟 4: 執行截圖 ---
            capture_params = {
                "layer": params.get("layer"),
                "show_annotations": params.get("show_annotations", False),
                "max_size": params.get("max_size", 800),
                "save_directory": params.get("save_directory") # Pass save directory if provided
            }
            capture_params = {k: v for k, v in capture_params.items() if v is not None}

            # 確保目標視圖是活動的
            current_active_view = rs.CurrentView()
            if target_view_name != current_active_view:
                 rs.CurrentView(target_view_name)
                 time.sleep(0.1)
                 log_message("Set view '{0}' as active for capture.".format(target_view_name))

            # --- NEW: Set display mode to Monochrome for capture ---
            try:
                view = sc.doc.Views.Find(target_view_name, False)
                if view:
                    original_display_mode = view.ActiveViewport.DisplayMode.Id
                    monochrome_mode_id = System.Guid("e7e414f1-3d73-44b3-9c58-3f563467d3e3") # Fixed ID for Monochrome mode
                    monochrome_mode = Rhino.Display.DisplayModeDescription.GetDisplayMode(monochrome_mode_id)
                    
                    if monochrome_mode:
                        log_message("Found 'Monochrome' display mode. Applying for capture.")
                        view.ActiveViewport.DisplayMode = monochrome_mode
                    else:
                        log_message("Warning: 'Monochrome' display mode not found. Falling back to 'Shaded'.")
                        shaded_mode_id = System.Guid("69E0C243-05C3-446C-B42C-299C113C251C") # Shaded mode GUID
                        shaded_mode = Rhino.Display.DisplayModeDescription.GetDisplayMode(shaded_mode_id)
                        if shaded_mode:
                            view.ActiveViewport.DisplayMode = shaded_mode
                        else:
                            log_message("Warning: Fallback 'Shaded' mode also not found. Capture will use current display mode.")
                    
                    sc.doc.Views.Redraw()
                    time.sleep(0.2) # Allow view to update
            except Exception as dm_err:
                log_message("Warning: Could not set display mode for capture: {0}".format(dm_err))
            # --- END NEW ---

            capture_result = self._capture_viewport(capture_params)
            log_message("Capture attempt finished.")

            # --- NEW: Restore original display mode ---
            if original_display_mode and view:
                try:
                    original_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(original_display_mode)
                    if original_mode_desc:
                         view.ActiveViewport.DisplayMode = original_mode_desc
                         log_message("Restored original display mode.")
                         sc.doc.Views.Redraw()
                    else:
                        log_message("Warning: Could not restore original display mode (description not found).")
                except Exception as dm_restore_err:
                    log_message("Warning: Could not restore original display mode: {0}".format(dm_restore_err))
            # --- END NEW ---

            # 恢復活動視圖
            if target_view_name != current_active_view:
                try:
                    rs.CurrentView(current_active_view)
                except Exception as e:
                    log_message("Warning: Could not restore original active view '{0}': {1}".format(current_active_view, e))


            # --- Restore camera if needed (Optional) ---
            # if original_camera_info:
            #     try:
            #         rs.ViewCameraTarget(target_view_name, original_camera_info['camera'], original_camera_info['target'])
            #         rs.ViewCameraLens(target_view_name, original_camera_info['lens'])
            #         sc.doc.Views.Redraw()
            #         log_message("Restored camera state for view '{}'.".format(target_view_name))
            #     except Exception as cam_restore_ex:
            #         log_message("Warning: Failed to restore camera state: {}".format(cam_restore_ex))


            return capture_result

        except Exception as e:
            log_message("Error in _capture_focused_view: {0}".format(str(e)))
            traceback.print_exc()
            return {"type": "text", "text": "Error during focused capture: {0}".format(str(e))}

    def _add_objects_to_group(self, params):
        """添加物件到群組，支持創建新群組或加入現有群組。"""
        try:
            object_ids_str = params.get("object_ids", [])
            group_name = params.get("group_name")
            create_new = params.get("create_new", True)
            delete_existing = params.get("delete_existing", False)
            
            # 驗證輸入
            if not object_ids_str:
                return {"status": "error", "message": "No object IDs provided."}
            
            if not group_name:
                # 如果沒有提供群組名，生成一個時間戳群組名
                group_name = "Group_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            # 將字符串ID轉換為有效的Guid對象
            valid_objects = []
            for id_str in object_ids_str:
                try:
                    obj_id = System.Guid(id_str)
                    if rs.IsObject(obj_id):
                        valid_objects.append(obj_id)
                except:
                    log_message("Warning: Invalid object ID: {0}".format(id_str))
                    continue
            
            if not valid_objects:
                return {"status": "error", "message": "No valid objects found from the provided IDs."}
            
            # 檢查群組是否存在
            group_exists = rs.IsGroup(group_name)
            
            # 如果群組存在且需要新群組，或者需要刪除現有群組
            if group_exists and (create_new or delete_existing):
                if delete_existing:
                    # 嘗試刪除現有群組
                    if not rs.DeleteGroup(group_name):
                        log_message("Warning: Could not delete existing group '{0}'.".format(group_name))
                        # 如果無法刪除且需要新群組，調整群組名
                        if create_new:
                            group_name = "{0}_{1}".format(group_name, datetime.now().strftime("%H%M%S"))
                            group_exists = False
                else:
                    # 不刪除但需要新群組，調整群組名
                    if create_new:
                        group_name = "{0}_{1}".format(group_name, datetime.now().strftime("%H%M%S"))
                        group_exists = False
            
            # 創建群組或添加到現有群組
            num_added = 0
            if not group_exists:
                # 創建新群組
                rs.AddGroup(group_name)
                log_message("Created new group: {0}".format(group_name))
            
            # 將物件添加到群組
            num_added = rs.AddObjectsToGroup(valid_objects, group_name)
            
            if num_added <= 0:
                # 即使沒有新物件被添加，也檢查群組中的物件數量
                objects_in_group_guids = rs.ObjectsByGroup(group_name, True)
                objects_in_group_count = len(objects_in_group_guids) if objects_in_group_guids else 0
                return {
                    "status": "warning", 
                    "message": "No new objects were added to group '{0}'. This could be due to objects already being in the group or other reasons.".format(group_name),
                    "group_name": group_name,
                    "objects_added_this_call": 0,
                    "total_objects_in_group": objects_in_group_count
                }
            
            # 獲取群組中的物件ID列表 (轉換為字串)
            objects_in_group_guids = rs.ObjectsByGroup(group_name, True)
            objects_in_group_ids_str = []
            if objects_in_group_guids:
                for guid_obj in objects_in_group_guids:
                    objects_in_group_ids_str.append(str(guid_obj))
            
            return {
                "status": "success",
                "message": "Added {0} objects to group '{1}'.".format(num_added, group_name),
                "group_name": group_name,
                "objects_added_this_call": num_added, # Renamed for clarity
                "total_objects_in_group": len(objects_in_group_ids_str), # Return the count
                "object_ids_in_group": objects_in_group_ids_str # Return list of string IDs
            }
            
        except Exception as e:
            log_message("Error in _add_objects_to_group: {0}".format(str(e)))
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

# Create and start server
server = RhinoMCPServer(HOST, PORT)
server.start()

# Add commands to Rhino
def start_server():
    """Start the RhinoMCP server"""
    server.start()

def stop_server():
    """Stop the RhinoMCP server"""
    server.stop()

# Automatically start the server when this script is loaded
start_server()
log_message("RhinoMCP script loaded. Server started automatically.")
log_message("To stop the server, run: stop_server()") 