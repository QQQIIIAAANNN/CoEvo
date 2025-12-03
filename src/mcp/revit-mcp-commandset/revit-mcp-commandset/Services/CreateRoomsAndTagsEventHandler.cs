using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using RevitMCPSDK.API.Interfaces;

namespace RevitMCPCommandSet.Services.Generated
{
    public class CreateRoomsAndTagsEventHandler : IExternalEventHandler, IWaitableExternalEventHandler
    {
        public object Result { get; private set; }
        public bool TaskCompleted { get; private set; }
        // 新增屬性以接收外部傳入的房間名稱
        public string RoomName { get; set; } 
        // 新增屬性以接收指定座標
        public double? X { get; set; }
        public double? Y { get; set; }

        private readonly System.Threading.ManualResetEvent _resetEvent = new System.Threading.ManualResetEvent(false);

        public void Execute(UIApplication app)
        {
            Document doc = null;
            try
            {
                doc = app.ActiveUIDocument.Document;
                View activeView = app.ActiveUIDocument.ActiveView;
                ElementId viewId = activeView.Id;

                // 檢查當前視圖是否支援 Tag (必須是平面圖、剖面圖等 2D 視圖)
                if (activeView.ViewType == ViewType.ThreeD)
                {
                    throw new InvalidOperationException("無法在 3D 視圖中建立標籤，請切換至平面圖");
                }

                using (Transaction tx = new Transaction(doc, "Create Rooms & Tags"))
                {
                    tx.Start();

                    int roomCount = 0;
                    int tagCount = 0;
                    ICollection<ElementId> newRoomIds = new List<ElementId>();
                    Level targetLevel = activeView.GenLevel;

                    // 獲取視圖相位 (Phase)
                    Phase viewPhase = null;
                    Parameter phaseParam = activeView.get_Parameter(BuiltInParameter.VIEW_PHASE);
                    if (phaseParam != null && phaseParam.StorageType == StorageType.ElementId)
                    {
                        ElementId phaseId = phaseParam.AsElementId();
                        if (phaseId != ElementId.InvalidElementId)
                        {
                            viewPhase = doc.GetElement(phaseId) as Phase;
                        }
                    }

                    // 模式 A: 指定座標建立單一房間
                    if (X.HasValue && Y.HasValue)
                    {
                        if (targetLevel == null)
                        {
                            throw new InvalidOperationException("當前視圖沒有關聯的樓層 (Level)，無法使用指定座標建立房間。");
                        }

                        // ⚠️ 重要：先檢查該位置是否已經有房間
                        UV point = new UV(X.Value, Y.Value);
                        XYZ checkPoint = new XYZ(X.Value, Y.Value, targetLevel.Elevation);
                        
                        FilteredElementCollector existingRoomCollector = new FilteredElementCollector(doc)
                            .OfClass(typeof(SpatialElement))
                            .OfCategory(BuiltInCategory.OST_Rooms)
                            .WhereElementIsNotElementType();
                        
                        Room existingRoom = null;
                        foreach (SpatialElement elem in existingRoomCollector)
                        {
                            Room room = elem as Room;
                            if (room != null && room.LevelId == targetLevel.Id)
                            {
                                // 檢查房間是否包含該點
                                if (room.IsPointInRoom(checkPoint))
                                {
                                    existingRoom = room;
                                    break;
                                }
                            }
                        }

                        if (existingRoom != null)
                        {
                            // 該位置已有房間，不創建新的，只更新名稱和標籤
                            string warningMsg = $"座標 ({X.Value}, {Y.Value}) 已存在房間 (ID: {existingRoom.Id}, 名稱: '{existingRoom.Name}')。";
                            System.Diagnostics.Debug.WriteLine($"⚠️  {warningMsg}");
                            
                            // 將既有房間視為"新創建"以便後續命名和標籤
                            newRoomIds.Add(existingRoom.Id);
                            roomCount = 0; // 實際沒創建新房間
                            
                            System.Diagnostics.Debug.WriteLine($"   → 將為既有房間更新名稱和標籤");
                        }
                        else
                        {
                            // 該位置沒有房間，創建新的
                            try
                            {
                                Room newRoom = doc.Create.NewRoom(targetLevel, point);
                                
                                if (newRoom != null)
                                {
                                    newRoomIds.Add(newRoom.Id);
                                    roomCount++;
                                    System.Diagnostics.Debug.WriteLine($"✓ 成功在座標 ({X.Value}, {Y.Value}) 創建房間 {newRoom.Id}");
                                }
                                else
                                {
                                    throw new InvalidOperationException($"在座標 ({X.Value}, {Y.Value}) 創建房間失敗：NewRoom 返回 null，該位置可能沒有封閉區域。");
                                }
                            }
                            catch (Autodesk.Revit.Exceptions.ArgumentException argEx)
                            {
                                string errorMsg = $"無法在座標 ({X.Value}, {Y.Value}) 創建房間：{argEx.Message}。請確保該點位於封閉的牆體區域內。";
                                System.Diagnostics.Debug.WriteLine($"✗ {errorMsg}");
                                throw new InvalidOperationException(errorMsg, argEx);
                            }
                            catch (Exception ex)
                            {
                                string errorMsg = $"在座標 ({X.Value}, {Y.Value}) 創建房間時發生未預期錯誤: {ex.Message}";
                                System.Diagnostics.Debug.WriteLine($"✗ {errorMsg}");
                                throw new InvalidOperationException(errorMsg, ex);
                            }
                        }
                    }
                    // 模式 B: 自動掃描建立所有房間
                    else
                    {
                        // 1. 獲取所有 Level
                        FilteredElementCollector levelCollector = new FilteredElementCollector(doc)
                            .OfClass(typeof(Level));

                        foreach (Level level in levelCollector)
                        {
                            // 2. 使用 NewRooms2 自動建立房間
                            ICollection<ElementId> ids = new List<ElementId>();
                            try 
                            {
                                if (viewPhase != null)
                                {
                                    ids = doc.Create.NewRooms2(level, viewPhase);
                                }
                                else
                                {
                                    ids = doc.Create.NewRooms2(level);
                                }
                            }
                            catch (Autodesk.Revit.Exceptions.ArgumentException) 
                            {
                                continue; 
                            }

                            if (ids.Count > 0)
                            {
                                foreach(var id in ids) newRoomIds.Add(id);
                                roomCount += ids.Count;
                            }
                        }
                    }

                    // 3. 處理新建立的房間 (命名 & Tag)
                    if (newRoomIds.Count > 0)
                    {
                        // 重要: 必須重新生成幾何圖形
                        doc.Regenerate();

                        // 先獲取當前視圖中已有的所有 Room Tag
                        FilteredElementCollector existingTagCollector = new FilteredElementCollector(doc, viewId)
                            .OfClass(typeof(IndependentTag))
                            .OfCategory(BuiltInCategory.OST_RoomTags);
                        
                        HashSet<ElementId> alreadyTaggedRoomIds = new HashSet<ElementId>();
                        foreach (IndependentTag existingTag in existingTagCollector)
                        {
                            foreach (var taggedId in existingTag.GetTaggedLocalElementIds())
                            {
                                alreadyTaggedRoomIds.Add(taggedId);
                            }
                        }

                        foreach (ElementId roomId in newRoomIds)
                        {
                            Room room = doc.GetElement(roomId) as Room;
                            if (room == null) continue;
                            
                            // 3.1 命名
                            if (!string.IsNullOrEmpty(RoomName))
                            {
                                try 
                                {
                                    room.Name = RoomName;
                                    System.Diagnostics.Debug.WriteLine($"✓ 房間 {room.Id} 已命名為 '{RoomName}'");
                                }
                                catch (Exception nameEx)
                                {
                                    System.Diagnostics.Debug.WriteLine($"✗ 房間 {room.Id} 命名失敗: {nameEx.Message}");
                                }
                            }

                            // 3.2 建立標籤 (Tag) - 只為尚未有標籤的房間創建
                            // 確保房間在當前視圖的 Level 上且有有效位置
                            bool shouldTag = (targetLevel == null || room.LevelId == targetLevel.Id);
                            
                            if (shouldTag && room.Location != null)
                            {
                                // ⚠️ 關鍵檢查：該房間是否已經有標籤
                                if (alreadyTaggedRoomIds.Contains(room.Id))
                                {
                                    System.Diagnostics.Debug.WriteLine($"⚠️  房間 {room.Id} ('{room.Name}') 已有標籤，跳過");
                                    continue;
                                }

                                LocationPoint locPoint = room.Location as LocationPoint;
                                if (locPoint != null) 
                                {
                                    XYZ tagPoint = locPoint.Point;
                                    try
                                    {
                                        Reference roomRef = new Reference(room);
                                        IndependentTag newTag = IndependentTag.Create(
                                            doc, 
                                            viewId, 
                                            roomRef, 
                                            false, 
                                            TagMode.TM_ADDBY_CATEGORY, 
                                            TagOrientation.Horizontal, 
                                            tagPoint
                                        );
                                        tagCount++;
                                        System.Diagnostics.Debug.WriteLine($"✓ 為房間 {room.Id} ('{room.Name}') 創建標籤 {newTag.Id}");
                                    }
                                    catch (Exception tagEx)
                                    {
                                        System.Diagnostics.Debug.WriteLine($"✗ 房間 {room.Id} 標籤創建失敗: {tagEx.Message}");
                                    }
                                }
                            }
                        }
                    }
                    
                    // 4. 【僅在自動模式下】補充標籤給現有房間
                    // ⚠️ 重要: 如果使用了指定座標 (X, Y)，則跳過此步驟，避免重複創建標籤
                    if (!X.HasValue && !Y.HasValue && targetLevel != null)
                    {
                        FilteredElementCollector roomCollector = new FilteredElementCollector(doc)
                            .OfClass(typeof(SpatialElement))
                            .OfCategory(BuiltInCategory.OST_Rooms)
                            .WhereElementIsNotElementType();

                        FilteredElementCollector tagCollector = new FilteredElementCollector(doc, viewId)
                            .OfClass(typeof(IndependentTag))
                            .OfCategory(BuiltInCategory.OST_RoomTags);

                        HashSet<ElementId> taggedRoomIds = new HashSet<ElementId>();
                        foreach (IndependentTag tag in tagCollector)
                        {
                            foreach (var linkId in tag.GetTaggedLocalElementIds())
                            {
                                taggedRoomIds.Add(linkId);
                            }
                        }

                        foreach (SpatialElement spatialElem in roomCollector)
                        {
                            Room room = spatialElem as Room;
                            if (room == null || room.LevelId != targetLevel.Id || room.Location == null) continue;
                            
                            // 檢查相位
                            if (viewPhase != null)
                            {
                                Parameter roomPhaseParam = room.get_Parameter(BuiltInParameter.ROOM_PHASE);
                                if (roomPhaseParam != null && roomPhaseParam.AsElementId() != viewPhase.Id) continue;
                            }
                            
                            // 只處理尚未被標記的現有房間 (不是新創建的)
                            if (!newRoomIds.Contains(room.Id) && !taggedRoomIds.Contains(room.Id))
                            {
                                LocationPoint locPoint = room.Location as LocationPoint;
                                if (locPoint == null) continue;
                                
                                XYZ tagPoint = locPoint.Point;
                                try
                                {
                                    Reference roomRef = new Reference(room);
                                    IndependentTag.Create(
                                        doc, 
                                        viewId, 
                                        roomRef, 
                                        false, 
                                        TagMode.TM_ADDBY_CATEGORY, 
                                        TagOrientation.Horizontal, 
                                        tagPoint
                                    );
                                    tagCount++;
                                }
                                catch (Exception tagEx)
                                {
                                    System.Diagnostics.Debug.WriteLine($"Tag Failed for existing Room {room.Id}: {tagEx.Message}");
                                }
                            }
                        }
                    }

                    tx.Commit();
                    Result = new { success = true, message = $"執行完成: 新增 {roomCount} 個房間, {tagCount} 個標籤", createdRooms = roomCount, createdTags = tagCount };
                }
            }
            catch (Exception ex)
            {
                Result = new { success = false, message = $"執行失敗: {ex.Message}" };
            }
            finally
            {
                TaskCompleted = true;
                _resetEvent.Set();
            }
        }

        public string GetName()
        {
            return "create_rooms_and_tags";
        }

        public bool WaitForCompletion(int timeoutMilliseconds = 15000)
        {
            return _resetEvent.WaitOne(timeoutMilliseconds);
        }
    }
}
