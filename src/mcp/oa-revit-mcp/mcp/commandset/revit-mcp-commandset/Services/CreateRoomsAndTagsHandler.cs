using System;
using System.Collections.Generic;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Autodesk.Revit.Creation;
using Autodesk.Revit.DB.Architecture;

public class CreateRoomsAndTagsHandler
{
    public void Execute(UIApplication uiapp)
    {
        Autodesk.Revit.DB.Document doc = uiapp.ActiveUIDocument.Document;

        // 選一個 plan view 作為 tag 所屬視圖
        View activeView = uiapp.ActiveUIDocument.ActiveView;
        ElementId viewId = activeView.Id;

        // 建立 transaction
        using (Transaction tx = new Transaction(doc, "Create Rooms + Room Tags"))
        {
            tx.Start();

            // 遍歷所有封閉區域 (可以是 Rooms 或 SpatialElement)
            // 這裡用 Room 類別做範例
            FilteredElementCollector collector = new FilteredElementCollector(doc)
                .OfCategory(BuiltInCategory.OST_Rooms)
                .WhereElementIsNotElementType();
            IList<Element> roomElements = collector.ToElements();

            foreach (Element e in roomElements)
            {
                Room room = e as Room;
                if (room == null)
                    continue;

                // 計算中心點 (簡單用 bounding box)
                BoundingBoxXYZ bbox = room.get_BoundingBox(activeView);
                if (bbox == null)
                    continue;

                XYZ center = (bbox.Min + bbox.Max) * 0.5;

                // 建立 RoomTag
                LinkElementId roomLinkId = new LinkElementId(room.Id);
                UV tagPoint = new UV(center.X, center.Y);

                try
                {
                    RoomTag tag = doc.Create.NewRoomTag(roomLinkId, tagPoint, viewId);
                }
                catch (Exception ex)
                {
                    TaskDialog.Show("Error", $"建立 RoomTag 失敗: {ex.Message}");
                }
            }

            tx.Commit();
        }
    }
}
