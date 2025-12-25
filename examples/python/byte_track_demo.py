import cv2
from ultralytics import YOLO
import trackforge
import time

def run_tracking(video_path="test_video.mp4", output_path="output_tracking.mp4"):
    # Load model
    model = YOLO("yolo11n.pt")
    
    # Initialize Tracker
    # track_thresh=0.1, track_buffer=30, match_thresh=0.8, det_thresh=0.1
    tracker = trackforge.ByteTrack(0.1, 30, 0.8, 0.1)
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Use MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    t0 = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run Detection
        results = model.predict(frame, verbose=False)
        
        # Prepare detections for Rust tracker
        detections_for_tracker = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # get tlwh
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                tlwh = [float(x1), float(y1), float(w), float(h)]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections_for_tracker.append((tlwh, conf, cls))
        
        # Update Tracker
        # Returns list of (track_id, tlwh, score, class_id)
        online_tracks = tracker.update(detections_for_tracker)
        
        # Draw Tracks
        for t in online_tracks:
            track_id = t[0]
            tlwh = t[1]
            score = t[2]
            class_id = t[3]
            
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            
            # Draw box
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw Label
            label = f"ID: {track_id} {model.names[class_id]} {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Draw frame count
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    t1 = time.time()
    print(f"Done. Processed {frame_count} frames in {t1-t0:.2f}s ({(frame_count / (t1-t0)):.1f} fps)")
    
    cap.release()
    out.release()
    print(f"Saved output video to {output_path}")

if __name__ == "__main__":
    run_tracking()
