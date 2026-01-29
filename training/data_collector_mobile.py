"""
Mobile-Friendly Data Collection Server for HandSpeak.ai
Collects skeleton images + landmark logs for training 26-class CNN

Features:
- WebSocket endpoint for mobile camera frames
- Generates 400×400 skeleton images (matching training pipeline)
- Logs landmarks in JSON for analysis
- Quality scoring to accept only clear samples
- Real-time feedback to mobile client

Usage:
    python data_collector_mobile.py
    Then open http://YOUR_IP:8002 on mobile browser
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import mediapipe_compat
from cvzone.HandTrackingModule import HandDetector

app = FastAPI(title="HandSpeak.ai Data Collector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_ROOT = Path("./dataset_v2_mobile")
SKELETON_DIR = DATASET_ROOT / "skeletons"
LANDMARK_DIR = DATASET_ROOT / "landmarks"
METADATA_FILE = DATASET_ROOT / "metadata.json"

# Create directories for all letters
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
for letter in LETTERS:
    (SKELETON_DIR / letter).mkdir(parents=True, exist_ok=True)

LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

# Quality thresholds
MIN_CONFIDENCE = 0.6
MIN_HAND_SIZE = 60  # pixels
OFFSET = 29  # Same as training pipeline

# Initialize hand detector
hd = HandDetector(maxHands=1)

# Sample counters (letter → count)
sample_counts: Dict[str, int] = {}
for letter in LETTERS:
    # Count existing samples
    existing = list((SKELETON_DIR / letter).glob("*.png"))
    sample_counts[letter] = len(existing)

# Thread safety for concurrent connections
count_lock = threading.Lock()
file_lock = threading.Lock()
active_connections = set()

# ============================================================================
# DATA COLLECTION LOGIC
# ============================================================================

class DataCollector:
    """Handles skeleton generation and landmark logging"""
    
    @staticmethod
    def process_frame(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[Dict], float]:
        """
        Process frame and extract skeleton + landmarks
        
        Returns:
            (success, skeleton_image, landmarks_dict, quality_score)
        """
        hands, frame = hd.findHands(frame, draw=False, flipType=True)
        
        if not hands or len(hands) == 0:
            return False, None, None, 0.0
        
        hand = hands[0]
        
        # cvzone returns [lmList, bbox, center]
        if isinstance(hand, list):
            if len(hand) < 2:
                return False, None, None, 0.0
            lmList = hand[0]
            bbox = hand[1]
        else:
            # If it's a dict (older cvzone version)
            lmList = hand.get('lmList', [])
            bbox = hand.get('bbox', [])
        
        if not bbox or len(bbox) < 4:
            return False, None, None, 0.0
            
        x, y, w, h = bbox
        
        # Quality check: hand size
        if w < MIN_HAND_SIZE or h < MIN_HAND_SIZE:
            return False, None, None, 0.0
        
        # Extract ROI
        try:
            roi = frame[max(0, y-OFFSET):y+h+OFFSET, max(0, x-OFFSET):x+w+OFFSET]
            if roi.size == 0:
                return False, None, None, 0.0
        except:
            return False, None, None, 0.0
        
        # Create 400×400 skeleton (EXACT same as training)
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Ensure we have 21 landmarks
        if len(lmList) != 21:
            return False, None, None, 0.0
        
        # Center skeleton on canvas
        os_x = ((400 - w) // 2) - 15
        os_y = ((400 - h) // 2) - 15
        
        # Draw skeleton connections
        DataCollector._draw_skeleton(white, lmList, x, y, os_x, os_y)
        
        # Calculate quality score
        quality = DataCollector._calculate_quality(lmList, w, h, frame.shape, x, y)
        
        # Prepare landmark metadata
        handedness = "Right"  # cvzone default
        if isinstance(hand, dict):
            handedness = hand.get('type', 'Right')
        
        landmarks_dict = {
            "landmarks": [[int(lm[0]), int(lm[1]), int(lm[2]) if len(lm) > 2 else 0] for lm in lmList],
            "bbox": [int(x), int(y), int(w), int(h)],
            "handedness": handedness,
            "frame_shape": list(frame.shape),
        }
        
        return quality >= MIN_CONFIDENCE, white, landmarks_dict, quality
    
    @staticmethod
    def _draw_skeleton(canvas: np.ndarray, lmList: List, x: int, y: int, os_x: int, os_y: int):
        """Draw skeleton EXACTLY like training pipeline"""
        # Finger connections
        fingers = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
        
        for start, end in fingers:
            for t in range(start, end):
                pt1_x = int(lmList[t][0] - x + os_x)
                pt1_y = int(lmList[t][1] - y + os_y)
                pt2_x = int(lmList[t+1][0] - x + os_x)
                pt2_y = int(lmList[t+1][1] - y + os_y)
                cv2.line(canvas, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 3)
        
        # Palm connections
        palm_lines = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]
        for idx1, idx2 in palm_lines:
            pt1_x = int(lmList[idx1][0] - x + os_x)
            pt1_y = int(lmList[idx1][1] - y + os_y)
            pt2_x = int(lmList[idx2][0] - x + os_x)
            pt2_y = int(lmList[idx2][1] - y + os_y)
            cv2.line(canvas, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 3)
        
        # Landmarks
        for i in range(21):
            pt_x = int(lmList[i][0] - x + os_x)
            pt_y = int(lmList[i][1] - y + os_y)
            cv2.circle(canvas, (pt_x, pt_y), 2, (0, 0, 255), 1)
    
    @staticmethod
    def _calculate_quality(lmList: List, w: int, h: int, frame_shape: Tuple, x: int, y: int) -> float:
        """Quality score based on hand clarity and positioning"""
        # Factor 1: Hand size (bigger = better, normalized)
        hand_area = w * h
        frame_area = frame_shape[0] * frame_shape[1]
        size_score = min(1.0, (hand_area / frame_area) * 20)  # Target 5% of frame
        
        # Factor 2: Finger spread (variance in Y positions)
        y_coords = [float(lmList[i][1]) for i in range(4, 21)]  # Skip wrist
        variance = np.var(y_coords)
        spread_score = min(1.0, variance / 1000)  # Normalize
        
        # Factor 3: Hand not cut off (check if landmarks near edges)
        edge_penalty = 0.0
        margin = 30
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        
        for lm in lmList:
            lm_x, lm_y = float(lm[0]), float(lm[1])
            if lm_x < margin or lm_x > frame_w - margin or lm_y < margin or lm_y > frame_h - margin:
                edge_penalty += 0.05
        
        edge_score = max(0.0, 1.0 - edge_penalty)
        
        # Combined score
        quality = (0.4 * size_score + 0.4 * spread_score + 0.2 * edge_score)
        return quality
    
    @staticmethod
    def save_sample(letter: str, skeleton: np.ndarray, landmarks: Dict, quality: float, user_id: str = "mobile_user") -> int:
        """Save skeleton image and landmark log (thread-safe)"""
        with count_lock:
            current_count = sample_counts.get(letter, 0)
            sample_counts[letter] = current_count + 1
            sample_number = current_count
        
        # Skeleton filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        skeleton_filename = f"{user_id}_{timestamp}_sample{sample_number:04d}_q{int(quality*100)}.png"
        skeleton_path = SKELETON_DIR / letter / skeleton_filename
        
        cv2.imwrite(str(skeleton_path), skeleton)
        
        # Landmark log entry
        landmark_entry = {
            **landmarks,
            "letter": letter,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "skeleton_path": str(skeleton_path.relative_to(DATASET_ROOT)),
            "quality_score": float(quality)
        }
        
        # Append to landmarks JSON (thread-safe)
        with file_lock:
            landmark_file = LANDMARK_DIR / f"{letter}_landmarks.json"
            if landmark_file.exists():
                try:
                    data = json.loads(landmark_file.read_text())
                except:
                    data = []
            else:
                data = []
            
            data.append(landmark_entry)
            landmark_file.write_text(json.dumps(data, indent=2))
        
        # Update metadata
        DataCollector._update_metadata()
        
        return sample_counts[letter]
    
    @staticmethod
    def _update_metadata():
        """Update dataset metadata file"""
        metadata = {
            "dataset_version": "2.0_mobile",
            "created": datetime.now().isoformat(),
            "total_samples": sum(sample_counts.values()),
            "samples_per_letter": sample_counts.copy(),
            "quality_threshold": MIN_CONFIDENCE,
            "min_hand_size": MIN_HAND_SIZE,
        }
        METADATA_FILE.write_text(json.dumps(metadata, indent=2))


# ============================================================================
# WEBSOCKET ENDPOINT FOR MOBILE
# ============================================================================

@app.websocket("/ws/collect")
async def websocket_collect(websocket: WebSocket):
    """
    WebSocket endpoint for mobile data collection
    
    Protocol:
        Mobile → Server: {"action": "capture", "letter": "A", "frame": "base64_jpeg", "userId": "user1"}
        Server → Mobile: {"success": true, "quality": 0.85, "count": 123, "preview": "base64_skeleton"}
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    active_connections.add(client_host)
    print(f"[CONNECT] Device connected from {client_host} (Total active: {len(active_connections)})")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                action = payload.get("action")
                
                if action == "capture":
                    letter = payload.get("letter", "A").upper()
                    frame_b64 = payload.get("frame", "")
                    user_id = payload.get("userId", "mobile_user")
                    
                    if letter not in LETTERS:
                        await websocket.send_json({
                            "success": False,
                            "error": f"Invalid letter: {letter}"
                        })
                        continue
                    
                    # Decode frame
                    frame_data = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({
                            "success": False,
                            "error": "Failed to decode frame"
                        })
                        continue
                    
                    # Process frame
                    success, skeleton, landmarks, quality = DataCollector.process_frame(frame)
                    
                    if not success:
                        await websocket.send_json({
                            "success": False,
                            "handDetected": False,
                            "quality": quality,
                            "message": "No clear hand detected or quality too low"
                        })
                        continue
                    
                    # Save sample
                    count = DataCollector.save_sample(letter, skeleton, landmarks, quality, user_id)
                    
                    # Encode skeleton preview for mobile
                    _, buffer = cv2.imencode('.jpg', skeleton, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    skeleton_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send success response
                    await websocket.send_json({
                        "success": True,
                        "handDetected": True,
                        "letter": letter,
                        "count": count,
                        "totalSamples": sum(sample_counts.values()),
                        "samplesPerLetter": sample_counts.copy(),
                        "quality": round(quality, 2),
                        "preview": skeleton_b64,
                        "message": f"✅ Saved {letter} sample #{count}"
                    })
                    
                    print(f"[SAVED] {letter} sample #{count} (quality: {quality:.2f}, user: {user_id}) Total: {sum(sample_counts.values())}")
                
                elif action == "stats":
                    # Return dataset statistics
                    await websocket.send_json({
                        "success": True,
                        "stats": {
                            "total": sum(sample_counts.values()),
                            "per_letter": sample_counts.copy(),
                            "min_needed": 500,
                            "recommended": 1000
                        }
                    })
                
                else:
                    await websocket.send_json({
                        "success": False,
                        "error": f"Unknown action: {action}"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "success": False,
                    "error": "Invalid JSON"
                })
            except Exception as e:
                print(f"[ERROR] Processing: {e}")
                await websocket.send_json({
                    "success": False,
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        active_connections.discard(client_host)
        print(f"[DISCONNECT] Device {client_host} disconnected (Active: {len(active_connections)})")
    except Exception as e:
        active_connections.discard(client_host)
        print(f"[ERROR] WebSocket from {client_host}: {e}")


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve mobile data collection interface"""
    html_path = Path(__file__).parent / "mobile_collector.html"
    if html_path.exists():
        return html_path.read_text()
    else:
        return """
        <html>
        <body>
            <h1>HandSpeak.ai Data Collector</h1>
            <p>mobile_collector.html not found. Create it to use the web interface.</p>
            <p>WebSocket endpoint: ws://YOUR_IP:8002/ws/collect</p>
        </body>
        </html>
        """

@app.get("/stats")
async def get_stats():
    """Get dataset statistics"""
    return {
        "total_samples": sum(sample_counts.values()),
        "samples_per_letter": sample_counts,
        "min_needed_per_letter": 500,
        "recommended_per_letter": 1000,
        "completion_percentage": {
            letter: round((count / 500) * 100, 1) 
            for letter, count in sample_counts.items()
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "dataset_root": str(DATASET_ROOT)}


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  HandSpeak.ai Mobile Data Collection Server")
    print("  Collecting Skeleton Images + Landmark Logs for Training")
    print("=" * 70)
    print(f"[INFO] Dataset location: {DATASET_ROOT.absolute()}")
    print(f"[INFO] Current samples: {sum(sample_counts.values())} total")
    print(f"[INFO] WebSocket: ws://YOUR_IP:8002/ws/collect")
    print(f"[INFO] Web interface: http://YOUR_IP:8002")
    print(f"[INFO] Stats API: http://localhost:8002/stats")
    print("=" * 70)
    print("\n📱 INSTRUCTIONS:")
    print("1. Open http://YOUR_IP:8002 on your mobile browser")
    print("2. Allow camera access")
    print("3. Select letter and capture samples")
    print("4. Aim for 500+ samples per letter")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
