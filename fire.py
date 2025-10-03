# === Controlled Orange-Purple Fire Magic with Clap Detection ===
import cv2, mediapipe as mp, numpy as np, time, math, random

# -----------------------
# Fire Particle
# -----------------------
class FireParticle:
    __slots__ = ("x","y","vx","vy","life","max_life","scale","temp","layer","follow_strength")
    def __init__(self, x, y, vx, vy, life, scale, temp, layer, follow_strength):
        self.x = float(x)
        self.y = float(y)
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.scale = scale
        self.temp = temp  # 0=orange, 1=purple
        self.layer = layer
        self.follow_strength = follow_strength  # How much it follows hand

# -----------------------
# Improved Clap Detector
# -----------------------
class ClapDetector:
    def __init__(self):
        self.distance_history = []
        self.last_clap_time = 0
        self.clap_cooldown = 0.8
        self.max_history = 10
    
    def detect_clap(self, hands):
        """Improved clap detection with history tracking"""
        if not hands or len(hands) < 2:
            self.distance_history = []
            return False
        
        # Get middle finger tips (more reliable than wrists)
        hand1 = hands[0]["lmList"]
        hand2 = hands[1]["lmList"]
        
        # Middle finger tips (landmark 12)
        x1, y1 = hand1[12][0], hand1[12][1]
        x2, y2 = hand2[12][0], hand2[12][1]
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Track distance history
        self.distance_history.append(distance)
        if len(self.distance_history) > self.max_history:
            self.distance_history.pop(0)
        
        # Need at least 5 frames
        if len(self.distance_history) < 5:
            return False
        
        # Detect rapid closing: hands were far (>200), now close (<100)
        was_far = any(d > 200 for d in self.distance_history[:5])
        is_close = distance < 100
        
        if was_far and is_close:
            current_time = time.time()
            if current_time - self.last_clap_time > self.clap_cooldown:
                self.last_clap_time = current_time
                self.distance_history = []  # Reset after clap
                return True
        
        return False
    
    def get_hands_distance(self, hands):
        """Get current distance between hands"""
        if not hands or len(hands) < 2:
            return 999
        
        hand1 = hands[0]["lmList"]
        hand2 = hands[1]["lmList"]
        x1, y1 = hand1[12][0], hand1[12][1]
        x2, y2 = hand2[12][0], hand2[12][1]
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# -----------------------
# Fire System
# -----------------------
class FireSystem:
    def __init__(self, w, h, maxp=3000):
        self.w = w
        self.h = h
        self.flames = []
        self.maxp = maxp
        
        # Orange and Purple colors (properly balanced)
        self.orange = np.array([255, 140, 30], dtype=np.float32) / 255.0    # Bright orange
        self.orange_red = np.array([255, 100, 50], dtype=np.float32) / 255.0  # Orange-red
        self.purple_red = np.array([200, 80, 150], dtype=np.float32) / 255.0  # Purple-red
        self.purple = np.array([180, 70, 200], dtype=np.float32) / 255.0    # Bright purple
        self.dark_purple = np.array([120, 40, 140], dtype=np.float32) / 255.0  # Dark purple
        
        self.flame_sprite = self.make_flame_sprite()
    
    def make_flame_sprite(self, size=80):
        """Soft flame shape"""
        ax = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(xx**2 + yy**2)
        
        # Teardrop flame
        angle = np.arctan2(yy, xx)
        flame_shape = (1 - r) * (1 + 0.3 * np.cos(angle * 3))
        flame_shape = np.clip(flame_shape, 0, 1)
        
        alpha = flame_shape ** 1.2
        alpha[r > 1] = 0
        return alpha.astype(np.float32)
    
    def spawn_fire(self, x, y, intensity=1.0, target_x=None, target_y=None):
        """Spawn controlled fire particles"""
        # Spawn more particles with higher intensity
        count = int(10 * intensity)
        
        for _ in range(count):
            if len(self.flames) >= self.maxp:
                break
            
            # Controlled upward motion (tighter cone)
            angle = random.uniform(-0.25, 0.25) * intensity  # Narrower spread
            speed = random.uniform(1.5, 3.5) * intensity
            vx = math.sin(angle) * speed
            vy = -abs(math.cos(angle) * speed) - 1.5
            
            life = random.uniform(0.6, 1.2)
            scale = random.uniform(0.5, 1.0) * intensity
            
            # Mix orange and purple (50/50 chance)
            temp = 0 if random.random() < 0.5 else 1  # 0=orange, 1=purple
            
            layer = "back" if random.random() < 0.3 else "front"
            
            # Stronger follow strength for more control
            follow_strength = 0.3 if target_x else 0
            
            p = FireParticle(x, y, vx, vy, life, scale, temp, layer, follow_strength)
            self.flames.append(p)
    
    def update(self, dt, finger_positions):
        """Update with hand tracking"""
        alive_flames = []
        
        for p in self.flames:
            # Follow nearest finger if follow_strength > 0
            if p.follow_strength > 0 and finger_positions:
                # Find nearest finger
                min_dist = 999999
                target_x, target_y = p.x, p.y
                
                for fx, fy in finger_positions:
                    dist = (fx - p.x)**2 + (fy - p.y)**2
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = fx, fy
                
                # Gentle attraction to finger
                if min_dist < 15000:  # Within range
                    dx = target_x - p.x
                    dy = target_y - p.y
                    p.vx += dx * 0.002 * p.follow_strength
                    p.vy += dy * 0.002 * p.follow_strength
            
            # Physics
            p.x += p.vx * dt * 60
            p.y += p.vy * dt * 60
            
            # Subtle turbulence (reduced)
            p.vx += random.uniform(-0.15, 0.15)
            p.vy -= 0.08  # Buoyancy
            
            # Drag
            p.vx *= 0.96
            p.vy *= 0.96
            
            p.life -= dt
            
            # Keep particles closer to hands
            if 0 < p.x < self.w and 0 < p.y < self.h and p.life > 0:
                alive_flames.append(p)
        
        self.flames = alive_flames
    
    def render_layer(self, frame, layer):
        out = np.zeros_like(frame, dtype=np.float32) / 255.0
        
        for p in self.flames:
            if p.layer != layer:
                continue
            
            size = int(80 * p.scale)
            if size < 8:
                continue
            
            sprite = cv2.resize(self.flame_sprite, (size, size))
            xs, ys = int(p.x - size // 2), int(p.y - size // 2)
            xe, ye = xs + size, ys + size
            
            if xs < 0 or ys < 0 or xe >= self.w or ye >= self.h:
                continue
            
            roi = out[ys:ye, xs:xe]
            
            # Color selection: Orange or Purple
            life_factor = p.life / p.max_life
            
            if p.temp == 0:  # Orange flame
                # Orange -> Orange-red -> dark
                if life_factor > 0.6:
                    color = self.orange
                elif life_factor > 0.3:
                    t = (life_factor - 0.3) / 0.3
                    color = t * self.orange + (1 - t) * self.orange_red
                else:
                    t = life_factor / 0.3
                    color = t * self.orange_red + (1 - t) * self.dark_purple
            else:  # Purple flame
                # Purple -> Purple-red -> dark
                if life_factor > 0.6:
                    color = self.purple
                elif life_factor > 0.3:
                    t = (life_factor - 0.3) / 0.3
                    color = t * self.purple + (1 - t) * self.purple_red
                else:
                    t = life_factor / 0.3
                    color = t * self.purple_red + (1 - t) * self.dark_purple
            
            # Controlled brightness
            brightness = life_factor * 0.7
            
            alpha = sprite * brightness
            roi += alpha[..., None] * color
            out[ys:ye, xs:xe] = np.clip(roi, 0, 1)
        
        # Medium blur for glow
        out = cv2.GaussianBlur(out, (13, 13), 3.5)
        return np.clip(out * 255, 0, 255).astype(np.uint8)
    
    def clear_all(self):
        self.flames = []

# -----------------------
# Motion Tracker
# -----------------------
class MotionTracker:
    def __init__(self):
        self.prev_positions = {}
        self.speeds = {}
    
    def update(self, finger_id, x, y):
        speed = 0
        if finger_id in self.prev_positions:
            px, py = self.prev_positions[finger_id]
            dx, dy = x - px, y - py
            speed = math.sqrt(dx * dx + dy * dy)
        
        self.prev_positions[finger_id] = (x, y)
        self.speeds[finger_id] = speed
        return speed
    
    def get_average_speed(self):
        if not self.speeds:
            return 0
        return sum(self.speeds.values()) / len(self.speeds)

# -----------------------
# Camera setup
# -----------------------
def open_first_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Using camera {i}")
                return cap, frame
        cap.release()
    raise RuntimeError("No working camera")

cap, frame = open_first_camera()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
H, W = frame.shape[:2]
print(f"📺 Resolution: {W}x{H}")

# -----------------------
# Main loop
# -----------------------
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation

fire = FireSystem(W, H)
clap_detector = ClapDetector()
motion_tracker = MotionTracker()

FINGERTIPS = [4, 8, 12, 16, 20]

fire_active = False

print("=" * 60)
print("🔥 CONTROLLED FIRE MAGIC")
print("=" * 60)
print("👏 Bring hands together (merge palms) to toggle fire")
print("🤚 Move hands FAST = STRONGER fire")
print("✋ Move hands SLOW = gentler fire")
print("Fire follows your fingertips!")
print("Press 'q' to quit")
print("=" * 60)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands, \
     mp_selfie.SelfieSegmentation(model_selection=1) as selfie:

    prev = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        seg = selfie.process(rgb)

        now = time.time()
        dt = now - prev
        prev = now

        # Get hands data
        hands_data = []
        if hand_results.multi_hand_landmarks:
            for hlm in hand_results.multi_hand_landmarks:
                lmList = [[int(lm.x * W), int(lm.y * H), int(lm.z * W)] 
                          for lm in hlm.landmark]
                hands_data.append({"lmList": lmList})
        
        # Detect clap
        if clap_detector.detect_clap(hands_data):
            fire_active = not fire_active
            if fire_active:
                print("🔥 FIRE ACTIVATED!")
            else:
                print("❄️  FIRE DEACTIVATED!")
                fire.clear_all()
        
        # Get hand distance for visual feedback
        hand_dist = clap_detector.get_hands_distance(hands_data)
        
        # Collect finger positions and spawn fire
        finger_positions = []
        avg_speed = 0
        
        if fire_active and hand_results.multi_hand_landmarks:
            speeds = []
            
            for hand_idx, hlm in enumerate(hand_results.multi_hand_landmarks):
                for finger_idx in FINGERTIPS:
                    lm = hlm.landmark[finger_idx]
                    x = int(lm.x * W)
                    y = int(lm.y * H)
                    
                    finger_positions.append((x, y))
                    
                    # Track speed
                    finger_id = f"{hand_idx}_{finger_idx}"
                    speed = motion_tracker.update(finger_id, x, y)
                    speeds.append(speed)
            
            # Calculate intensity from speed
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                intensity = np.clip(avg_speed / 15.0, 0.3, 2.5)
            else:
                intensity = 0.5
            
            # Spawn fire from each finger
            for x, y in finger_positions:
                fire.spawn_fire(x, y, intensity)

        # Update fire with finger tracking
        fire.update(dt, finger_positions if fire_active else [])

        # Render
        fire_back = fire.render_layer(frame, "back")
        fire_front = fire.render_layer(frame, "front")

        mask = (seg.segmentation_mask > 0.5).astype(np.uint8)

        back_blend = cv2.addWeighted(frame, 1.0, fire_back, 0.9, 0)
        hand_layer = np.where(mask[..., None] == 1, frame, back_blend)
        final = cv2.addWeighted(hand_layer, 1.0, fire_front, 0.9, 0)

        # Status overlay
        status_color = (50, 255, 50) if fire_active else (100, 100, 255)
        status_text = "🔥 FIRE: ON" if fire_active else "FIRE: OFF"
        cv2.putText(final, status_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3, cv2.LINE_AA)
        
        # Intensity bar
        if fire_active:
            intensity_pct = int(np.clip(avg_speed / 20.0, 0, 1) * 100)
            bar_width = int(300 * intensity_pct / 100)
            cv2.rectangle(final, (20, 80), (320, 110), (50, 50, 50), -1)
            cv2.rectangle(final, (20, 80), (20 + bar_width, 110), (0, 165, 255), -1)
            cv2.putText(final, f"INTENSITY: {intensity_pct}%", (25, 102),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Hand distance indicator
        if len(hands_data) >= 2:
            dist_pct = int(np.clip((300 - hand_dist) / 200 * 100, 0, 100))
            indicator = "👏 CLAP!" if hand_dist < 100 else f"Distance: {int(hand_dist)}"
            color = (0, 255, 0) if hand_dist < 100 else (255, 200, 0)
            cv2.putText(final, indicator, (20, H - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(final, "Show BOTH hands to activate!", (20, H - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        
        # Particle count
        cv2.putText(final, f"Particles: {len(fire.flames)}", (W - 250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("🔥 Controlled Fire Magic", final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()