# === Harry Potter Magic Wand VFX ===
import cv2, mediapipe as mp, numpy as np, time, math, random

# -----------------------
# Magic Particle (Sparkles & Energy)
# -----------------------
class MagicParticle:
    __slots__ = ("x","y","vx","vy","life","max_life","scale","color","type","layer")
    def __init__(self, x, y, vx, vy, life, scale, color, ptype, layer):
        self.x = float(x)
        self.y = float(y)
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.scale = scale
        self.color = color
        self.type = ptype  # "sparkle", "trail", "energy"
        self.layer = layer  # "front" or "back"

# -----------------------
# Spell Trail (wand path)
# -----------------------
class SpellTrail:
    def __init__(self, max_points=30):
        self.points = []
        self.max_points = max_points
    
    def add(self, x, y):
        self.points.append((x, y, time.time()))
        if len(self.points) > self.max_points:
            self.points.pop(0)
    
    def get_active(self, duration=0.5):
        now = time.time()
        return [(x, y, now - t) for x, y, t in self.points if now - t < duration]

# -----------------------
# Magic System
# -----------------------
class MagicSystem:
    def __init__(self, w, h, maxp=3000):
        self.w = w
        self.h = h
        self.particles = []
        self.maxp = maxp
        self.trail = SpellTrail()
        
        # Magic colors (golden, blue, white sparkles)
        self.colors = {
            "gold": np.array([255, 215, 0], dtype=np.float32) / 255.0,
            "blue": np.array([100, 150, 255], dtype=np.float32) / 255.0,
            "white": np.array([255, 255, 255], dtype=np.float32) / 255.0,
            "cyan": np.array([0, 255, 255], dtype=np.float32) / 255.0,
            "purple": np.array([200, 100, 255], dtype=np.float32) / 255.0,
        }
        
        # Sprites
        self.sparkle_sprite = self.make_sparkle_sprite()
        self.glow_sprite = self.make_glow_sprite()
    
    def make_sparkle_sprite(self, size=32):
        """Star-shaped sparkle"""
        sprite = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx, dy = i - center, j - center
                r = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx)
                # 4-pointed star
                star = abs(math.cos(angle * 2)) * 0.7 + 0.3
                val = max(0, 1 - r / (center * star))
                sprite[j, i] = val ** 2
        return sprite
    
    def make_glow_sprite(self, size=64, sigma=0.4):
        """Soft glow"""
        ax = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2
        alpha = np.exp(-r2 / (2 * sigma**2))
        alpha[r2 > 1] = 0
        return alpha.astype(np.float32)
    
    def spawn_wand_magic(self, x, y, intensity=1.0):
        """Spawn magic particles from wand tip"""
        count = int(8 * intensity)
        for _ in range(count):
            if len(self.particles) >= self.maxp:
                break
            
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.5, 4.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 1.0  # Slight upward drift
            
            life = random.uniform(0.5, 1.2)
            scale = random.uniform(0.3, 0.8)
            
            # Mix of colors
            color_choice = random.choice(["gold", "blue", "white", "cyan"])
            color = self.colors[color_choice]
            
            ptype = random.choice(["sparkle", "sparkle", "glow"])
            layer = random.choice(["front", "back"])
            
            p = MagicParticle(x, y, vx, vy, life, scale, color, ptype, layer)
            self.particles.append(p)
    
    def spawn_trail_magic(self, x, y):
        """Spawn particles along the wand trail"""
        if len(self.particles) >= self.maxp:
            return
        
        # Energy trail particles
        for _ in range(2):
            vx = random.uniform(-0.5, 0.5)
            vy = random.uniform(-0.5, 0.5)
            life = random.uniform(0.3, 0.6)
            scale = random.uniform(0.5, 1.0)
            
            color = self.colors[random.choice(["gold", "blue", "purple"])]
            
            p = MagicParticle(x, y, vx, vy, life, scale, color, "glow", "back")
            self.particles.append(p)
    
    def update(self, dt):
        alive = []
        for p in self.particles:
            # Physics
            p.x += p.vx * dt * 60
            p.y += p.vy * dt * 60
            p.vy += 0.05  # Gravity
            p.vx *= 0.98  # Drag
            p.vy *= 0.98
            p.life -= dt
            
            if 0 < p.x < self.w and 0 < p.y < self.h and p.life > 0:
                alive.append(p)
        
        self.particles = alive
    
    def render_layer(self, frame, layer):
        out = np.zeros_like(frame, dtype=np.float32) / 255.0
        
        for p in self.particles:
            if p.layer != layer:
                continue
            
            # Choose sprite
            if p.type == "sparkle":
                sprite = self.sparkle_sprite
                base_size = 32
            else:
                sprite = self.glow_sprite
                base_size = 64
            
            size = int(base_size * p.scale)
            if size < 4:
                continue
            
            alpha = cv2.resize(sprite, (size, size))
            xs, ys = int(p.x - size // 2), int(p.y - size // 2)
            xe, ye = xs + size, ys + size
            
            if xs < 0 or ys < 0 or xe >= self.w or ye >= self.h:
                continue
            
            roi = out[ys:ye, xs:xe]
            
            # Fade based on life
            life_factor = p.life / p.max_life
            brightness = life_factor * 1.5
            
            a = alpha * brightness
            roi += a[..., None] * p.color
            out[ys:ye, xs:xe] = np.clip(roi, 0, 1)
        
        # Blur for magic glow
        out = cv2.GaussianBlur(out, (15, 15), 4.0)
        return (out * 255).astype(np.uint8)
    
    def render_trail(self, frame):
        """Draw glowing wand trail"""
        overlay = frame.copy()
        trail_points = self.trail.get_active(duration=0.4)
        
        if len(trail_points) > 1:
            for i in range(len(trail_points) - 1):
                x1, y1, age1 = trail_points[i]
                x2, y2, age2 = trail_points[i + 1]
                
                # Fade based on age
                alpha = max(0, 1 - age1 / 0.4)
                thickness = int(15 * alpha) + 2
                
                # Golden trail color
                color = (0, 215, 255)  # BGR: Gold
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                        color, thickness, cv2.LINE_AA)
        
        # Blend trail
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

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
H, W = frame.shape[:2]

# -----------------------
# Main loop
# -----------------------
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation

magic = MagicSystem(W, H)

print("=" * 60)
print("🪄 HARRY POTTER MAGIC WAND")
print("=" * 60)
print("Point with your INDEX finger to cast spells!")
print("Move your finger to create magic trails")
print("Press 'q' to quit")
print("=" * 60)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
     mp_selfie.SelfieSegmentation(model_selection=1) as selfie:

    prev = time.time()
    last_pos = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        seg = selfie.process(rgb)

        now = time.time()
        dt = now - prev
        prev = now

        # Detect wand (index finger tip)
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                # Index finger tip (landmark 8)
                index_tip = hlm.landmark[8]
                x = int(index_tip.x * W)
                y = int(index_tip.y * H)
                
                # Check if finger is extended (simple check)
                index_mcp = hlm.landmark[5]  # Index finger base
                dy = index_mcp.y - index_tip.y
                
                if dy > 0.05:  # Finger is pointing
                    # Calculate movement intensity
                    if last_pos:
                        dx = x - last_pos[0]
                        dy = y - last_pos[1]
                        speed = math.sqrt(dx*dx + dy*dy)
                        intensity = min(speed / 20.0, 3.0)
                    else:
                        intensity = 1.0
                    
                    # Spawn magic
                    magic.spawn_wand_magic(x, y, intensity)
                    magic.trail.add(x, y)
                    
                    # Trail sparkles
                    if random.random() < 0.5:
                        magic.spawn_trail_magic(x, y)
                    
                    last_pos = (x, y)
                    
                    # Draw wand tip indicator
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                else:
                    last_pos = None
        else:
            last_pos = None

        magic.update(dt)

        # Render trail first
        frame = magic.render_trail(frame)

        # Split magic into layers
        magic_back = magic.render_layer(frame, "back")
        magic_front = magic.render_layer(frame, "front")

        # Segmentation mask
        mask = (seg.segmentation_mask > 0.5).astype(np.uint8)

        # Compose: back magic + hand + front magic
        back_blend = cv2.addWeighted(frame, 1.0, magic_back, 1.0, 0)
        hand_layer = np.where(mask[..., None] == 1, frame, back_blend)
        final = cv2.addWeighted(hand_layer, 1.0, magic_front, 1.0, 0)

        # Add text overlay
        cv2.putText(final, "Point your wand (index finger)!", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(final, f"Particles: {len(magic.particles)}", (20, H - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("🪄 Harry Potter Magic", final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()