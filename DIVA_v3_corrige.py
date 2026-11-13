"""
D.I.V.A v3 - Dispositif Intelligent de Vision Assistee
-------------------------------------------------------
pip install opencv-python numpy Pillow mediapipe==0.10.14 ultralytics

Lancer :  python diva_v3.py            (options ci-dessous)
  --cam N            camera (defaut 0)
  --w, --h           resolution (defaut 1280x720)
  --model            modele YOLO (defaut yolov8n.pt)
  --hq               precision max (plus lourd)
  --imgsz N          taille inference YOLO (defaut 640)
  --ui-scale F       taille UI (0.8, 1.0, 1.2) - defaut 1.0
  --text-scale F     taille texte (defaut 1.0)
  --high-contrast    palette renforcee
  --cb-mode          palette daltonien (bleu/jaune)
  --flip 0|1         miroir camera (defaut 1)
  --custom-piece P   PNG a ajouter a la palette
  --export-frames    sauvegarde chaque frame de la creation

MODES (touches 1 a 6)
  1 DESSIN         dessine dans l'air -> hologramme + correction des formes
  2 VISAGES        detection + reconnaissance (r = memoriser)
  3 OBJETS         detection YOLO
  4 MOUVEMENT      detection de mouvement (+/- = sensibilite)
  5 CORPS          mains + posture
  6 CONSTRUCTION   atelier : assemble des pieces, cree armures / vehicules / objets

TOUCHES GLOBALES
  a alarme ON/OFF | b fond sombre | s capture | h aide | m son on/off | q quitter

CONSTRUCTION
  PINCE (pouce+index) ou SOURIS = prendre / deposer / cliquer
  Palette a gauche -> glisse une piece dans l'atelier (a droite).
  Les pieces s'aimantent entre elles et sur l'axe central (symetrie auto).
  ANCRAGE = ou la creation est portee (voir ANCHORS ci-dessous).
  Raccourcis : [ ] rotation | + - taille | x suppr | d copie | y symetrie
               w ancrage | n nouveau | tab creation suivante | v couleur
               m masquer | z annuler | k sauver | g groupe | o opacite
               p pause animation
"""
import argparse
import json
import math
import os
import random
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ------------------------------------------------------------------ palette
BASE_COLORS = [(255, 220, 40), (50, 190, 255), (60, 60, 255), (110, 255, 110), (255, 100, 200)]
CB_COLORS   = [(255, 200, 60), (255, 160, 0), (0, 200, 255), (0, 240, 200), (180, 180, 255)]
COLOR_NAMES = ["CYAN", "OR", "ROUGE", "VERT", "VIOLET"]
CB_NAMES    = ["JAUNE", "ORANGE", "BLEU", "TURQUOISE", "GRIS"]
CYAN, WHITE = BASE_COLORS[0], (255, 255, 255)
GREEN, ORANGE, RED = BASE_COLORS[3], (60, 160, 255), (80, 80, 255)
DB_PATH = "diva_faces.npz"
BUILD_PATH = "diva_builds.json"
LOG_PATH = "diva_alarm_log.txt"
OUT_DIR = "diva_captures"
FONT = cv2.FONT_HERSHEY_SIMPLEX

MODES = {ord("1"): "DESSIN", ord("2"): "VISAGES", ord("3"): "OBJETS",
         ord("4"): "MOUVEMENT", ord("5"): "CORPS", ord("6"): "CONSTRUCTION"}

# ------------------------------------------------------------------ etat global UI
class UI:
    scale = 1.0
    text_scale = 1.0
    high_contrast = False
    cb_mode = False
    sound = True
    show_help = False
    export_frames = False
    custom_piece_path = None

COLORS = BASE_COLORS

def init_ui(args):
    UI.scale = max(0.6, min(2.0, args.ui_scale))
    UI.text_scale = max(0.6, min(2.5, args.text_scale))
    UI.high_contrast = args.high_contrast
    UI.cb_mode = args.cb_mode
    UI.export_frames = args.export_frames
    UI.custom_piece_path = args.custom_piece
    global COLORS, COLOR_NAMES
    if UI.cb_mode:
        COLORS = CB_COLORS
        COLOR_NAMES = CB_NAMES
    else:
        COLORS = BASE_COLORS
        COLOR_NAMES = ["CYAN", "OR", "ROUGE", "VERT", "VIOLET"]

def BEEP(freq=1200, dur=60):
    if not UI.sound:
        return
    try:
        import winsound
        winsound.Beep(freq, dur)
    except Exception:
        pass

# ------------------------------------------------------------------ utilitaires
def txt(img, s, org, scale=0.55, color=None, th=None):
    color = color if color is not None else (CYAN if not UI.cb_mode else COLORS[0])
    th = th if th is not None else (2 if UI.high_contrast else 1)
    sc = scale * UI.text_scale
    cv2.putText(img, s, (int(org[0]) + 1, int(org[1]) + 1), FONT, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, sc, color, th, cv2.LINE_AA)

def brackets(img, x1, y1, x2, y2, color=None, th=None):
    color = color if color is not None else CYAN
    th = th if th is not None else (3 if UI.high_contrast else 2)
    L = max(12, int(min(x2 - x1, y2 - y1) * 0.25))
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, th, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, th, cv2.LINE_AA)

def smooth(pts, k=5):
    p = np.array(pts, np.float32)
    if len(p) < k + 2:
        return p
    ker = np.ones(k) / k
    pad = k // 2
    xs = np.convolve(np.pad(p[:, 0], (pad, pad), mode="edge"), ker, "valid")
    ys = np.convolve(np.pad(p[:, 1], (pad, pad), mode="edge"), ker, "valid")
    return np.stack([xs, ys], 1)

def glow_of(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    return cv2.convertScaleAbs(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), alpha=3.0)

def add_holo(out, fill, line, fill_alpha=0.28):
    out = cv2.add(out, cv2.convertScaleAbs(fill, alpha=fill_alpha))
    return cv2.add(out, cv2.add(glow_of(line), line))

class AdaptiveEMA:
    def __init__(self, gain=1.0):
        self.v = None
        self.gain = gain

    def reset(self):
        self.v = None

    def __call__(self, p):
        p = np.asarray(p, np.float32)
        if self.v is None:
            self.v = p.copy()
        else:
            base = np.linalg.norm(p - self.v) / 60.0
            a = float(np.clip(0.25 + base * self.gain, 0.20, 0.95))
            self.v += a * (p - self.v)
        return self.v.copy()

# ------------------------------------------------------------------ detection formes
def recognize(raw):
    p = smooth(raw, 5)
    if len(p) < 10:
        return None
    x, y, w, h = cv2.boundingRect(p.astype(np.int32))
    if math.hypot(w, h) < 40:
        return None
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    gap = float(np.linalg.norm(p[0] - p[-1]))
    if gap > 0.22 * length:
        vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        dev = np.abs((p[:, 0] - x0) * vy - (p[:, 1] - y0) * vx)
        if dev.max() < 0.07 * length:
            t = (p[:, 0] - x0) * vx + (p[:, 1] - y0) * vy
            a = np.array([x0 + vx * t.min(), y0 + vy * t.min()])
            b = np.array([x0 + vx * t.max(), y0 + vy * t.max()])
            return dict(kind="line", pts=np.array([a, b]), label="LIGNE")
        return dict(kind="curve", pts=p, label="COURBE")
    cnt = p.reshape(-1, 1, 2).astype(np.float32)
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if area <= 0 or per <= 0:
        return None
    circ = 4 * math.pi * area / (per * per)
    approx = cv2.approxPolyDP(cnt, 0.03 * per, True).reshape(-1, 2).astype(np.float32)
    n = len(approx)
    if n == 3:
        return dict(kind="poly", pts=approx, label="TRIANGLE")
    if n == 4:
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw * rh > 0 and cv2.contourArea(approx) / (rw * rh) > 0.75:
            box = cv2.boxPoints(rect)
            lab = "CARRE" if min(rw, rh) / max(rw, rh) > 0.85 else "RECTANGLE"
            return dict(kind="poly", pts=box, label=lab)
        return dict(kind="poly", pts=approx, label="QUADRILATERE")
    if circ > 0.75 and len(p) >= 5:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        ell = cv2.fitEllipse(cnt)
        (ex, ey), (a, b), _ = ell
        if min(a, b) / max(a, b) > 0.82 and abs(r - max(a, b) / 2) < 0.15 * r:
            return dict(kind="circle", pts=np.array([[cx, cy]]), extra=float(r), label="CERCLE")
        return dict(kind="ellipse", pts=np.array([[cx, cy]]), extra=ell, label="ELLIPSE")
    if n <= 8:
        return dict(kind="poly", pts=approx, label="POLYGONE %d" % n)
    return dict(kind="curve", pts=np.vstack([p, p[:1]]), label="COURBE")

def render_shape(img, s, color, th=2):
    k, P = s["kind"], s["pts"]
    if k == "line":
        cv2.line(img, tuple(int(v) for v in P[0]), tuple(int(v) for v in P[1]), color, th, cv2.LINE_AA)
    elif k == "circle":
        cv2.circle(img, (int(P[0][0]), int(P[0][1])), int(s["extra"]), color, th, cv2.LINE_AA)
    elif k == "ellipse":
        cv2.ellipse(img, s["extra"], color, th, cv2.LINE_AA)
    elif k == "poly":
        cv2.polylines(img, [np.int32(P)], True, color, th, cv2.LINE_AA)
    else:
        cv2.polylines(img, [np.int32(P)], False, color, th, cv2.LINE_AA)

def fingers_up(lm):
    return [lm[t].y < lm[p].y for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]

# ------------------------------------------------------------------ canvas dessin
class Canvas:
    def __init__(self):
        self.shapes, self.cur = [], []
        self.miss, self.last = 0, None
        self.filt = AdaptiveEMA(gain=1.2)
        self.palm_t = None
        self.correct = True
        self.color = 0
        self.toast = ("", 0.0)
        self._last_gesture = None

    def clear(self):
        self.shapes, self.cur = [], []

    def undo(self):
        if self.shapes:
            self.shapes.pop()

    def finish(self):
        pts, self.cur = self.cur, []
        if len(pts) < 8:
            return
        s = recognize(pts) if self.correct else None
        if s is None:
            s = dict(kind="curve", pts=smooth(pts, 5), label="TRAIT")
        s["col"] = self.color
        self.shapes.append(s)
        if s.get("label") and self.correct:
            self.toast = ("CORRIGE : " + s["label"], time.time() + 2)

    def update(self, lm, w, h):
        state = "none"
        if lm:
            up = fingers_up(lm)
            tip = np.array([lm[8].x * w, lm[8].y * h], np.float32)
            self.last = self.filt(tip)
            if up == [True, False, False, False]:
                state = "draw"
            elif all(up):
                state = "palm"
            elif up[0] and up[1]:
                state = "move"
            else:
                state = "idle"
        else:
            self.last = None
            self.filt.reset()
        if state == "draw":
            self.miss = 0
            self.cur.append(self.last.copy())
        elif self.cur:
            self.miss += 1
            if self.miss >= 4:
                self.finish()
        now = time.time()
        if state == "palm":
            if self.palm_t is None:
                self.palm_t = now
            elif now - self.palm_t > 1.0:
                self.clear()
                self.palm_t = now + 999
        else:
            self.palm_t = None
        self._last_gesture = state
        return state

def render_layers(shapes, cur, size):
    h, w = size
    layer = np.zeros((h, w, 3), np.uint8)
    for s in shapes:
        render_shape(layer, s, COLORS[s.get("col", 0) % len(COLORS)], 2)
    if len(cur) > 1:
        cv2.polylines(layer, [np.int32(cur)], False, WHITE, 2, cv2.LINE_AA)
    return layer, glow_of(layer)

# ------------------------------------------------------------------ visages
def face_signature(lm, w, h):
    # FaceMesh fournit x/y/z, mais pas de champ visibility sur ses landmarks.
    if lm is None or len(lm) <= 263:
        return None
    p = np.array([[l.x * w, l.y * h, l.z * w] for l in lm], np.float32)
    if not np.isfinite(p).all():
        return None
    if not (0 <= lm[33].x <= 1 and 0 <= lm[33].y <= 1
            and 0 <= lm[263].x <= 1 and 0 <= lm[263].y <= 1):
        return None
    p -= p[1]
    d = p[263, :2] - p[33, :2]
    scale = float(np.linalg.norm(d))
    if scale < 1e-3:
        return None
    ang = math.atan2(d[1], d[0])
    c, s = math.cos(-ang), math.sin(-ang)
    xy = p[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
    p = np.concatenate([xy, p[:, 2:3]], 1) / scale
    if np.any(np.abs(p) > 3.0):
        return None
    return p.astype(np.float32)

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with np.load(DB_PATH, allow_pickle=False) as data:
            db = {}
            for name in data.files:
                samples = np.asarray(data[name], dtype=np.float32)
                if samples.ndim == 3 and samples.shape[0] > 0 and samples.shape[2] == 3:
                    db[str(name)] = samples
            return db
    except (OSError, ValueError, EOFError) as exc:
        print("[DIVA] base visages illisible :", exc)
        return {}

def save_db(db):
    # Ecriture atomique : une interruption ne detruit pas la base existante.
    tmp = DB_PATH + ".tmp"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **db)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DB_PATH)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def identify(sig, db):
    best, best_d = None, float("inf")
    sig = np.asarray(sig, dtype=np.float32)
    for name, samples in db.items():
        try:
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim != 3 or samples.shape[1:] != sig.shape:
                continue
            dist = float(np.linalg.norm(samples - sig[None, ...], axis=2).mean(axis=1).min())
            if np.isfinite(dist) and dist < best_d:
                best, best_d = name, dist
        except (TypeError, ValueError, FloatingPointError):
            continue
    return best, best_d

# ------------------------------------------------------------------ mouvement / alarme
class Motion:
    def __init__(self):
        self.min_area = 500
        self.reset()

    def reset(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=36, detectShadows=False)
        self.warm, self.mask, self.boxes = 0, None, []
        self.ratio, self.area, self.sc = 0.0, 0, 1.0

    def process(self, frame):
        h, w = frame.shape[:2]
        sc = 480.0 / w if w > 480 else 1.0
        small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1.0 else frame
        small = cv2.GaussianBlur(small, (5, 5), 0)
        m = self.bg.apply(small)
        m = cv2.dilate(cv2.medianBlur(m, 5), None, iterations=2)
        self.warm += 1
        boxes, area = [], 0
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x / sc), int(y / sc), int((x + bw) / sc), int((y + bh) / sc)))
                area += a
        self.ratio = float((m > 0).mean())
        if self.warm < 25:
            boxes, area = [], 0
        self.mask, self.sc, self.boxes, self.area = m, sc, boxes, area

class Alarm:
    def __init__(self):
        self.armed, self.arm_at = False, 0.0
        self.hits, self.ring_until, self.cool_until = 0, 0.0, 0.0
        self.sound, self.last = False, ""

    @property
    def ringing(self):
        return time.time() < self.ring_until

    def arm(self, motion):
        self.armed, self.arm_at, self.hits = True, time.time() + 5, 0
        motion.reset()

    def disarm(self):
        self.armed, self.sound, self.ring_until = False, False, 0.0

    def check(self, motion, frame):
        now = time.time()
        if not self.armed or now < self.arm_at or now < self.cool_until:
            return
        hit = bool(motion.boxes) and motion.ratio < 0.6
        self.hits = self.hits + 1 if hit else max(0, self.hits - 1)
        if self.hits >= 4:
            self.trigger(motion, frame)

    def trigger(self, motion, frame):
        now = time.time()
        self.hits, self.ring_until, self.cool_until = 0, now + 6, now + 9
        snap = frame.copy()
        for x1, y1, x2, y2 in motion.boxes:
            cv2.rectangle(snap, (x1, y1), (x2, y2), RED, 2)
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now()
        path = os.path.join(OUT_DIR, "alarme_%s.jpg" % stamp.strftime("%Y%m%d_%H%M%S"))
        cv2.imwrite(path, snap)
        self.last = stamp.strftime("%H:%M:%S")
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("%s mouvement detecte (%d zones) -> %s\n" % (
                    stamp.isoformat(timespec="seconds"), len(motion.boxes), path))
        except Exception:
            pass
        if not self.sound:
            self.sound = True
            threading.Thread(target=self._siren, daemon=True).start()

    def _siren(self):
        end = self.ring_until
        while self.sound and time.time() < end:
            if UI.sound:
                BEEP(1400, 180)
                BEEP(900, 180)
            time.sleep(0.05)
        self.sound = False

def draw_alarm(out, al):
    h, w = out.shape[:2]
    now = time.time()
    if al.ringing:
        if int(now * 5) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, 18)
        txt(out, "!!! ALARME : MOUVEMENT DETECTE %s !!!" % al.last,
            (max(10, w // 2 - 290), h // 2), 0.9, RED, 2)
    elif al.armed and now < al.arm_at:
        txt(out, "ARMEMENT DANS %d s - quitte le champ" % math.ceil(al.arm_at - now),
            (max(10, w // 2 - 200), 70), 0.7, ORANGE, 2)

# ------------------------------------------------------------------ pieces construction
def _ell(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=36):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)

def _rot(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.asarray(p, np.float32) @ np.array([[c, -s], [s, c]], np.float32)

def _poly(n, r=0.5, rot=0.0):
    return np.stack([(r * math.cos(rot + k * 2 * math.pi / n), r * math.sin(rot + k * 2 * math.pi / n))
                     for k in range(n)], 1).astype(np.float32)

def _star(r_out=0.5, r_in=0.22, points=5, rot=-math.pi / 2):
    pts = []
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = rot + k * math.pi / points
        pts.append((r * math.cos(a), r * math.sin(a)))
    return np.array(pts, np.float32)

# --- pieces de base (existantes)
def P_plaque(an):
    o = [(-.5, -.3), (-.3, -.5), (.3, -.5), (.5, -.3), (.5, .3), (.3, .5), (-.3, .5), (-.5, .3)]
    i = [(-.3, -.2), (-.2, -.3), (.2, -.3), (.3, -.2), (.3, .2), (.2, .3), (-.2, .3), (-.3, .2)]
    return [(o, True, True, None), (i, True, False, None)]

def P_triangle(an):
    return [([(0, -.5), (.5, .5), (-.5, .5)], True, True, None),
            ([(0, -.5), (0, .5)], False, False, None)]

def P_cercle(an):
    return [(_ell(0, 0, .5, .5), True, True, None), (_ell(0, 0, .3, .3), True, False, None)]

def P_hexagone(an):
    return [(_poly(6, .5, math.pi / 6), True, True, None)]

def P_barre(an):
    return [([(-.5, -.15), (.5, -.15), (.5, .15), (-.5, .15)], True, True, None),
            ([(-.5, 0), (.5, 0)], False, False, None)]

def P_roue(an):
    r = an["roll"]
    spokes = [([(0, 0), (.5 * math.cos(r + k * math.pi / 2), .5 * math.sin(r + k * math.pi / 2))],
               False, False, None) for k in range(4)]
    return [(_ell(0, 0, .5, .5), True, True, None), (_ell(0, 0, .3, .3), True, False, None)] + spokes

def P_aile(an):
    return [([(-.5, .5), (-.5, -.1), (.5, -.5), (.5, .15)], True, True, None),
            ([(-.5, .35), (.5, -.15)], False, False, None), ([(-.5, .15), (.5, -.32)], False, False, None)]

def P_dome(an):
    a = np.linspace(math.pi, 2 * math.pi, 24)
    arc = np.stack([.5 * np.cos(a), np.sin(a) + .5], 1)
    return [(arc, True, True, None), (arc * .6 + np.array([0, .2]), True, False, None)]

def P_propulseur(an):
    f = an["flick"]
    return [([(-.25, -.5), (.25, -.5), (.4, 0), (-.4, 0)], True, True, None),
            ([(-.3, .03), (0, .03 + .47 * f), (.3, .03)], True, True, ORANGE),
            ([(-.14, .03), (0, .03 + .28 * f), (.14, .03)], True, True, WHITE)]

def P_helice(an):
    a = an["t"] * 18
    blade = [(-.5, -.05), (.5, -.05), (.5, .05), (-.5, .05)]
    return [(_rot(blade, a), True, True, None), (_rot(blade, a + math.pi / 2), True, True, None),
            (_ell(0, 0, .5, .5), True, False, None), (_ell(0, 0, .07, .07), True, True, WHITE)]

def P_visiere(an):
    return [([(-.5, -.15), (.5, -.15), (.35, .25), (-.35, .25)], True, True, None),
            ([(-.5, -.15), (.5, -.15)], False, False, WHITE)]

def P_canon(an):
    return [([(-.5, -.12), (.25, -.12), (.25, -.22), (.5, -.1), (.5, .1), (.25, .22), (.25, .12), (-.5, .12)],
             True, True, None)]

def P_arc(an):
    pul = .27 + .04 * math.sin(an["t"] * 4)
    prims = [(_ell(0, 0, .5, .5), True, True, None), (_ell(0, 0, .38, .38), True, False, None),
             (_ell(0, 0, pul, pul), True, True, WHITE)]
    for k in range(3):
        a = k * 2 * math.pi / 3 + math.pi / 2
        prims.append(([(0, 0), (.38 * math.cos(a), .38 * math.sin(a))], False, False, None))
    return prims

def P_bouclier(an):
    return [([(-.5, -.5), (.5, -.5), (.5, .1), (0, .5), (-.5, .1)], True, True, None),
            ([(-.3, -.3), (.3, -.3), (.3, .05), (0, .3), (-.3, .05)], True, False, None)]

# --- Nouvelles pieces (creativite libre)
def P_etoile(an):
    return [(_star(.5, .2, 5), True, True, None), (_star(.25, .1, 5), True, False, None)]

def P_fleche(an):
    return [([(-.5, -.1), (.1, -.1), (.1, -.3), (.5, 0), (.1, .3), (.1, .1), (-.5, .1)], True, True, None)]

def P_coeur(an):
    t = np.linspace(0, 2 * math.pi, 60)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1) / 34.0
    return [(pts, True, True, None), (pts * .6, True, False, None)]

def P_engrenage(an):
    t = an["t"] * 2
    n = 12
    pts = []
    for k in range(n * 2):
        r = 0.5 if k % 2 == 0 else 0.4
        a = t + k * math.pi / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return [(np.array(pts, np.float32), True, True, None), (_ell(0, 0, .18, .18), True, False, None)]

def P_croix(an):
    return [([(-.15, -.5), (.15, -.5), (.15, -.15), (.5, -.15), (.5, .15), (.15, .15),
              (.15, .5), (-.15, .5), (-.15, .15), (-.5, .15), (-.5, -.15), (-.15, -.15)],
             True, True, None)]

def P_rond_plein(an):
    return [(_ell(0, 0, .5, .5), True, True, None)]

def P_arc_plein(an):
    a = np.linspace(math.pi, 2 * math.pi, 20)
    arc = np.stack([.5 * np.cos(a), .5 * np.sin(a) + .5], 1)
    return [(arc, True, True, None)]

def P_capsule(an):
    th = np.linspace(-math.pi / 2, math.pi / 2, 16)
    right = np.stack([.2 * np.cos(th) + .3, .5 * np.sin(th)], 1)
    left = np.stack([.2 * np.cos(th + math.pi) - .3, .5 * np.sin(th + math.pi)], 1)
    pts = np.vstack([right, left])
    return [(pts, True, True, None)]

def P_losange(an):
    return [([(0, -.5), (.5, 0), (0, .5), (-.5, 0)], True, True, None),
            ([(0, -.3), (.3, 0), (0, .3), (-.3, 0)], True, False, None)]

def P_parallelogramme(an):
    return [([(-.3, -.4), (.5, -.4), (.3, .4), (-.5, .4)], True, True, None)]

def P_flamme(an):
    t = np.linspace(-math.pi / 2, math.pi * 1.5, 30)
    r = .5 + .1 * np.sin(an["t"] * 8 + t * 3)
    pts = np.stack([r * np.cos(t) * .5, r * np.sin(t) * .5 - .1], 1)
    return [(pts, True, True, None), (pts * .55, True, True, ORANGE)]

def P_eclair(an):
    return [([(-.1, -.5), (.25, -.1), (0, -.1), (.15, .5), (-.25, .05), (0, .05)],
             True, True, None)]

def P_spirale(an):
    t = np.linspace(0, 4 * math.pi, 60)
    r = t / (4 * math.pi) * .5
    pts = np.stack([r * np.cos(t + an["t"] * 2), r * np.sin(t + an["t"] * 2)], 1)
    return [(pts, False, False, None)]

def P_oeil(an):
    a = np.linspace(0, math.pi, 20)
    top = np.stack([-.5 + a / math.pi, -np.sin(a) * .3], 1)
    bot = np.stack([.5 - a / math.pi, np.sin(a) * .3], 1)
    pts = np.vstack([top, bot])
    return [(pts, True, True, None), (_ell(0, 0, .15, .15), True, True, WHITE),
            (_ell(0, 0, .07, .07), True, True, None)]

def P_antenne(an):
    return [([(-.05, .5), (-.05, -.3), (.05, -.3), (.05, .5)], True, True, None),
            (_ell(0, -.4, .12, .12), True, True, None),
            (_ell(0, -.4, .06, .06), True, True, WHITE)]

def P_roue_dentee(an):
    t = an["roll"]
    n = 20
    pts = []
    for k in range(n * 2):
        r = 0.5 if k % 2 == 0 else 0.44
        a = t + k * math.pi / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return [(np.array(pts, np.float32), True, True, None), (_ell(0, 0, .15, .15), True, False, None)]

def P_tuyau(an):
    return [([(-.5, -.25), (.3, -.25), (.3, -.4), (.5, -.4), (.5, .4), (.3, .4), (.3, .25), (-.5, .25)],
             True, True, None)]

def P_anneau(an):
    return [(_ell(0, 0, .5, .5), True, True, None), (_ell(0, 0, .32, .32), True, False, None)]

def P_grille(an):
    prims = [([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)], True, False, None)]
    for i in range(1, 4):
        x = -.5 + i * .25
        prims.append(([(x, -.5), (x, .5)], False, False, None))
    for i in range(1, 4):
        y = -.5 + i * .25
        prims.append(([(-.5, y), (.5, y)], False, False, None))
    return prims

def P_cristal(an):
    return [([(0, -.5), (.3, -.2), (.15, .5), (-.15, .5), (-.3, -.2)], True, True, None),
            ([(0, -.5), (0, .5)], False, False, WHITE),
            ([(.3, -.2), (-.3, -.2)], False, False, WHITE)]

def P_custom(an):
    path = UI.custom_piece_path
    if path and os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            s = max(w, h)
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                ys, xs = np.where(alpha > 127)
                if len(xs) >= 3:
                    # Calculer l'enveloppe sur les pixels avant normalisation.
                    pts_px = np.stack([xs, ys], axis=1).astype(np.int32)
                    hull = cv2.convexHull(pts_px).reshape(-1, 2).astype(np.float32)
                    center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
                    hull = (hull - center) / float(s)
                    return [(hull, True, True, None)]
    return [(_poly(8, .4), True, True, None)]

PARTS = [
    ("plaque", P_plaque), ("triangle", P_triangle), ("cercle", P_cercle), ("hexagone", P_hexagone),
    ("barre", P_barre), ("roue", P_roue), ("aile", P_aile), ("dome", P_dome),
    ("propulseur", P_propulseur), ("helice", P_helice), ("visiere", P_visiere),
    ("canon", P_canon), ("arc", P_arc), ("bouclier", P_bouclier),
    ("etoile", P_etoile), ("fleche", P_fleche), ("coeur", P_coeur),
    ("engrenage", P_engrenage), ("croix", P_croix), ("rond_plein", P_rond_plein),
    ("arc_plein", P_arc_plein), ("capsule", P_capsule), ("losange", P_losange),
    ("parallelogramme", P_parallelogramme), ("flamme", P_flamme), ("eclair", P_eclair),
    ("spirale", P_spirale), ("oeil", P_oeil), ("antenne", P_antenne),
    ("roue_dentee", P_roue_dentee), ("tuyau", P_tuyau), ("anneau", P_anneau),
    ("grille", P_grille), ("cristal", P_cristal), ("custom", P_custom),
]
PART_FN = dict(PARTS)

DEFSIZE = {
    "plaque": (60, 60), "triangle": (50, 50), "cercle": (50, 50), "hexagone": (50, 50),
    "barre": (80, 14), "roue": (50, 50), "aile": (90, 50), "dome": (60, 40),
    "propulseur": (30, 50), "helice": (70, 70), "visiere": (60, 30), "canon": (80, 26),
    "arc": (44, 44), "bouclier": (56, 64), "etoile": (50, 50), "fleche": (70, 30),
    "coeur": (50, 46), "engrenage": (56, 56), "croix": (44, 44), "rond_plein": (44, 44),
    "arc_plein": (50, 30), "capsule": (60, 24), "losange": (44, 44),
    "parallelogramme": (60, 40), "flamme": (40, 60), "eclair": (36, 60),
    "spirale": (56, 56), "oeil": (64, 40), "antenne": (16, 60),
    "roue_dentee": (60, 60), "tuyau": (60, 26), "anneau": (50, 50),
    "grille": (50, 50), "cristal": (36, 60), "custom": (60, 60),
}

ANCHORS = ["MONDE", "TETE", "TORSE", "DOS", "EPAULES", "MAINS", "AVANT-BRAS", "BOTTES"]

# ------------------------------------------------------------------ atelier
class Part(dict):
    """Petit wrapper pour clarté (dict sous-jacent, JSON-friendly)."""

def new_part(kind, x, y, color, opacity=1.0):
    w, h = DEFSIZE[kind]
    return dict(k=kind, x=float(x), y=float(y), w=float(w), h=float(h),
                r=0.0, c=int(color), op=float(opacity), grp=None)

def new_creation(name, anchor="MONDE", cx=380.0, cy=380.0):
    return dict(name=name, parts=[], anchor=anchor, sym=True, adj=1.0,
                wx=cx, wy=cy, ws=1.0, roll=0.0, face=1, vis=True, anim=True)

def part_prims(part, an):
    w, h, r = part["w"], part["h"], math.radians(part.get("r", 0.0))
    out = []
    for pts, closed, fill, col in PART_FN[part["k"]](an):
        p = np.asarray(pts, np.float32) * np.array([w, h], np.float32)
        p = _rot(p, r) + np.array([part["x"], part["y"]], np.float32)
        out.append((p, closed, fill, COLORS[part["c"] % len(COLORS)] if col is None else col))
    return out

def draw_instance(fill_img, line_img, parts, inst, an, sym, th=2):
    ca, sa = math.cos(inst["rot"]), math.sin(inst["rot"])
    R = np.array([[ca, sa], [-sa, ca]], np.float32)
    c = np.asarray(inst["c"], np.float32)
    for part in parts:
        op = part.get("op", 1.0)
        prims = part_prims(part, an)
        for m in ((1, -1) if sym else (1,)):
            for p, closed, fill, col in prims:
                q = p * np.array([m * inst["flip"], 1], np.float32)
                pi = np.int32(np.round((q @ R) * inst["s"] + c))
                if fill and op > 0.02:
                    fcol = tuple(int(v * op) for v in col)
                    cv2.fillPoly(fill_img, [pi], fcol)
                if op > 0.02:
                    lcol = col if op >= 0.95 else tuple(int(v * op) for v in col)
                    cv2.polylines(line_img, [pi], closed, lcol, th, cv2.LINE_AA)

def creation_span(cr):
    if not cr["parts"]:
        return 60.0, 60.0
    xs, ys = [], []
    for p in cr["parts"]:
        r = math.radians(p.get("r", 0.0))
        hw = (abs(p["w"] * math.cos(r)) + abs(p["h"] * math.sin(r))) / 2
        hh = (abs(p["w"] * math.sin(r)) + abs(p["h"] * math.cos(r))) / 2
        xs += [p["x"] - hw, p["x"] + hw]
        ys += [p["y"] - hh, p["y"] + hh]
        if cr["sym"]:
            xs += [-(p["x"] - hw), -(p["x"] + hw)]
    return max(max(xs) - min(xs), 20.0), max(max(ys) - min(ys), 20.0)

def body_instances(anchor, lm, w, h, span_w):
    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h], np.float32)
    def vis(*ids):
        return all(lm[i].visibility > 0.4 for i in ids)
    def fit(target):
        return float(np.clip(target / max(span_w, 20.0), 0.15, 5.0))
    def ang_up(d):
        return math.atan2(d[1], d[0]) + math.pi / 2

    if not vis(11, 12):
        return []
    a, b = pt(11), pt(12)
    L, R = (a, b) if a[0] < b[0] else (b, a)
    sw = float(np.linalg.norm(R - L))
    if sw < 10:
        return []
    mid = (L + R) / 2
    hip = (pt(23) + pt(24)) / 2 if vis(23, 24) else mid + np.array([0, sw * 1.3], np.float32)
    trunk_rot = ang_up(mid - hip)
    out = []
    if anchor == "TORSE":
        out.append(dict(c=mid + (hip - mid) * 0.42, s=fit(sw * 1.5), rot=trunk_rot, flip=1, back=False))
    elif anchor == "DOS":
        out.append(dict(c=mid + (hip - mid) * 0.35, s=fit(sw * 2.4), rot=trunk_rot, flip=1, back=True))
    elif anchor == "TETE":
        if vis(7, 8):
            e1, e2 = pt(7), pt(8)
            e1, e2 = (e1, e2) if e1[0] < e2[0] else (e2, e1)
            hs = float(np.linalg.norm(e2 - e1))
            rot = math.atan2(e2[1] - e1[1], e2[0] - e1[0])
            c = (e1 + e2) / 2
        elif vis(0):
            hs, rot, c = sw * 0.45, 0.0, pt(0)
        else:
            return []
        hs = max(hs, sw * 0.3)
        c = c + np.array([math.sin(rot), -math.cos(rot)], np.float32) * hs * 0.12
        out.append(dict(c=c, s=fit(hs * 1.8), rot=rot, flip=1, back=False))
    elif anchor == "EPAULES":
        d = (R - L) / sw
        out.append(dict(c=L - d * sw * 0.08, s=fit(sw * 0.62), rot=trunk_rot, flip=1, back=False))
        out.append(dict(c=R + d * sw * 0.08, s=fit(sw * 0.62), rot=trunk_rot, flip=-1, back=False))
    elif anchor in ("MAINS", "AVANT-BRAS"):
        pairs = [(13, 15), (14, 16)]
        pairs.sort(key=lambda p: pt(p[1])[0])
        for idx, (e, wr) in enumerate(pairs):
            if not vis(e, wr):
                continue
            d = pt(wr) - pt(e)
            if anchor == "MAINS":
                out.append(dict(c=pt(wr) + d * 0.22, s=fit(sw * 0.42), rot=ang_up(d),
                                flip=1 if idx == 0 else -1, back=False))
            else:
                out.append(dict(c=(pt(e) + pt(wr)) / 2, s=fit(sw * 0.5), rot=ang_up(d),
                                flip=1 if idx == 0 else -1, back=False))
    elif anchor == "BOTTES":
        pairs = [(25, 27), (26, 28)]
        pairs.sort(key=lambda p: pt(p[1])[0])
        for idx, (k, an_) in enumerate(pairs):
            if not vis(k, an_):
                continue
            d = pt(an_) - pt(k)
            out.append(dict(c=pt(an_) + d * 0.08, s=fit(sw * 0.5), rot=ang_up(d),
                            flip=1 if idx == 0 else -1, back=False))
    return out

class Build:
    def __init__(self):
        self.creations = [new_creation("CREATION 1")]
        self.cur, self.sel = 0, None
        self.drag = self.wdrag = self.pal = self.hold = None
        self.hold_t = 0.0
        self.color = 0
        self.snap = (None, None, 0.0)
        self.group_mode = False

    @property
    def cr(self):
        return self.creations[self.cur]

def save_builds(b):
    data = {"cur": b.cur, "creations": b.creations}

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o

    tmp = BUILD_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conv(data), f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, BUILD_PATH)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def load_builds(b):
    if not os.path.exists(BUILD_PATH):
        return
    try:
        with open(BUILD_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("creations"), list):
            raise ValueError("format de sauvegarde invalide")
        crs = []
        for c in data["creations"]:
            if not isinstance(c, dict):
                continue
            base = new_creation(str(c.get("name", "CREATION"))[:80])
            for key in ("anchor", "sym", "adj", "wx", "wy", "ws", "roll", "face", "vis", "anim"):
                if key in c:
                    base[key] = c[key]
            if base["anchor"] not in ANCHORS:
                base["anchor"] = "MONDE"
            for key in ("adj", "wx", "wy", "ws", "roll"):
                try:
                    base[key] = float(base[key])
                    if not math.isfinite(base[key]):
                        raise ValueError
                except (TypeError, ValueError):
                    base[key] = new_creation("CREATION")[key]
            base["parts"] = []
            for p in c.get("parts", []):
                if not isinstance(p, dict) or p.get("k") not in PART_FN:
                    continue
                try:
                    part = new_part(p["k"], p["x"], p["y"], p.get("c", 0), p.get("op", 1.0))
                    part.update({key: p[key] for key in ("w", "h", "r", "grp") if key in p})
                    for key in ("x", "y", "w", "h", "r", "op"):
                        part[key] = float(part[key])
                        if not math.isfinite(part[key]):
                            raise ValueError
                    part["c"] = int(part["c"]) % len(COLORS)
                    part["w"] = float(np.clip(part["w"], 8, 500))
                    part["h"] = float(np.clip(part["h"], 8, 500))
                    base["parts"].append(part)
                except (TypeError, ValueError, KeyError):
                    continue
            base["sym"] = bool(base["sym"])
            base["vis"] = bool(base["vis"])
            base["anim"] = bool(base["anim"])
            base["face"] = 1 if base["face"] >= 0 else -1
            base["adj"] = float(np.clip(base["adj"], 0.3, 3.0))
            base["ws"] = float(np.clip(base["ws"], 0.1, 5.0))
            crs.append(base)
        if crs:
            b.creations = crs
            b.cur = int(np.clip(int(data.get("cur", 0)), 0, len(crs) - 1))
    except Exception as e:
        print("[DIVA] creations illisibles, on repart de zero :", e)

def layout(w, h):
    pw = int(min(340, w * 0.28) * UI.scale)
    ph = int(min(320, h * 0.42) * UI.scale)
    bx0, by0 = w - pw - 12, 46
    rowh = int(np.clip(((h - (by0 + ph + 8) - 30) / 7.0) * UI.scale, 20 * UI.scale, 36 * UI.scale))
    bw = max(40, (pw - 4) // 2)
    buttons = []
    for r, (a, c) in enumerate(BTN_ROWS):
        y0 = by0 + ph + 8 + r * (rowh + 2)
        buttons.append((a, (bx0, y0, bx0 + bw, y0 + rowh)))
        buttons.append((c, (bx0 + bw + 4, y0, bx0 + 2 * bw + 4, y0 + rowh)))
    # Trois colonnes fixes evitent que la palette deborde verticalement.
    cols = 3
    rows = max(1, int(math.ceil(len(PARTS) / float(cols))))
    ss = int(max(18, min(62 * UI.scale, (h - 58) / rows - 6 * UI.scale)))
    slots = []
    for i in range(len(PARTS)):
        col, row = i % cols, i // cols
        x0, y0 = 12 + col * (ss + 6), 46 + row * (ss + 6)
        slots.append((x0, y0, x0 + ss, y0 + ss))
    return dict(bench=(bx0, by0, bx0 + pw, by0 + ph), buttons=buttons, slots=slots, ss=ss, cols=cols)

def inside(r, x, y):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]

def hit_part(cr, lx, ly):
    for i in reversed(range(len(cr["parts"]))):
        p = cr["parts"][i]
        r = math.radians(p.get("r", 0.0))
        for m in ((1, -1) if cr["sym"] else (1,)):
            ax, ay = lx * m - p["x"], ly - p["y"]
            ux = ax * math.cos(r) + ay * math.sin(r)
            uy = -ax * math.sin(r) + ay * math.cos(r)
            if abs(ux) <= p["w"] / 2 and abs(uy) <= p["h"] / 2:
                return i, m == -1
    return None

def snap_pos(cr, idx, x, y, thr=9.0):
    p = cr["parts"][idx]
    cx, cy = [0.0], [0.0]
    if cr["sym"]:
        cx += [p["w"] / 2, -p["w"] / 2]
    for j, o in enumerate(cr["parts"]):
        if j == idx:
            continue
        for s in (1, -1):
            cx += [o["x"] + s * (o["w"] + p["w"]) / 2, o["x"] + s * (o["w"] - p["w"]) / 2]
            cy += [o["y"] + s * (o["h"] + p["h"]) / 2, o["y"] + s * (o["h"] - p["h"]) / 2]
        cx.append(o["x"])
        cy.append(o["y"])
    sx = sy = None
    nx = min(cx, key=lambda v: abs(v - x))
    if abs(nx - x) < thr:
        x, sx = nx, nx
    ny = min(cy, key=lambda v: abs(v - y))
    if abs(ny - y) < thr:
        y, sy = ny, ny
    return x, y, sx, sy

BTN_ROWS = [("NOUVEAU", "VIDER"), ("ROT -", "ROT +"), ("TAILLE -", "TAILLE +"),
            ("COULEUR", "SYMETRIE"), ("ANCRAGE", "CREATION"), ("COPIER", "SUPPR"),
            ("SAUVER", "MASQUER"), ("GROUPE", "OPACITE")]
REPEAT = ("ROT -", "ROT +", "TAILLE -", "TAILLE +", "OPACITE")
BUILD_KEYS = {ord("["): "ROT -", ord("]"): "ROT +", ord("+"): "TAILLE +", ord("="): "TAILLE +",
              ord("-"): "TAILLE -", ord("x"): "SUPPR", ord("d"): "COPIER", ord("y"): "SYMETRIE",
              ord("w"): "ANCRAGE", ord("n"): "NOUVEAU", 9: "CREATION", ord("v"): "COULEUR",
              ord("k"): "SAUVER", ord("m"): "MASQUER", ord("g"): "GROUPE", ord("o"): "OPACITE"}

def do_button(app, label, now):
    b = app.build
    cr = b.cr
    p = cr["parts"][b.sel] if b.sel is not None and b.sel < len(cr["parts"]) else None
    if label == "NOUVEAU":
        b.creations.append(new_creation("CREATION %d" % (len(b.creations) + 1)))
        b.cur, b.sel = len(b.creations) - 1, None
        app.say("Nouvelle creation")
        BEEP(1000, 50)
    elif label == "VIDER":
        cr["parts"].clear()
        b.sel = None
        BEEP(600, 80)
    elif label in ("ROT -", "ROT +"):
        if p:
            p["r"] = (p.get("r", 0.0) + (6 if label == "ROT +" else -6)) % 360
    elif label in ("TAILLE -", "TAILLE +"):
        f = 1.05 if label == "TAILLE +" else 0.95
        if p:
            p["w"] = float(np.clip(p["w"] * f, 8, 260))
            p["h"] = float(np.clip(p["h"] * f, 8, 260))
        else:
            cr["adj"] = float(np.clip(cr["adj"] * f, 0.3, 3.0))
    elif label == "COULEUR":
        b.color = (b.color + 1) % len(COLORS)
        if p:
            p["c"] = b.color
        BEEP(1400, 40)
    elif label == "SYMETRIE":
        cr["sym"] = not cr["sym"]
    elif label == "ANCRAGE":
        cr["anchor"] = ANCHORS[(ANCHORS.index(cr["anchor"]) + 1) % len(ANCHORS)]
        app.say("Ancrage : " + cr["anchor"])
    elif label == "CREATION":
        b.cur, b.sel = (b.cur + 1) % len(b.creations), None
    elif label == "COPIER":
        if p:
            q = dict(p)
            q["x"] = float(p["x"] + 14)
            q["y"] = float(p["y"] + 14)
            cr["parts"].append(q)
            b.sel = len(cr["parts"]) - 1
    elif label == "SUPPR":
        if p:
            del cr["parts"][b.sel]
            b.sel = None
        elif cr["parts"]:
            cr["parts"].pop()
        b.sel = None
    elif label == "SAUVER":
        save_builds(b)
        app.say("Creations sauvegardees")
        BEEP(1200, 100)
    elif label == "MASQUER":
        cr["vis"] = not cr.get("vis", True)
    elif label == "GROUPE":
        if p:
            gid = p.get("grp")
            if gid is None:
                gid = int(time.time() * 1000) % 100000
                p["grp"] = gid
            for q in cr["parts"]:
                if b.sel is not None and abs(cr["parts"].index(q) - b.sel) < 3:
                    if q.get("grp") is None:
                        q["grp"] = gid
            app.say("Groupe cree (id %d)" % gid)
    elif label == "OPACITE":
        if p:
            p["op"] = 0.3 if p.get("op", 1.0) > 0.9 else (0.6 if p.get("op", 1.0) > 0.5 else 1.0)

class Pointer:
    def __init__(self):
        self.filt = AdaptiveEMA(gain=1.0)
        self.down = self.prev = False
        self.pos = self.last = None

    def update(self, lm, w, h, now, mouse):
        if mouse and now - mouse["t"] < 2.0:
            self.pos = np.array([mouse["x"], mouse["y"]], np.float32)
            down = bool(mouse["down"])
            self.filt.reset()
        elif lm:
            a = np.array([lm[4].x * w, lm[4].y * h], np.float32)
            b = np.array([lm[8].x * w, lm[8].y * h], np.float32)
            ref = max(1.0, float(np.linalg.norm(np.array(
                [(lm[0].x - lm[9].x) * w, (lm[0].y - lm[9].y) * h]))))
            d = float(np.linalg.norm(a - b)) / ref
            down = d < (0.50 if self.down else 0.32)
            self.pos = self.filt((a + b) / 2)
        else:
            self.pos, down = None, False
            self.filt.reset()
        press = down and not self.prev
        release = (not down) and self.prev
        self.prev = self.down = down
        if self.pos is not None:
            self.last = (float(self.pos[0]), float(self.pos[1]))
        elif not release:
            return None
        x, y = self.last if self.last else (0, 0)
        return dict(x=x, y=y, down=down, press=press, release=release)

def build_interact(app, ptr, lay, now):
    b = app.build
    cr = b.cr
    bx0, by0, bx1, by1 = lay["bench"]
    ox, oy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    if ptr is None:
        return
    x, y = ptr["x"], ptr["y"]
    inbench = inside(lay["bench"], x, y)

    if ptr["press"]:
        for i, r in enumerate(lay["slots"]):
            if inside(r, x, y):
                b.pal = i
                return
        for label, r in lay["buttons"]:
            if inside(r, x, y):
                do_button(app, label, now)
                b.hold, b.hold_t = (label if label in REPEAT else None), now + 0.35
                return
        if inbench:
            hit = hit_part(cr, x - ox, y - oy)
            if hit is not None:
                i, mir = hit
                p = cr["parts"][i]
                m = -1 if mir else 1
                b.sel = i
                b.drag = dict(i=i, m=m, ox=p["x"] - (x - ox) * m, oy=p["y"] - (y - oy))
            else:
                b.sel = None
            return
        for i in reversed(range(len(b.creations))):
            c = b.creations[i]
            if c["anchor"] == "MONDE" and c.get("vis", True):
                sw_, sh_ = creation_span(c)
                if math.hypot(x - c["wx"], y - c["wy"]) < max(sw_, sh_) * c["ws"] * c["adj"] * 0.6 + 24:
                    b.cur, b.sel = i, None
                    b.wdrag = dict(i=i, dx=c["wx"] - x, dy=c["wy"] - y)
                    return

    if ptr["down"]:
        if b.drag and b.drag["i"] < len(cr["parts"]):
            p = cr["parts"][b.drag["i"]]
            nx = (x - ox) * b.drag["m"] + b.drag["ox"]
            ny = (y - oy) + b.drag["oy"]
            nx, ny, sx, sy = snap_pos(cr, b.drag["i"], nx, ny)
            dx, dy = nx - p["x"], ny - p["y"]
            p["x"], p["y"] = float(nx), float(ny)
            gid = p.get("grp")
            if gid is not None:
                for q in cr["parts"]:
                    if q is not p and q.get("grp") == gid:
                        q["x"] += dx
                        q["y"] += dy
            b.snap = (sx, sy, now)
        if b.wdrag:
            c = b.creations[b.wdrag["i"]]
            nx, ny = x + b.wdrag["dx"], y + b.wdrag["dy"]
            mv = nx - c["wx"]
            c["roll"] += mv * 0.06
            if abs(mv) > 3:
                c["face"] = 1 if mv > 0 else -1
            c["wx"], c["wy"] = float(nx), float(ny)
        if b.hold and now >= b.hold_t:
            do_button(app, b.hold, now)
            b.hold_t = now + 0.07

    if ptr["release"]:
        if b.pal is not None and inbench:
            cr["parts"].append(new_part(PARTS[b.pal][0], x - ox, y - oy, b.color))
            b.sel = len(cr["parts"]) - 1
            BEEP(1100, 40)
        b.pal = b.drag = b.wdrag = b.hold = None

def draw_help(out):
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    out[:] = cv2.addWeighted(overlay, 0.85, out, 0.15, 0)
    lines = [
        "D.I.V.A v3 - AIDE",
        "",
        "MODES: 1 Dessin | 2 Visages | 3 Objets | 4 Mouvement | 5 Corps | 6 Construction",
        "GLOBAL: a alarme | b fond | s capture | m son | h aide | q quitter",
        "",
        "DESSIN: index = dessiner | main ouverte 1s = effacer | c = effacer | z = annuler",
        "        f = correction ON/OFF | v = couleur | e = export PNG transparent",
        "",
        "VISAGES: r = memoriser (tape le nom + Entree) | +/- = seuil",
        "",
        "CONSTRUCTION: pince/click = prendre/poser | [ ] rotation | + - taille",
        "              x suppr | d copie | y symetrie | w ancrage | n nouveau",
        "              tab = creation suivante | v couleur | m masquer | z annuler",
        "              k sauver | g groupe | o opacite | p pause animation",
        "",
        "ANCRAGES: MONDE TETE TORSE DOS EPAULES MAINS AVANT-BRAS BOTTES",
        "",
        "Appuie sur h pour fermer cette aide",
    ]
    y = 40
    for line in lines:
        col = CYAN if line.startswith("D.I.V.A") else WHITE
        txt(out, line, (40, y), 0.5, col)
        y += int(26 * UI.text_scale)
    return out

def mode_build(app, frame):
    h, w = frame.shape[:2]
    now = time.time()
    b = app.build
    lay = layout(w, h)
    hand = app.model("hand1").process(app.rgb)
    lm = hand.multi_hand_landmarks[0].landmark if hand.multi_hand_landmarks else None
    ptr = app.pointer.update(lm, w, h, now, app.mouse)
    pose = app.model("pose_seg").process(app.rgb)
    plm = pose.pose_landmarks.landmark if pose.pose_landmarks else None
    build_interact(app, ptr, lay, now)
    cr = b.cr
    bx0, by0, bx1, by1 = lay["bench"]
    ox, oy = (bx0 + bx1) // 2, (by0 + by1) // 2

    base = cv2.convertScaleAbs(frame, alpha=0.30) if app.dark else frame.copy()
    if app.dark:
        base[:, :, 0] = cv2.add(base[:, :, 0], 18)
    zeros = lambda: np.zeros((h, w, 3), np.uint8)
    fill_f, line_f, fill_b, line_b, fill_u, line_u = (zeros() for _ in range(6))
    an0 = dict(t=now, roll=0.0, flick=0.85 + 0.15 * math.sin(now * 40) + 0.05 * random.random())
    back_used = worn = False

    for c in b.creations:
        if not c.get("vis", True) or not c["parts"]:
            continue
        an = dict(an0, roll=c["roll"]) if c.get("anim", True) else dict(an0, roll=0.0, flick=0.5)
        if c["anchor"] == "MONDE":
            insts = [dict(c=np.array([c["wx"], c["wy"]], np.float32), s=c["ws"] * c["adj"],
                          rot=0.0, flip=c["face"], back=False)]
        else:
            worn = True
            sw_, _ = creation_span(c)
            insts = body_instances(c["anchor"], plm, w, h, sw_) if plm else []
            for it in insts:
                it["s"] *= c["adj"]
        for it in insts:
            if it["back"]:
                back_used = True
                draw_instance(fill_b, line_b, c["parts"], it, an, c["sym"])
            else:
                draw_instance(fill_f, line_f, c["parts"], it, an, c["sym"])

    an_b = dict(an0, roll=cr["roll"]) if cr.get("anim", True) else dict(an0, roll=0.0, flick=0.5)
    draw_instance(fill_u, line_u, cr["parts"], dict(c=(ox, oy), s=1.0, rot=0.0, flip=1), an_b, cr["sym"])
    if b.sel is not None and b.sel < len(cr["parts"]):
        p = cr["parts"][b.sel]
        box = _rot(np.array([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)], np.float32) * [p["w"], p["h"]],
                   math.radians(p["r"])) + [p["x"] + ox, p["y"] + oy]
        cv2.polylines(line_u, [np.int32(box)], True, WHITE, 1, cv2.LINE_AA)
    ss = lay["ss"]
    for i, (name, _) in enumerate(PARTS):
        r = lay["slots"][i]
        icon_size = ss * 0.62
        icon = dict(k=name, x=0.0, y=0.0, w=icon_size, h=icon_size, r=0.0, c=b.color, op=1.0, grp=None)
        try:
            draw_instance(fill_u, line_u, [icon],
                          dict(c=((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), s=1.0, rot=0.0, flip=1),
                          dict(an0, roll=0.0), False)
        except Exception:
            pass
    if b.pal is not None and ptr:
        ghost = new_part(PARTS[b.pal][0], 0, 0, b.color)
        draw_instance(fill_u, line_u, [ghost],
                      dict(c=(ptr["x"], ptr["y"]), s=1.0, rot=0.0, flip=1),
                      dict(an0, roll=0.0), False)

    out = base
    if back_used:
        seg = pose.segmentation_mask
        if seg is not None:
            alpha = (1.0 - np.clip((cv2.resize(seg, (w, h)) - 0.35) / 0.3, 0, 1))[..., None]
            fill_b = (fill_b * alpha).astype(np.uint8)
            line_b = (line_b * alpha).astype(np.uint8)
        out = add_holo(out, fill_b, line_b)
    px1 = 12 + lay["cols"] * (ss + 6)
    py1 = 46 + ((len(PARTS) + lay["cols"] - 1) // lay["cols"]) * (ss + 6)
    for (x0, y0, x1, y1) in ((bx0, by0, bx1, by1), (6, 40, px1, py1)):
        out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    for gx in range(bx0 + (ox - bx0) % 20, bx1, 20):
        cv2.line(out, (gx, by0), (gx, by1), (60, 45, 20), 1)
    for gy in range(by0 + (oy - by0) % 20, by1, 20):
        cv2.line(out, (bx0, gy), (bx1, gy), (60, 45, 20), 1)
    cv2.line(out, (ox - 8, oy), (ox + 8, oy), (140, 120, 40), 1)
    cv2.line(out, (ox, oy - 8), (ox, oy + 8), (140, 120, 40), 1)
    if cr["sym"]:
        for yy in range(by0, by1, 12):
            cv2.line(out, (ox, yy), (ox, yy + 6), (200, 170, 40), 1)
    if now - b.snap[2] < 0.4:
        if b.snap[0] is not None:
            cv2.line(out, (int(ox + b.snap[0]), by0), (int(ox + b.snap[0]), by1), WHITE, 1)
        if b.snap[1] is not None:
            cv2.line(out, (bx0, int(oy + b.snap[1])), (bx1, int(oy + b.snap[1])), WHITE, 1)
    out = add_holo(out, cv2.add(fill_f, fill_u), cv2.add(line_f, line_u))

    cv2.rectangle(out, (bx0, by0), (bx1, by1), CYAN, 1)
    txt(out, "ATELIER - %s%s" % (cr["name"], "" if cr.get("vis", True) else " (masquee)"),
        (bx0 + 6, by0 - 8), 0.5)
    labels = {"ANCRAGE": "ANCR: " + cr["anchor"],
              "CREATION": "CRE %d/%d" % (b.cur + 1, len(b.creations)),
              "COULEUR": COLOR_NAMES[b.color],
              "SYMETRIE": "SYM " + ("ON" if cr["sym"] else "OFF"),
              "MASQUER": "AFFICHER" if not cr.get("vis", True) else "MASQUER",
              "GROUPE": "GRP", "OPACITE": "OPA"}
    hover = None
    for label, r in lay["buttons"]:
        hot = ptr is not None and inside(r, ptr["x"], ptr["y"])
        if hot:
            hover = label
        ov = out[r[1]:r[3], r[0]:r[2]]
        ov[:] = (ov * 0.3 + np.array((90, 60, 10) if hot else (40, 25, 0)) * 0.7).astype(np.uint8)
        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), CYAN if hot else (150, 120, 30), 1)
        txt(out, labels.get(label, label), (r[0] + 4, r[3] - 8), 0.38, WHITE if hot else CYAN)
    for i, r in enumerate(lay["slots"]):
        hot = ptr is not None and inside(r, ptr["x"], ptr["y"])
        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), WHITE if hot else (120, 100, 30), 1)
        if hot:
            hover = PARTS[i][0].upper()
    if ptr:
        c = (int(ptr["x"]), int(ptr["y"]))
        cv2.circle(out, c, 10, GREEN if ptr["down"] else WHITE, -1 if ptr["down"] else 2, cv2.LINE_AA)
    if worn and not plm:
        txt(out, "Place-toi face camera pour porter la creation",
            (w // 2 - 260, h - 44), 0.55, ORANGE)
    info = "%s | ancrage: %s | pieces: %d%s" % (cr["name"], cr["anchor"], len(cr["parts"]),
                                                 (" | " + hover) if hover else "")
    return out, info

# ------------------------------------------------------------------ application
class App:
    def __init__(self, model_path, hq=False, imgsz=640):
        self.mode, self.canvas, self.dark = "DESSIN", Canvas(), True
        self.db, self.thresh, self.enroll = load_db(), 0.10, None
        self.typing, self.buf = False, ""
        self.yolo_path, self.last_det = model_path, None
        self.yolo_imgsz = imgsz
        self.motion, self.alarm, self._motion_prev = Motion(), Alarm(), False
        self.build = Build()
        load_builds(self.build)
        self.pointer, self.mouse = Pointer(), None
        self.frame_i, self._m = 0, {}
        self.msg, self.hq, self.rgb = ("", 0.0), hq, None
        self._last_frame = None

    def model(self, name):
        if name not in self._m:
            H, F, P = mp.solutions.hands, mp.solutions.face_mesh, mp.solutions.pose
            comp = 1 if self.hq else 0
            makers = {
                "hand1": lambda: H.Hands(max_num_hands=1, model_complexity=comp,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7),
                "hand2": lambda: H.Hands(max_num_hands=2, model_complexity=comp),
                "face": lambda: F.FaceMesh(max_num_faces=4,
                                           min_detection_confidence=0.6,
                                           min_tracking_confidence=0.6),
                "pose": lambda: P.Pose(model_complexity=comp),
                "pose_seg": lambda: P.Pose(model_complexity=1 if self.hq else 0,
                                           enable_segmentation=True,
                                           smooth_segmentation=True,
                                           min_detection_confidence=0.6,
                                           min_tracking_confidence=0.6),
                "yolo": self._load_yolo,
            }
            self._m[name] = makers[name]()
        return self._m[name]

    def _load_yolo(self):
        from ultralytics import YOLO
        print("[DIVA] chargement YOLO...")
        return YOLO(self.yolo_path)

    def say(self, s, secs=2.5):
        self.msg = (s, time.time() + secs)

    def on_mouse(self, event, x, y, flags, param):
        m = self.mouse or dict(x=0, y=0, down=False, t=0.0)
        m.update(x=x, y=y, t=time.time())
        if event == cv2.EVENT_LBUTTONDOWN:
            m["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            m["down"] = False
        self.mouse = m

# ------------------------------------------------------------------ modes
def mode_draw(app, frame):
    h, w = frame.shape[:2]
    res = app.model("hand1").process(app.rgb)
    lm = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None
    state = app.canvas.update(lm, w, h)

    base = cv2.convertScaleAbs(frame, alpha=0.30) if app.dark else frame.copy()
    if app.dark:
        base[:, :, 0] = cv2.add(base[:, :, 0], 18)
    layer, glow = render_layers(app.canvas.shapes, app.canvas.cur, (h, w))
    out = cv2.add(base, cv2.add(glow, layer))
    out[::3] = (out[::3] * 0.85).astype(np.uint8)

    if app.canvas.last is not None:
        c = {"draw": GREEN, "move": ORANGE, "palm": RED}.get(state, WHITE)
        cv2.circle(out, (int(app.canvas.last[0]), int(app.canvas.last[1])), 9, c, 2, cv2.LINE_AA)
    if app.canvas.palm_t is not None and 0 <= time.time() - app.canvas.palm_t < 1.2:
        pr = min(1.0, (time.time() - app.canvas.palm_t) / 1.2)
        cv2.ellipse(out, (w // 2, h // 2), (40, 40), 0, 0, int(360 * pr), RED, 4, cv2.LINE_AA)
        txt(out, "EFFACEMENT...", (w // 2 - 60, h // 2 + 70), 0.6, RED)
    info = "correction: %s | couleur: %s | formes: %d" % (
        "ON" if app.canvas.correct else "OFF",
        COLOR_NAMES[app.canvas.color], len(app.canvas.shapes))
    return out, info, layer, glow

def mode_faces(app, frame):
    h, w = frame.shape[:2]
    res = app.model("face").process(app.rgb)
    out = cv2.convertScaleAbs(frame, alpha=0.6) if app.dark else frame.copy()
    n = 0
    if res.multi_face_landmarks:
        faces = sorted(
            res.multi_face_landmarks,
            key=lambda f: -(
                (max(l.x for l in f.landmark) - min(l.x for l in f.landmark))
                * (max(l.y for l in f.landmark) - min(l.y for l in f.landmark))
            )
        )
        for i, fl in enumerate(faces):
            n += 1
            lm = fl.landmark
            x1, x2 = int(min(l.x for l in lm) * w), int(max(l.x for l in lm) * w)
            y1, y2 = int(min(l.y for l in lm) * h), int(max(l.y for l in lm) * h)
            sig = face_signature(lm, w, h)
            if sig is None:
                brackets(out, x1, y1, x2, y2, ORANGE)
                continue
            for l in lm[::6]:
                cv2.circle(out, (int(l.x * w), int(l.y * h)), 1, CYAN, -1)
            if app.enroll is not None and i == 0:
                e = app.enroll
                if time.time() - e["t"] > 0.12:
                    e["samples"].append(sig)
                    e["t"] = time.time()
                brackets(out, x1, y1, x2, y2, ORANGE)
                txt(out, "MEMORISATION %s %d/25 - tourne un peu la tete" % (
                    e["name"], len(e["samples"])), (x1, y1 - 10), 0.5, ORANGE)
                if len(e["samples"]) >= 25:
                    app.db[e["name"]] = np.stack(e["samples"])
                    try:
                        save_db(app.db)
                        app.say("Visage memorise : " + e["name"])
                        BEEP(1400, 150)
                    except OSError as exc:
                        app.say("Erreur sauvegarde visage : " + str(exc)[:55], 4.0)
                        app.db.pop(e["name"], None)
                    app.enroll = None
                continue
            name, d = identify(sig, app.db)
            ok = name is not None and d < app.thresh
            col = GREEN if ok else RED
            brackets(out, x1, y1, x2, y2, col)
            txt(out, ("%s (d=%.3f)" % (name, d)) if ok else ("INCONNU (d=%.3f)" % d if name else "INCONNU"),
                (x1, y1 - 10), 0.6, col)
    info = "visages: %d | connus: %d | seuil: %.2f" % (n, len(app.db), app.thresh)
    return out, info

def mode_objects(app, frame):
    try:
        yolo = app.model("yolo")
    except Exception as exc:
        out = cv2.convertScaleAbs(frame, alpha=0.6) if app.dark else frame.copy()
        txt(out, "YOLO indisponible : %s" % str(exc)[:90], (20, 70), 0.55, RED, 2)
        return out, "erreur chargement YOLO"
    if app.frame_i % 2 == 0 or app.last_det is None:
        try:
            r = yolo.predict(frame, imgsz=app.yolo_imgsz, conf=0.35, iou=0.5, verbose=False)[0]
        except Exception as exc:
            out = cv2.convertScaleAbs(frame, alpha=0.6) if app.dark else frame.copy()
            txt(out, "Erreur inference YOLO : %s" % str(exc)[:90], (20, 70), 0.55, RED, 2)
            return out, "erreur inference YOLO"
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append((x1, y1, x2, y2, float(b.conf[0]), r.names[int(b.cls[0])]))
        app.last_det = dets
    out = cv2.convertScaleAbs(frame, alpha=0.6) if app.dark else frame.copy()
    for x1, y1, x2, y2, c, name in app.last_det:
        brackets(out, x1, y1, x2, y2)
        txt(out, "%s %d%%" % (name, c * 100), (x1, max(15, y1 - 8)), 0.55)
    cnt = Counter(d[5] for d in app.last_det)
    return out, "objets: %d  %s" % (len(app.last_det), dict(cnt))

def mode_motion(app, frame):
    h, w = frame.shape[:2]
    mo = app.motion
    out = cv2.convertScaleAbs(frame, alpha=0.5)
    if mo.mask is not None:
        tint = np.zeros_like(frame)
        tint[cv2.resize(mo.mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0] = CYAN
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    for x1, y1, x2, y2 in mo.boxes:
        brackets(out, x1, y1, x2, y2)
    if mo.warm < 25:
        txt(out, "CALIBRAGE DU FOND... ne bouge pas", (20, 70), 0.7, ORANGE, 2)
    elif mo.boxes:
        txt(out, "MOUVEMENT DETECTE", (20, 70), 0.8, RED, 2)
    al = app.alarm
    state = "ARMEE" if al.armed else "OFF (touche a)"
    return out, "zones: %d | seuil zone: %d | alarme: %s" % (len(mo.boxes), mo.min_area, state)

def mode_body(app, frame):
    out = cv2.convertScaleAbs(frame, alpha=0.5) if app.dark else frame.copy()
    d = mp.solutions.drawing_utils
    spec = d.DrawingSpec(color=CYAN, thickness=2, circle_radius=2)
    pr = app.model("pose").process(app.rgb)
    if pr.pose_landmarks:
        d.draw_landmarks(out, pr.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, spec, spec)
    hr = app.model("hand2").process(app.rgb)
    nh = 0
    if hr.multi_hand_landmarks:
        for hl in hr.multi_hand_landmarks:
            nh += 1
            d.draw_landmarks(out, hl, mp.solutions.hands.HAND_CONNECTIONS, spec, spec)
    return out, "posture: %s | mains: %d" % ("oui" if pr.pose_landmarks else "non", nh)

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--hq", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ui-scale", type=float, default=1.0)
    ap.add_argument("--text-scale", type=float, default=1.0)
    ap.add_argument("--high-contrast", action="store_true")
    ap.add_argument("--cb-mode", action="store_true")
    ap.add_argument("--flip", type=int, default=1)
    ap.add_argument("--custom-piece", default=None)
    ap.add_argument("--export-frames", action="store_true")
    args = ap.parse_args()
    if args.w < 160 or args.h < 120:
        ap.error("--w doit etre >= 160 et --h >= 120")
    if args.imgsz < 160:
        ap.error("--imgsz doit etre >= 160")
    if args.flip not in (0, 1):
        ap.error("--flip accepte uniquement 0 ou 1")

    init_ui(args)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    if not cap.isOpened():
        print("Camera introuvable. Essaie --cam 1")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    app = App(args.model, args.hq, args.imgsz)
    cv2.namedWindow("D.I.V.A", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("D.I.V.A", app.on_mouse)
    fps, t0 = 0.0, time.time()
    layer = glow = None
    frame_save_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            app.frame_i += 1
            small = frame if w <= 640 else cv2.resize(
                frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
            app.rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            active = app.mode == "MOUVEMENT" or app.alarm.armed
            if active and not app._motion_prev:
                app.motion.reset()
            app._motion_prev = active
            if active:
                app.motion.process(frame)
                app.alarm.check(app.motion, frame)

            info = ""
            if app.mode == "DESSIN":
                out, info, layer, glow = mode_draw(app, frame)
            elif app.mode == "VISAGES":
                out, info = mode_faces(app, frame)
            elif app.mode == "OBJETS":
                out, info = mode_objects(app, frame)
            elif app.mode == "MOUVEMENT":
                out, info = mode_motion(app, frame)
            elif app.mode == "CONSTRUCTION":
                out, info = mode_build(app, frame)
            else:
                out, info = mode_body(app, frame)

            if UI.export_frames and app.mode == "CONSTRUCTION":
                p = os.path.join(OUT_DIR, "frame_%05d.png" % frame_save_idx)
                cv2.imwrite(p, out)
                frame_save_idx += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t0)
            t0 = now
            h, w = out.shape[:2]
            cv2.rectangle(out, (0, 0), (w, int(34 * UI.scale)), (30, 20, 0), -1)
            txt(out, "D.I.V.A v3 | %s | %s" % (app.mode, info), (12, 23), 0.55)
            al = app.alarm
            al_txt = "ALARME !!" if al.ringing else ("ALARME ARMEE" if al.armed else "alarme off")
            txt(out, "%s | %d FPS" % (al_txt, fps), (w - 230, 23), 0.5,
                RED if al.ringing else CYAN)
            txt(out, "1 Dessin 2 Visages 3 Objets 4 Mouvement 5 Corps 6 Construction | "
                     "a alarme b fond s capture m son h aide q quitter",
                (12, h - 28), 0.42, (200, 200, 200))
            if app.canvas.toast[1] > now and app.mode == "DESSIN":
                txt(out, app.canvas.toast[0], (w // 2 - 110, 70), 0.8, GREEN, 2)
            if app.msg[1] > now:
                txt(out, app.msg[0], (w // 2 - 150, 100), 0.8, GREEN, 2)
            draw_alarm(out, al)
            if app.typing:
                cv2.rectangle(out, (w // 2 - 260, h // 2 - 30), (w // 2 + 260, h // 2 + 30), (30, 20, 0), -1)
                cv2.rectangle(out, (w // 2 - 260, h // 2 - 30), (w // 2 + 260, h // 2 + 30), CYAN, 2)
                txt(out, "Nom : " + app.buf + "_", (w // 2 - 245, h // 2 + 8), 0.8, WHITE, 2)
            if UI.show_help:
                out = draw_help(out)

            cv2.imshow("D.I.V.A", out)
            k = cv2.waitKey(1) & 0xFF

            if app.typing:
                if k == 13:
                    if app.buf.strip():
                        app.enroll = {"name": app.buf.strip(), "samples": [], "t": 0.0}
                    app.typing, app.buf = False, ""
                elif k == 27:
                    app.typing, app.buf = False, ""
                elif k == 8:
                    app.buf = app.buf[:-1]
                elif 32 <= k < 127:
                    app.buf += chr(k)
                continue

            if k in (ord("q"), 27):
                break
            elif k in MODES:
                new_mode = MODES[k]
                if new_mode != app.mode:
                    app.mode = new_mode
                    if new_mode == "OBJETS":
                        app.last_det = None
            elif k == ord("a"):
                if al.armed:
                    al.disarm()
                    app.say("Alarme desactivee")
                else:
                    al.arm(app.motion)
                    app.say("Alarme armee dans 5 s")
            elif k == ord("b"):
                app.dark = not app.dark
            elif k == ord("m") and app.mode != "CONSTRUCTION":
                UI.sound = not UI.sound
                app.say("Son: " + ("ON" if UI.sound else "OFF"))
            elif k == ord("h"):
                UI.show_help = not UI.show_help
            elif k == ord("s"):
                p = os.path.join(OUT_DIR, "diva_%d.png" % int(time.time()))
                Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(p)
                app.say("Capture : " + p)
            elif app.mode == "DESSIN":
                if k == ord("c"):
                    app.canvas.clear()
                elif k == ord("z"):
                    app.canvas.undo()
                elif k == ord("f"):
                    app.canvas.correct = not app.canvas.correct
                elif k == ord("v"):
                    app.canvas.color = (app.canvas.color + 1) % len(COLORS)
                elif k == ord("e") and layer is not None:
                    hol = cv2.add(layer, glow)
                    rgba = np.dstack([cv2.cvtColor(hol, cv2.COLOR_BGR2RGB), hol.max(axis=2)])
                    p = os.path.join(OUT_DIR, "hologramme_%d.png" % int(time.time()))
                    Image.fromarray(rgba, "RGBA").save(p)
                    app.say("Export transparent : " + p)
            elif app.mode == "VISAGES":
                if k == ord("r"):
                    app.typing = True
                elif k in (ord("+"), ord("=")):
                    app.thresh = min(0.5, app.thresh + 0.01)
                elif k == ord("-"):
                    app.thresh = max(0.02, app.thresh - 0.01)
            elif app.mode == "MOUVEMENT":
                if k in (ord("+"), ord("=")):
                    app.motion.min_area = max(100, int(app.motion.min_area / 1.25))
                elif k == ord("-"):
                    app.motion.min_area = min(20000, int(app.motion.min_area * 1.25))
            elif app.mode == "CONSTRUCTION":
                if k in BUILD_KEYS:
                    do_button(app, BUILD_KEYS[k], time.time())
                elif k == ord("z") and app.build.cr["parts"]:
                    app.build.cr["parts"].pop()
                    app.build.sel = None
                elif k == ord("p"):
                    app.build.cr["anim"] = not app.build.cr.get("anim", True)

    finally:
        app.alarm.disarm()
        try:
            if any(c["parts"] for c in app.build.creations):
                save_builds(app.build)
        except Exception as exc:
            print("[DIVA] sauvegarde finale impossible :", exc)
        for model in app._m.values():
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
