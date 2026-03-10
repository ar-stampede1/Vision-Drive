import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(
    page_title="VisionDrive",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

@st.cache_resource

CLASS_NAMES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]
SPEED_CLASSES = {c: int(c.split()[-1]) for c in CLASS_NAMES if c.startswith('Speed Limit')}
BOX_COLORS = {'Green Light': (0, 255, 120), 'Red Light': (255, 30, 70), 'Stop': (255, 30, 70)}
for cls in SPEED_CLASSES:
    BOX_COLORS[cls] = (255, 190, 0)


# ── Load CSS ───────────────────────────────────────────────────────────────────
def load_css(path) -> None:
    css_file = Path(path)
    if css_file.exists():
        css = css_file.read_text(encoding="utf-8")
        st.markdown(
            '<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900'
            '&family=Barlow+Condensed:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400'
            '&family=Share+Tech+Mono&display=swap" rel="stylesheet">',
            unsafe_allow_html=True,
        )
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"style.css not found at: {path}")

load_css(Path(__file__).parent / "style.css")


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_speed_limit(dets):
    best, bc = None, 0
    for d in dets:
        if d['label'] in SPEED_CLASSES and d['conf'] > bc:
            best, bc = d['label'], d['conf']
    return best


def draw_boxes(frame, dets, thresh):
    for d in dets:
        if d['conf'] < thresh:
            continue
        x1, y1, x2, y2 = d['box']
        c = BOX_COLORS.get(d['label'], (0, 200, 255))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, -1)
        cv2.addWeighted(overlay, .08, frame, .92, 0, frame)
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), tuple(int(v*.25) for v in c), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"{d['label']}  {d['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .52, 1)
        cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+10, y1), c, -1)
        cv2.putText(frame, label, (x1+5, y1-4), cv2.FONT_HERSHEY_SIMPLEX, .52, (4, 8, 14), 1, cv2.LINE_AA)
    return frame


def run_inference(mdl, frame, thresh):
    dets = []
    try:
        results = mdl(frame, conf=thresh, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if 0 <= cls_id < len(CLASS_NAMES):
                    dets.append({
                        'label': CLASS_NAMES[cls_id],
                        'conf':  float(box.conf[0]),
                        'box':   tuple(map(int, box.xyxy[0]))
                    })
    except Exception:
        pass
    return dets


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model_err = ""
try:
    model = load_model()
    model_ok = True
except Exception as e:
    model = None
    model_ok = False
    model_err = str(e)


# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    source_mode="webcam",
    driver_speed=60,
    conf_thresh=0.40,
    dets=[],
    last_sign="---",
    last_speed_sign=None,
    cam_running=False,
    img_annotated=None,
    img_name="",
    vid_running=False,
    vid_tmp_path="",
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Source mode selector — instant, no query-param reload ─────────────────────
# Rendered as hidden Streamlit widgets; JS left panel calls
# window.parent.postMessage to trigger st.rerun via a tiny receiver component.
# Speed + conf now use st.slider → zero-latency session state update.


def update_state(new_dets):
    st.session_state.dets = new_dets
    if new_dets:
        st.session_state.last_sign       = new_dets[0]['label']
        st.session_state.last_speed_sign = get_speed_limit(new_dets)
    else:
        # Keep last_sign sticky — only reset speed sign when no dets
        st.session_state.last_speed_sign = None


# ── Invisible Streamlit controls (speed, conf, source) ────────────────────────
# These live outside visible columns so they don't affect layout.
# JS in the left panel posts messages → a tiny JS snippet in the page
# sets hidden number inputs → Streamlit reads them on next interaction.
#
# SIMPLER APPROACH: use st.session_state directly via on_change callbacks
# with st.slider (which re-runs instantly without any page navigation).

with st.container():
    # Hidden row — zero-height visually, but Streamlit still processes it
    _hc1, _hc2, _hc3 = st.columns([1, 1, 1])
    with _hc1:
        new_speed = st.slider(
            "speed_hidden", 0, 140,
            st.session_state.driver_speed,
            step=5,
            key="speed_slider",
            label_visibility="collapsed",
        )
        if new_speed != st.session_state.driver_speed:
            st.session_state.driver_speed = new_speed

    with _hc2:
        new_conf = st.slider(
            "conf_hidden", 10, 95,
            int(st.session_state.conf_thresh * 100),
            step=5,
            key="conf_slider",
            label_visibility="collapsed",
        )
        if new_conf != int(st.session_state.conf_thresh * 100):
            st.session_state.conf_thresh = new_conf / 100

    with _hc3:
        src_map  = {"webcam": 0, "image": 1, "video": 2}
        src_rmap = {0: "webcam", 1: "image", 2: "video"}
        new_src_idx = st.radio(
            "src_hidden",
            options=[0, 1, 2],
            format_func=lambda x: src_rmap[x],
            index=src_map[st.session_state.source_mode],
            key="src_radio",
            label_visibility="collapsed",
            horizontal=True,
        )
        new_src = src_rmap[new_src_idx]
        if new_src != st.session_state.source_mode:
            st.session_state.source_mode     = new_src
            st.session_state.cam_running     = False
            st.session_state.vid_running     = False
            st.session_state.img_annotated   = None
            st.session_state.dets            = []
            st.session_state.last_sign       = "---"
            st.session_state.last_speed_sign = None


# ── Read current state ─────────────────────────────────────────────────────────
driver_speed    = st.session_state.driver_speed
source_mode     = st.session_state.source_mode
conf_thresh     = st.session_state.conf_thresh
dets            = st.session_state.dets

# ── Recompute speed limit from CURRENT dets every render ──────────────────────
# This is the key fix: speed_limit_val is always derived fresh from the
# current detection list AND the current driver_speed — no stale state.
speed_sign_label = get_speed_limit(dets)
speed_limit_val  = SPEED_CLASSES.get(speed_sign_label)   # int or None
last_sign        = st.session_state.last_sign


# ── HTML card builders ─────────────────────────────────────────────────────────
def _wcard(icon, title, desc, color, bg, anim):
    a = f"animation:{anim};" if anim else ""
    return (
        f'<div class="warn-card" style="border-color:{color}44;background:{bg};{a}">'
        f'<div class="warn-l" style="background:{color}1a;border-right:1px solid {color}33">'
        f'<span class="warn-ico">{icon}</span></div>'
        f'<div class="warn-b"><div class="warn-t" style="color:{color}">{title}</div>'
        f'<div class="warn-d">{desc}</div></div></div>'
    )


def build_warn_html(dets, speed, spd_lim):
    has_stop  = any(d['label'] == 'Stop'        for d in dets)
    has_red   = any(d['label'] == 'Red Light'   for d in dets)
    has_green = any(d['label'] == 'Green Light' for d in dets)
    spd_sign  = get_speed_limit(dets)
    w = ""
    if has_stop:
        w += _wcard("🛑", "STOP SIGN", "Come to a complete stop immediately.",
                    "#ff2244", "rgba(255,34,68,.13)", "pulseCard 1.2s ease-in-out infinite")
    if has_red:
        w += _wcard("🔴", "RED LIGHT", "Do not proceed through intersection.",
                    "#ff2244", "rgba(255,34,68,.13)", "pulseCard 1.2s ease-in-out infinite")
    if spd_sign:
        lim = SPEED_CLASSES[spd_sign]
        if speed > lim:
            w += _wcard("🚨", f"OVER LIMIT — {lim} KM/H",
                        f"⚡ {speed} km/h detected · Reduce by {speed-lim} km/h NOW",
                        "#ff2244", "rgba(255,34,68,.18)", "pulseCard .8s ease-in-out infinite")
        else:
            w += _wcard("✓", f"SPEED OK — {lim} KM/H",
                        f"Within zone limit at {speed} km/h",
                        "#00ffaa", "rgba(0,255,170,.08)", "")
    if has_green and not has_red:
        w += _wcard("🟢", "GREEN LIGHT", "Intersection clear — proceed safely.",
                    "#00ffaa", "rgba(0,255,170,.08)", "")
    if not w:
        w += _wcard("◉", "SCANNING", "No traffic signs detected in frame.",
                    "#2a4a60", "rgba(15,25,40,.8)", "")
    return w


def build_det_html(dets):
    h = ""
    for d in sorted(dets, key=lambda x: x['conf'], reverse=True)[:6]:
        if d['label'].startswith('Speed'):
            bc, tc, tag = "#ffaa00", "rgba(255,170,0,.1)", "SPD"
        elif d['label'] in ('Red Light', 'Stop'):
            bc, tc, tag = "#ff2244", "rgba(255,34,68,.1)", "DNG"
        else:
            bc, tc, tag = "#00ffaa", "rgba(0,255,170,.08)", "SIG"
        h += (
            f'<div class="det-row">'
            f'<span class="det-lbl">{d["label"]}</span>'
            f'<span class="det-tag" style="color:{bc};border-color:{bc}44;background:{tc}">{tag}</span>'
            f'<span class="det-conf" style="color:{bc}">{d["conf"]:.0%}</span></div>'
        )
    return h or '<div class="det-empty">No detections in frame</div>'


# ── Derived display values ─────────────────────────────────────────────────────
if speed_limit_val and driver_speed > speed_limit_val:
    sc = "#ff2244"
elif driver_speed > 100:
    sc = "#ffaa00"
else:
    sc = "#00d4ff"

if speed_limit_val:
    lb_col = "#ff2244" if driver_speed > speed_limit_val else "#00ffaa"
    lb_bg  = "rgba(255,34,68,.1)" if driver_speed > speed_limit_val else "rgba(0,255,170,.07)"
    lb_txt = "⚠ OVER LIMIT" if driver_speed > speed_limit_val else f"LIMIT {speed_limit_val} km/h"
else:
    lb_col, lb_bg, lb_txt = "#2a4a60", "transparent", "NO LIMIT SIGN"

alert_color = (
    "#ff2244" if (any(d['label'] in ('Stop', 'Red Light') for d in dets) or
                  (speed_limit_val and driver_speed > speed_limit_val))
    else "#00ffaa" if (any(d['label'] == 'Green Light' for d in dets) or
                       (speed_limit_val and driver_speed <= speed_limit_val))
    else "#00d4ff"
)

NUM_SEGS  = 24
lit_segs  = round(min(driver_speed / 140, 1) * NUM_SEGS)
segs_html = "".join(
    f'<div class="sd sd-{"g" if i<12 else "y" if i<18 else "r"} {"lit" if i<lit_segs else ""}"></div>'
    for i in range(NUM_SEGS)
)

conf_pct    = int(conf_thresh * 100)
spd_lim_js  = speed_limit_val if speed_limit_val else 0
warn_html   = build_warn_html(dets, driver_speed, speed_limit_val)
det_html    = build_det_html(dets)

feed_is_live = (
    (source_mode == "webcam" and st.session_state.cam_running) or
    (source_mode == "image"  and st.session_state.img_annotated is not None) or
    (source_mode == "video"  and st.session_state.vid_running)
)

standby_msg = {
    "webcam": ("CAMERA STANDBY",  "Press START CAMERA to begin"),
    "image":  ("NO IMAGE LOADED", "Upload an image file to run detection"),
    "video":  ("NO VIDEO LOADED", "Upload a video file, then press RUN"),
}[source_mode]


# ══════════════════════════════════════════════════════════════════════════════
#  SPEED ALERT BANNER
# ══════════════════════════════════════════════════════════════════════════════
is_over_limit = speed_limit_val is not None and driver_speed > speed_limit_val
over_by       = (driver_speed - speed_limit_val) if is_over_limit else 0

if is_over_limit:
    st.markdown(f"""
    <style>
    @keyframes bannerFlash {{
      0%,100% {{ opacity:1; background:rgba(255,34,68,.18); }}
      50%      {{ opacity:.7; background:rgba(255,34,68,.32); }}
    }}
    @keyframes bannerGlow {{
      0%,100% {{ box-shadow: 0 0 0px #ff2244; }}
      50%      {{ box-shadow: 0 0 24px #ff224488; }}
    }}
    .spd-over-banner {{
      display:flex; align-items:center; justify-content:space-between;
      padding:10px 28px;
      background:rgba(255,34,68,.18);
      border-bottom:2px solid #ff224466;
      animation: bannerFlash 1s ease-in-out infinite, bannerGlow 1s ease-in-out infinite;
      position:relative; z-index:9999;
    }}
    .sob-left  {{ display:flex; align-items:center; gap:14px; }}
    .sob-icon  {{ font-size:1.5rem; animation: iconBounce2 .5s ease-in-out infinite alternate; }}
    @keyframes iconBounce2 {{ 0%{{transform:scale(1);}} 100%{{transform:scale(1.25);}} }}
    .sob-title {{ font-family:'Orbitron',monospace; font-weight:900; font-size:.95rem;
                  color:#ff2244; letter-spacing:.1em;
                  text-shadow:0 0 14px #ff2244; }}
    .sob-sub   {{ font-family:'Share Tech Mono',monospace; font-size:.65rem;
                  color:#ff8899; letter-spacing:.1em; margin-top:2px; }}
    .sob-right {{ display:flex; align-items:center; gap:22px; }}
    .sob-stat  {{ text-align:center; }}
    .sob-stat-val {{ font-family:'Orbitron',monospace; font-weight:900;
                     font-size:1.4rem; color:#ff2244;
                     text-shadow:0 0 20px #ff2244; line-height:1; }}
    .sob-stat-lbl {{ font-family:'Share Tech Mono',monospace; font-size:.52rem;
                     color:#ff6677; letter-spacing:.14em; }}
    .sob-divider {{ width:1px; height:36px; background:#ff224433; }}
    .sob-pill {{ font-family:'Orbitron',monospace; font-size:.65rem; font-weight:700;
                 padding:6px 16px; border-radius:6px;
                 background:#ff224422; border:1px solid #ff224466;
                 color:#ff2244; letter-spacing:.08em;
                 text-shadow:0 0 10px #ff2244; }}
    </style>
    <div class="spd-over-banner">
      <div class="sob-left">
        <div class="sob-icon">⚠️</div>
        <div>
          <div class="sob-title">⚡ SPEED LIMIT EXCEEDED — SLOW DOWN</div>
          <div class="sob-sub">Detected limit: {speed_limit_val} km/h &nbsp;·&nbsp; Your speed: {driver_speed} km/h &nbsp;·&nbsp; Reduce by {over_by} km/h</div>
        </div>
      </div>
      <div class="sob-right">
        <div class="sob-stat">
          <div class="sob-stat-val">{driver_speed}</div>
          <div class="sob-stat-lbl">YOUR SPEED</div>
        </div>
        <div class="sob-divider"></div>
        <div class="sob-stat">
          <div class="sob-stat-val">{speed_limit_val}</div>
          <div class="sob-stat-lbl">LIMIT</div>
        </div>
        <div class="sob-divider"></div>
        <div class="sob-stat">
          <div class="sob-stat-val" style="color:#ffaa00">-{over_by}</div>
          <div class="sob-stat-lbl">REDUCE BY</div>
        </div>
        <div class="sob-pill">REDUCE SPEED NOW</div>
      </div>
    </div>
    <script>
    (function() {{
      function beep() {{
        try {{
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.connect(g); g.connect(ctx.destination);
          o.type = 'square';
          o.frequency.setValueAtTime(880, ctx.currentTime);
          o.frequency.setValueAtTime(660, ctx.currentTime + 0.15);
          g.gain.setValueAtTime(0.3, ctx.currentTime);
          g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
          o.start(ctx.currentTime);
          o.stop(ctx.currentTime + 0.4);
        }} catch(e) {{}}
      }}
      beep();
      setInterval(beep, 3000);
    }})();
    </script>
    """, unsafe_allow_html=True)


st.markdown(f"""
<div class="vd-topbar">
  <div class="tb-alert-bar"
       style="background:{alert_color};box-shadow:0 0 14px {alert_color},0 0 28px {alert_color}44;opacity:.8;">
  </div>
  <div class="logo">
    <div class="logo-hex">🚗</div>
    <div>
      <div class="logo-name">VISIONDRIVE</div>
      <div class="logo-tag">Traffic Intelligence · v2.0</div>
    </div>
  </div>
  <div class="tb-pills">
    <span class="pill pill-live">◉ LIVE</span>
    <span class="pill">YOLO v8</span>
    <span class="pill">15 CLASSES</span>
    <span class="pill" id="tb-conf-pill">CONF {conf_pct}%</span>
  </div>
  <div class="tb-right">
    <div class="led {'led-on' if model_ok else 'led-off'}"></div>
    <span class="tb-stat">{'MODEL ONLINE' if model_ok else 'MODEL ERROR'}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  3-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left_col, center_col, right_col = st.columns([238, 600, 272], gap="small")


# ─── LEFT PANEL ───────────────────────────────────────────────────────────────
# The JS panel is now DISPLAY-ONLY for speed / conf / source.
# All real state changes go through the Streamlit sliders/radio above
# (which are hidden but functional).  The JS drag on mouseup calls
# Streamlit's internal setComponentValue bridge via the hidden slider
# by programmatically moving it — no page navigation needed.
#
# HOW THE BRIDGE WORKS:
#   1. User drags the visual track in the iframe.
#   2. On mouseup, JS dispatches a CustomEvent("streamlit:setComponentValue")
#      that Streamlit's widget bridge picks up and syncs to session_state.
#   3. Streamlit reruns instantly (same as any slider change).
#
# Because the hidden sliders ARE the source of truth, the left panel's
# JS only needs to render the current value it receives via the `spd`
# and `conf_pct` template variables — it never owns state.

def build_left_panel(speed, conf_pct, sc, lb_col, lb_bg, lb_txt,
                     spd_lim_js, segs_html, source_mode):
    src_w = lambda s: "active" if source_mode == s else ""
    vt_pct = min(speed / 140, 1) * 100
    vt_top = 188 * (1 - min(speed / 140, 1)) - 14
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Barlow+Condensed:wght@400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
:root{{
  --ink:#020509;--ink2:#060e18;--ink3:#0b1928;--ink4:#101f30;
  --c1:#00d4ff;--c2:#00ffaa;--c3:#ff2244;--c4:#ffaa00;--c5:#7040ff;
  --txt:#daeeff;--txt2:#6a9ab8;--txt3:#3a5f7a;
  --bord:rgba(0,212,255,.15);--bord2:rgba(0,212,255,.07);
}}
html,body{{
  background:linear-gradient(175deg,#060e18 0%,#020509 100%);
  color:var(--txt);
  font-family:'Barlow Condensed',sans-serif;
  height:100%;overflow-y:auto;overflow-x:hidden;
  border-right:1px solid var(--bord);
  position:relative;
}}
body::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--c1),var(--c2));opacity:.45;pointer-events:none;}}
.sec{{display:flex;align-items:center;gap:8px;padding:13px 16px 9px;
  font-family:'Share Tech Mono',monospace;font-size:.62rem;letter-spacing:.22em;
  color:var(--txt2);text-transform:uppercase;border-bottom:1px solid var(--bord2);}}
.sd-led{{width:5px;height:5px;border-radius:50%;flex-shrink:0;}}
.src-wrap{{padding:10px 13px;display:flex;flex-direction:column;gap:5px;}}
.src-btn{{
  display:flex;align-items:center;gap:9px;padding:10px 13px;border-radius:8px;
  background:var(--ink3);border:1px solid var(--bord2);color:var(--txt2);
  font-family:'Barlow Condensed',sans-serif;font-size:.95rem;font-weight:500;
  cursor:pointer;transition:all .18s;letter-spacing:.03em;user-select:none;
}}
.src-btn:hover{{background:var(--ink4);border-color:var(--bord);color:var(--txt);}}
.src-btn.active{{
  background:rgba(0,212,255,.09);border-color:rgba(0,212,255,.38);
  color:var(--c1);box-shadow:0 0 16px rgba(0,212,255,.1);
}}
.conf-wrap{{padding:10px 14px 12px;}}
.conf-lbl{{display:flex;justify-content:space-between;align-items:center;
  font-family:'Share Tech Mono',monospace;font-size:.58rem;letter-spacing:.18em;
  color:var(--txt2);text-transform:uppercase;margin-bottom:9px;}}
.conf-lbl span{{color:var(--c1);font-size:.7rem;}}
input[type=range]{{
  -webkit-appearance:none;appearance:none;
  width:100%;height:4px;border-radius:2px;outline:none;cursor:pointer;
  background:linear-gradient(90deg,var(--c1) {conf_pct}%,rgba(0,212,255,.12) {conf_pct}%);
}}
input[type=range]::-webkit-slider-thumb{{
  -webkit-appearance:none;width:15px;height:15px;border-radius:50%;
  background:var(--c1);cursor:grab;
  box-shadow:0 0 10px var(--c1),0 0 22px rgba(0,212,255,.4);
}}
input[type=range]::-webkit-slider-thumb:active{{cursor:grabbing;}}
.spd-alert{{
  display:none;align-items:center;gap:10px;
  margin:10px 13px 0;padding:10px 14px;border-radius:9px;
  border:1px solid #ff224466;background:rgba(255,34,68,.13);
  animation:alertPulse 1s ease-in-out infinite;
}}
.spd-alert.active{{ display:flex; }}
@keyframes alertPulse{{
  0%,100%{{box-shadow:0 0 0 rgba(255,34,68,0);background:rgba(255,34,68,.13);}}
  50%{{box-shadow:0 0 22px rgba(255,34,68,.45);background:rgba(255,34,68,.22);}}
}}
.alert-icon{{font-size:1.3rem;flex-shrink:0;animation:iconBounce .6s ease-in-out infinite alternate;}}
@keyframes iconBounce{{0%{{transform:scale(1);}}100%{{transform:scale(1.2);}}}}
.alert-txt{{flex:1;}}
.alert-title{{font-family:'Orbitron',monospace;font-size:.7rem;font-weight:700;
  color:#ff2244;letter-spacing:.06em;margin-bottom:2px;}}
.alert-sub{{font-family:'Share Tech Mono',monospace;font-size:.58rem;
  color:#ff8899;letter-spacing:.08em;}}
.alert-reduce{{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:900;
  color:#ff2244;text-align:center;margin-left:4px;text-shadow:0 0 12px #ff2244;}}
.spd-module{{display:flex;flex-direction:column;align-items:center;
  padding:14px 12px 10px;position:relative;overflow:hidden;}}
.spd-module::before{{content:'SPEED';position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);font-family:'Orbitron',monospace;
  font-size:4.5rem;font-weight:900;color:rgba(0,212,255,.018);
  letter-spacing:.3em;pointer-events:none;white-space:nowrap;}}
.spd-hdr{{font-family:'Share Tech Mono',monospace;font-size:.55rem;
  letter-spacing:.22em;color:var(--txt3);text-transform:uppercase;margin-bottom:10px;}}
.spd-num{{font-family:'Orbitron',monospace;font-weight:900;font-size:3.6rem;
  line-height:1;text-align:center;transition:color .3s,text-shadow .3s;}}
.spd-kmh{{font-family:'Share Tech Mono',monospace;font-size:.6rem;
  letter-spacing:.24em;color:var(--txt3);margin-top:3px;}}
.spd-badge{{font-family:'Share Tech Mono',monospace;font-size:.56rem;
  letter-spacing:.13em;padding:4px 12px;border-radius:4px;margin-top:9px;
  border:1px solid;transition:all .3s;}}
.vtrack-zone{{display:flex;align-items:center;gap:12px;margin-top:14px;
  width:100%;justify-content:center;}}
.tick-col{{display:flex;flex-direction:column;justify-content:space-between;
  height:188px;font-family:'Share Tech Mono',monospace;font-size:.48rem;
  color:var(--txt3);text-align:right;padding:4px 0;}}
.vtrack{{position:relative;width:34px;height:188px;cursor:ns-resize;
  user-select:none;flex-shrink:0;}}
.vt-bg{{position:absolute;inset:0;border-radius:17px;background:var(--ink3);
  border:1px solid rgba(0,212,255,.18);box-shadow:inset 0 0 18px rgba(0,0,0,.6);}}
.vt-fill{{position:absolute;bottom:0;left:0;right:0;border-radius:17px;
  height:{vt_pct:.1f}%;transition:height .07s ease,background .3s;}}
.vt-glow{{position:absolute;bottom:0;left:15%;right:15%;border-radius:17px;
  height:{vt_pct:.1f}%;filter:blur(7px);opacity:.45;transition:height .07s ease,background .3s;}}
.vt-handle{{position:absolute;left:50%;width:28px;height:28px;
  top:{vt_top:.1f}px;transform:translateX(-50%);border-radius:50%;
  border:2px solid;cursor:grab;z-index:2;transition:box-shadow .15s;}}
.vt-handle:active{{cursor:grabbing;}}
.vt-handle::after{{content:'';position:absolute;top:50%;left:50%;
  width:9px;height:9px;border-radius:50%;
  transform:translate(-50%,-50%);}}
.seg-col{{display:flex;flex-direction:column-reverse;gap:3px;height:188px;padding:4px 0;}}
.sd{{width:7px;border-radius:2px;flex:1;opacity:.1;transition:opacity .07s,box-shadow .07s;}}
.sd-g{{background:#00ffaa;}}.sd-y{{background:#ffaa00;}}.sd-r{{background:#ff2244;}}
.sd.lit{{opacity:1;}}
.sd-g.lit{{box-shadow:0 0 5px #00ffaa;}}
.sd-y.lit{{box-shadow:0 0 5px #ffaa00;}}
.sd-r.lit{{box-shadow:0 0 5px #ff2244;}}
</style>
</head>
<body>

<div class="sec">
  <div class="sd-led" style="background:var(--c1);box-shadow:0 0 6px var(--c1)"></div>
  INPUT SOURCE
</div>
<div class="src-wrap">
  <!-- Buttons click the hidden Streamlit radio widget via postMessage -->
  <div class="src-btn {src_w('webcam')}" onclick="setSource(0)">📷 Webcam Live</div>
  <div class="src-btn {src_w('image')}"  onclick="setSource(1)">🖼️ Image</div>
  <div class="src-btn {src_w('video')}"  onclick="setSource(2)">🎞️ Video</div>
</div>

<div class="sec">
  <div class="sd-led" style="background:var(--c1);box-shadow:0 0 6px var(--c1)"></div>
  CONFIDENCE
</div>
<div class="conf-wrap">
  <div class="conf-lbl">THRESHOLD <span id="conf-val">{conf_pct}%</span></div>
  <input type="range" min="10" max="95" value="{conf_pct}" id="conf-sl"
    oninput="liveConf(this.value)" onchange="commitConf(this.value)">
</div>

<div class="sec">
  <div class="sd-led" style="background:var(--c4);box-shadow:0 0 6px var(--c4)"></div>
  VEHICLE SPEED
</div>
<div class="spd-module">
  <div class="spd-hdr">↕ DRAG TO ADJUST SPEED</div>
  <div class="spd-num" id="spd-num">{speed}</div>
  <div class="spd-kmh">KM / H</div>
  <div class="spd-badge" id="spd-badge"
       style="color:{lb_col};border-color:{lb_col}44;background:{lb_bg}">{lb_txt}</div>
</div>

<div class="spd-alert {'active' if (spd_lim_js > 0 and speed > spd_lim_js) else ''}" id="spd-alert">
  <div class="alert-icon">⚠️</div>
  <div class="alert-txt">
    <div class="alert-title">SLOW DOWN!</div>
    <div class="alert-sub" id="alert-sub">Speed limit: {spd_lim_js} km/h</div>
  </div>
  <div class="alert-reduce" id="alert-reduce">{f'-{speed - spd_lim_js} km/h' if spd_lim_js > 0 and speed > spd_lim_js else ''}</div>
</div>

<div style="display:flex;flex-direction:column;align-items:center;padding:8px 12px 10px;">
  <div class="vtrack-zone">
    <div class="tick-col">
      <div>140</div><div>120</div><div>100</div><div>80</div>
      <div>60</div><div>40</div><div>20</div><div>0</div>
    </div>
    <div class="vtrack" id="vtrack">
      <div class="vt-bg"></div>
      <div class="vt-fill" id="vfill"></div>
      <div class="vt-glow" id="vglow"></div>
      <div class="vt-handle" id="vhandle"></div>
    </div>
    <div class="seg-col" id="segs">{segs_html}</div>
  </div>
</div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// STATE — initialised from Python (server-rendered values are always correct)
// ─────────────────────────────────────────────────────────────────────────────
const TRACK_H  = 188;
const MAX_SPD  = 140;
const SPD_LIM  = {spd_lim_js};   // 0 means no limit sign detected
let   spd      = {speed};
let   dragging = false, startY = 0, startSpd = 0;

// DOM refs
const track       = document.getElementById('vtrack');
const vfill       = document.getElementById('vfill');
const vglow       = document.getElementById('vglow');
const vhandle     = document.getElementById('vhandle');
const spdNum      = document.getElementById('spd-num');
const badge       = document.getElementById('spd-badge');
const segsEl      = document.getElementById('segs');
const alertBox    = document.getElementById('spd-alert');
const alertSub    = document.getElementById('alert-sub');
const alertReduce = document.getElementById('alert-reduce');
const dots        = [...segsEl.querySelectorAll('.sd')];

// ── Helpers ──────────────────────────────────────────────────────────────────
function speedColor(s) {{
  if (SPD_LIM > 0 && s > SPD_LIM) return '#ff2244';
  if (s > 100) return '#ffaa00';
  return '#00d4ff';
}}
function badgeText(s) {{
  if (SPD_LIM <= 0) return 'NO LIMIT SIGN';
  if (s > SPD_LIM)  return '⚠ OVER LIMIT';
  return 'LIMIT ' + SPD_LIM + ' km/h';
}}
function badgeColor(s) {{
  if (SPD_LIM <= 0) return ['#2a4a60','transparent'];
  if (s > SPD_LIM)  return ['#ff2244','rgba(255,34,68,.1)'];
  return ['#00ffaa','rgba(0,255,170,.07)'];
}}

// ── Visual render (runs every drag frame — NO server calls) ──────────────────
function render(s) {{
  const pct  = Math.min(s / MAX_SPD, 1);
  const c    = speedColor(s);
  const [bc, bb] = badgeColor(s);

  vhandle.style.top         = (TRACK_H * (1 - pct) - 14) + 'px';
  vfill.style.height        = (pct * 100) + '%';
  vglow.style.height        = (pct * 100) + '%';
  vfill.style.background    = 'linear-gradient(0deg,' + c + 'cc,' + c + ')';
  vglow.style.background    = c;
  vhandle.style.borderColor = c;
  vhandle.style.background  = 'radial-gradient(circle,' + c + '44,' + c + '11)';
  vhandle.style.boxShadow   = '0 0 16px ' + c + ',0 0 34px ' + c + '44';
  spdNum.textContent        = s;
  spdNum.style.color        = c;
  spdNum.style.textShadow   = '0 0 40px ' + c + ',0 0 80px ' + c + '55';
  badge.textContent         = badgeText(s);
  badge.style.color         = bc;
  badge.style.borderColor   = bc + '44';
  badge.style.background    = bb;

  // Segment bar
  const lit = Math.round(pct * dots.length);
  dots.forEach((d, i) => {{
    const on   = i < lit;
    d.style.opacity   = on ? '1' : '0.1';
    const zone = d.classList.contains('sd-g') ? '#00ffaa'
               : d.classList.contains('sd-y') ? '#ffaa00' : '#ff2244';
    d.style.boxShadow = on ? '0 0 5px ' + zone : 'none';
  }});

  // In-panel alert (immediate, no server needed)
  if (SPD_LIM > 0 && s > SPD_LIM) {{
    const over = s - SPD_LIM;
    alertBox.classList.add('active');
    alertSub.textContent    = 'Speed limit: ' + SPD_LIM + ' km/h';
    alertReduce.textContent = '-' + over + ' km/h';
  }} else {{
    alertBox.classList.remove('active');
    alertReduce.textContent = '';
  }}
}}

render(spd);  // initial paint with server values

// ── DRAG — updates visuals instantly; commits to Streamlit only on mouseup ───
// We find the hidden Streamlit speed slider in the parent window and
// programmatically set its value, then fire an 'input' + 'change' event
// so Streamlit registers the update and reruns.

function commitToStreamlit(sliderId, value) {{
  try {{
    // Walk up to parent window (iframe → Streamlit page)
    const parentDoc = window.parent.document;
    // Streamlit sliders render as <input type="range"> with a data-testid attribute.
    // We target the first range input inside the column that matches our key.
    const sliders = parentDoc.querySelectorAll('input[type="range"]');
    sliders.forEach(sl => {{
      const label = sl.closest('[data-testid="stSlider"]');
      if (!label) return;
      // Match by aria-label or nearby label text
      const labelText = label.querySelector('label') ? label.querySelector('label').textContent : '';
      if (labelText.includes(sliderId) || sl.getAttribute('aria-label') === sliderId) {{
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
          window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(sl, value);
        sl.dispatchEvent(new Event('input',  {{bubbles:true}}));
        sl.dispatchEvent(new Event('change', {{bubbles:true}}));
      }}
    }});
  }} catch(e) {{
    // Fallback: query param (old method) — only if cross-origin blocks us
    const u = new URL(window.parent.location.href);
    u.searchParams.set(sliderId === 'speed_hidden' ? 'speed' : 'conf', value);
    window.parent.location.href = u.toString();
  }}
}}

track.addEventListener('mousedown', e => {{
  dragging = true; startY = e.clientY; startSpd = spd;
  document.body.style.cursor = 'grabbing'; e.preventDefault();
}});
document.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const delta = Math.round(-(e.clientY - startY) / TRACK_H * MAX_SPD / 5) * 5;
  spd = Math.max(0, Math.min(MAX_SPD, startSpd + delta));
  render(spd);  // instant visual feedback — no server involved
}});
document.addEventListener('mouseup', () => {{
  if (!dragging) return;
  dragging = false;
  document.body.style.cursor = '';
  // Commit the final value to Streamlit
  commitToStreamlit('speed_hidden', spd);
}});
track.addEventListener('touchstart', e => {{
  dragging = true; startY = e.touches[0].clientY; startSpd = spd; e.preventDefault();
}}, {{passive:false}});
document.addEventListener('touchmove', e => {{
  if (!dragging) return;
  const delta = Math.round(-(e.touches[0].clientY - startY) / TRACK_H * MAX_SPD / 5) * 5;
  spd = Math.max(0, Math.min(MAX_SPD, startSpd + delta));
  render(spd);
}}, {{passive:false}});
document.addEventListener('touchend', () => {{
  if (!dragging) return; dragging = false;
  commitToStreamlit('speed_hidden', spd);
}});

// ── CONFIDENCE slider ─────────────────────────────────────────────────────────
function liveConf(v) {{
  // Instant visual update only
  document.getElementById('conf-val').textContent = v + '%';
  const sl = document.getElementById('conf-sl');
  sl.style.background =
    'linear-gradient(90deg,#00d4ff ' + v + '%,rgba(0,212,255,.12) ' + v + '%)';
}}
let confTimer = null;
function commitConf(v) {{
  // Debounce — commit 400ms after last change
  clearTimeout(confTimer);
  confTimer = setTimeout(() => commitToStreamlit('conf_hidden', v), 400);
}}

// ── SOURCE buttons — click the hidden Streamlit radio ────────────────────────
function setSource(idx) {{
  try {{
    const parentDoc = window.parent.document;
    // Streamlit radio renders as <input type="radio"> buttons
    const radios = parentDoc.querySelectorAll('input[type="radio"]');
    let count = 0;
    radios.forEach(r => {{
      const wrapper = r.closest('[data-testid="stRadio"]');
      if (!wrapper) return;
      if (count === idx) {{
        r.click();
      }}
      count++;
    }});
  }} catch(e) {{
    // Fallback query param
    const srcNames = ['webcam','image','video'];
    const u = new URL(window.parent.location.href);
    u.searchParams.set('src', srcNames[idx]);
    window.parent.location.href = u.toString();
  }}
}}
</script>
</body>
</html>"""

with left_col:
    components.html(
        build_left_panel(driver_speed, conf_pct, sc, lb_col, lb_bg, lb_txt,
                         spd_lim_js, segs_html, source_mode),
        height=820, scrolling=False
    )


# ─── CENTER PANEL ─────────────────────────────────────────────────────────────
with center_col:

    if source_mode == "image":
        img_file = st.file_uploader(
            "Drop an image or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_up",
        )
        if img_file is not None:
            if img_file.name != st.session_state.img_name:
                nparr    = np.frombuffer(img_file.read(), np.uint8)
                frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                new_dets = run_inference(model, frame, conf_thresh) if model_ok else []
                ann      = draw_boxes(frame.copy(), new_dets, conf_thresh)
                ann_rgb  = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                update_state(new_dets)
                st.session_state.img_annotated = ann_rgb
                st.session_state.img_name      = img_file.name

            # Recompute warn/det HTML with CURRENT speed (not stale)
            dets            = st.session_state.dets
            speed_sign_label = get_speed_limit(dets)
            speed_limit_val  = SPEED_CLASSES.get(speed_sign_label)
            warn_html        = build_warn_html(dets, driver_speed, speed_limit_val)
            det_html         = build_det_html(dets)
            feed_is_live     = st.session_state.img_annotated is not None

    elif source_mode == "video":
        vid_file = st.file_uploader(
            "Drop a video or click to browse",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_up",
        )

    hud_mode_label = {
        "webcam": "◉ LIVE DETECTION",
        "image":  "◉ IMAGE DETECTION",
        "video":  "◉ VIDEO DETECTION",
    }[source_mode]

    st.markdown(f"""
    <div class="vd-center">
      <div class="vd-feed-zone" id="vd-feed-zone">
        <div class="corner-bl"></div>
        <div class="corner-br"></div>
        <div class="scan-line"></div>
        <div class="vd-standby {'hidden' if feed_is_live else ''}" id="vd-standby">
          <div class="ph-icon">{'📷' if source_mode=='webcam' else '🖼️' if source_mode=='image' else '🎞️'}</div>
          <div class="ph-msg">{standby_msg[0]}</div>
          <div class="ph-sub">{standby_msg[1]}</div>
        </div>
        <div class="hud-bottom {'live' if feed_is_live else ''}">
          <div class="hud-chip">{hud_mode_label}</div>
          <div class="hud-chip" style="margin-left:auto">CONF {conf_pct}%</div>
        </div>
    """, unsafe_allow_html=True)

    frame_ph = st.empty()

    if source_mode == "image" and st.session_state.img_annotated is not None:
        frame_ph.image(st.session_state.img_annotated, use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Controls
    if source_mode == "webcam":
        if not st.session_state.cam_running:
            if st.button("📷  START CAMERA", use_container_width=True):
                st.session_state.cam_running = True
                st.rerun()
        else:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            if st.button("⏹  STOP CAMERA", use_container_width=True):
                st.session_state.cam_running = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif source_mode == "image":
        if st.session_state.img_annotated is not None:
            if st.button("🗑️  CLEAR IMAGE", use_container_width=True):
                st.session_state.img_annotated   = None
                st.session_state.img_name        = ""
                st.session_state.dets            = []
                st.session_state.last_sign       = "---"
                st.session_state.last_speed_sign = None
                st.rerun()

    elif source_mode == "video":
        c1v, c2v = st.columns(2)
        run_pressed = c1v.button("▶  RUN DETECTION", use_container_width=True,
                                  disabled=st.session_state.vid_running)
        with c2v:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            stop_pressed = st.button("⏹  STOP", use_container_width=True, key="vstop",
                                      disabled=not st.session_state.vid_running)
            st.markdown('</div>', unsafe_allow_html=True)

        if stop_pressed:
            st.session_state.vid_running = False
            if st.session_state.vid_tmp_path and os.path.exists(st.session_state.vid_tmp_path):
                try: os.unlink(st.session_state.vid_tmp_path)
                except Exception: pass
            st.session_state.vid_tmp_path = ""
            st.rerun()

        if run_pressed and vid_file is not None:
            suffix = Path(vid_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(vid_file.read())
                st.session_state.vid_tmp_path = tmp.name
            st.session_state.vid_running = True
            st.rerun()
        elif run_pressed:
            st.warning("Please upload a video file first.")


# ─── RIGHT PANEL ──────────────────────────────────────────────────────────────
with right_col:
    st.markdown(f"""
    <div class="panel-r">
      <div class="sec">
        <div class="sec-dot" style="background:var(--c5);box-shadow:0 0 6px var(--c5)"></div>
        ACTIVE WARNINGS
      </div>
      <div class="warn-scroll">{warn_html}</div>
      <div class="last-chip">
        <span class="lc-lbl">Last Sign</span>
        <span class="lc-val">{last_sign}</span>
      </div>
      <div class="sec">
        <div class="sec-dot" style="background:var(--c4);box-shadow:0 0 6px var(--c4)"></div>
        DETECTIONS
      </div>
      <div class="det-wrap">{det_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WEBCAM LOOP
# ══════════════════════════════════════════════════════════════════════════════
if source_mode == "webcam" and st.session_state.cam_running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible.")
        st.session_state.cam_running = False
    else:
        cnt = 0
        while st.session_state.cam_running and cnt < 9000:
            ret, frame = cap.read()
            if not ret:
                break
            new_dets = run_inference(model, frame, conf_thresh) if model_ok else []
            ann      = draw_boxes(frame.copy(), new_dets, conf_thresh)
            ann_rgb  = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            update_state(new_dets)
            frame_ph.image(ann_rgb, use_container_width=True)
            cnt += 1
            time.sleep(0.033)
        cap.release()
        st.session_state.cam_running = False


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO LOOP
# ══════════════════════════════════════════════════════════════════════════════
if source_mode == "video" and st.session_state.vid_running:
    tmp_path = st.session_state.vid_tmp_path
    if not tmp_path or not os.path.exists(tmp_path):
        st.session_state.vid_running = False
    else:
        cap          = cv2.VideoCapture(tmp_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        fskip        = max(1, int(fps // 15))
        fidx         = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        prog_ph      = st.empty()
        count_ph     = st.empty()

        while cap.isOpened() and st.session_state.vid_running:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            if fidx % fskip != 0:
                continue
            new_dets = run_inference(model, frame, conf_thresh) if model_ok else []
            ann      = draw_boxes(frame.copy(), new_dets, conf_thresh)
            ann_rgb  = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            update_state(new_dets)
            frame_ph.image(ann_rgb, use_container_width=True)
            prog_ph.progress(min(fidx / total_frames, 1.0))
            count_ph.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.6rem;'
                f'color:#3a6070;letter-spacing:.14em;text-align:center;">'
                f'FRAME {fidx} / {total_frames}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(1 / 15)

        cap.release()
        try: os.unlink(tmp_path)
        except Exception: pass
        prog_ph.empty()
        count_ph.empty()
        st.session_state.vid_running  = False
        st.session_state.vid_tmp_path = ""

        st.rerun()
