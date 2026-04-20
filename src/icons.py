"""
PivotOS Icon System — Lucide Icons
====================================
LinkedIn-grade SVG icon library for Streamlit.

All icons from Lucide (https://lucide.dev) — MIT licensed.
ViewBox: 0 0 24 24 | Stroke: 2px | Linecap: round | Linejoin: round
Fill: none | Stroke: currentColor

Usage:
    from src.icons import icon, icon_badge

    st.markdown(f'{icon("check")} Saved', unsafe_allow_html=True)
    st.markdown(f'<div>{icon("target", 20, "#0A66C2")} High ROI</div>', unsafe_allow_html=True)

CSS variables injected via get_icon_css() — call once at app start.
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# LinkedIn-aligned CSS variables
# ─────────────────────────────────────────────────────────────────────────────

ICON_CSS = """
<style>
:root {
  --icon-primary:   #0A66C2;
  --icon-success:   #057642;
  --icon-warning:   #A05A00;
  --icon-danger:    #B71C1C;
  --icon-muted:     rgba(0,0,0,0.45);
  --icon-subtle:    rgba(0,0,0,0.28);
  --icon-inverse:   rgba(255,255,255,0.85);
}
.li-icon {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  vertical-align:middle;
  flex-shrink:0;
}
.li-icon-box {
  width:36px;height:36px;
  border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SVG paths — Lucide 0.363
# ─────────────────────────────────────────────────────────────────────────────

_PATHS: dict[str, str] = {
    # Status
    "check":
        '<polyline points="20 6 9 17 4 12"/>',
    "x":
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/>',
    "check-circle":
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>',
    "alert-triangle":
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>',
    "alert-circle":
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="12"/>'
        '<line x1="12" y1="16" x2="12.01" y2="16"/>',
    "info":
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>',
    "help-circle":
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>',

    # Navigation / Actions
    "arrow-right":
        '<line x1="5" y1="12" x2="19" y2="12"/>'
        '<polyline points="12 5 19 12 12 19"/>',
    "arrow-up":
        '<line x1="12" y1="19" x2="12" y2="5"/>'
        '<polyline points="5 12 12 5 19 12"/>',
    "chevron-right":
        '<polyline points="9 18 15 12 9 6"/>',
    "chevron-down":
        '<polyline points="6 9 12 15 18 9"/>',
    "external-link":
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
        '<polyline points="15 3 21 3 21 9"/>'
        '<line x1="10" y1="14" x2="21" y2="3"/>',
    "refresh-cw":
        '<polyline points="23 4 23 10 17 10"/>'
        '<polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "download":
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>',
    "upload":
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" y1="3" x2="12" y2="15"/>',
    "plus":
        '<line x1="12" y1="5" x2="12" y2="19"/>'
        '<line x1="5" y1="12" x2="19" y2="12"/>',
    "trash-2":
        '<polyline points="3 6 5 6 21 6"/>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" y1="11" x2="10" y2="17"/>'
        '<line x1="14" y1="11" x2="14" y2="17"/>',
    "copy":
        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>'
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "save":
        '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
        '<polyline points="17 21 17 13 7 13 7 21"/>'
        '<polyline points="7 3 7 8 15 8"/>',
    "settings":
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',

    # Career / Jobs
    "briefcase":
        '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>'
        '<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "target":
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>',
    "compass":
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "map":
        '<polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>'
        '<line x1="9" y1="3" x2="9" y2="18"/>'
        '<line x1="15" y1="6" x2="15" y2="21"/>',
    "rocket":
        '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/>'
        '<path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
        '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/>'
        '<path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>',
    "trending-up":
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/>',
    "award":
        '<circle cx="12" cy="8" r="6"/>'
        '<path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
    "trophy":
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>'
        '<path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
        '<path d="M4 22h16"/>'
        '<path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
        '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
        '<path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>',

    # People
    "user":
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>'
        '<circle cx="12" cy="7" r="4"/>',
    "users":
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
        '<circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
        '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>',

    # Documents
    "file-text":
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/>'
        '<line x1="16" y1="17" x2="8" y2="17"/>'
        '<polyline points="10 9 9 9 8 9"/>',
    "clipboard":
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>',
    "book-open":
        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
        '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',

    # Data / Analysis
    "bar-chart-2":
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>',
    "pie-chart":
        '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>'
        '<path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "activity":
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "layers":
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>',
    "git-branch":
        '<line x1="6" y1="3" x2="6" y2="15"/>'
        '<circle cx="18" cy="6" r="3"/>'
        '<circle cx="6" cy="18" r="3"/>'
        '<path d="M18 9a9 9 0 0 1-9 9"/>',

    # AI / Tech
    "cpu":
        '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<line x1="9" y1="1" x2="9" y2="4"/>'
        '<line x1="15" y1="1" x2="15" y2="4"/>'
        '<line x1="9" y1="20" x2="9" y2="23"/>'
        '<line x1="15" y1="20" x2="15" y2="23"/>'
        '<line x1="20" y1="9" x2="23" y2="9"/>'
        '<line x1="20" y1="14" x2="23" y2="14"/>'
        '<line x1="1" y1="9" x2="4" y2="9"/>'
        '<line x1="1" y1="14" x2="4" y2="14"/>',
    "brain":
        '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>'
        '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>',
    "zap":
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "search":
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "microscope":
        '<path d="M6 18h8"/>'
        '<path d="M3 22h18"/>'
        '<path d="M14 22a7 7 0 1 0 0-14h-1"/>'
        '<path d="M9 14h2"/>'
        '<path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>'
        '<path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
    "flask-conical":
        '<path d="M14 2v6l3 3"/>'
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
        '<path d="M6.5 17L4.5 3h15L17.5 17"/>'
        '<path d="M15 8l-3.5 6-3.5-6"/>',
    "eye":
        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
        '<circle cx="12" cy="12" r="3"/>',
    "telescope":
        '<path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44"/>'
        '<path d="m13.56 11.747 4.332-.924"/>'
        '<path d="m16 21-3.105-6.21"/>'
        '<path d="M16.485 5.94a2 2 0 0 1 1.455-2.425l1.09-.272a1 1 0 0 1 1.212.727l1.515 6.06a1 1 0 0 1-.727 1.213l-1.09.272a2 2 0 0 1-2.425-1.455z"/>'
        '<path d="m6.158 8.633 1.114 4.456"/>'
        '<path d="m8 21 3.105-6.21"/>'
        '<circle cx="12" cy="21" r="1"/>',

    # Communication
    "mic":
        '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>'
        '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/>'
        '<line x1="12" y1="19" x2="12" y2="23"/>'
        '<line x1="8" y1="23" x2="16" y2="23"/>',
    "message-square":
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "mail":
        '<path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>'
        '<polyline points="22,6 12,13 2,6"/>',
    "link":
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "linkedin":
        '<path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/>'
        '<rect x="2" y="9" width="4" height="12"/>'
        '<circle cx="4" cy="4" r="2"/>',

    # Finance
    "dollar-sign":
        '<line x1="12" y1="1" x2="12" y2="23"/>'
        '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "scale":
        '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/>'
        '<path d="M7 21h10"/>'
        '<path d="M12 3v18"/>'
        '<path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',

    # Buildings / Company
    "building-2":
        '<path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/>'
        '<path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/>'
        '<path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/>'
        '<path d="M10 6h4"/><path d="M10 10h4"/>'
        '<path d="M10 14h4"/><path d="M10 18h4"/>',
    "globe":
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',

    # Time / Progress
    "clock":
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>',
    "calendar":
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
        '<line x1="16" y1="2" x2="16" y2="6"/>'
        '<line x1="8" y1="2" x2="8" y2="6"/>'
        '<line x1="3" y1="10" x2="21" y2="10"/>',
    "timer":
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="6" x2="12" y2="12"/>'
        '<path d="m9 3 6 0"/>',

    # Misc
    "shield":
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "flame":
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 3z"/>',
    "star":
        '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "wrench":
        '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "ruler":
        '<path d="M21.3 8.7 8.7 21.3c-1 1-2.5 1-3.4 0l-2.6-2.6c-1-1-1-2.5 0-3.4L15.3 2.7c1-1 2.5-1 3.4 0l2.6 2.6c1 1 1 2.5 0 3.4Z"/>'
        '<path d="m7.5 10.5 2 2"/><path d="m10.5 7.5 2 2"/><path d="m13.5 4.5 2 2"/>',
    "dna":
        '<path d="M2 15c6.667-6 13.333 0 20-6"/>'
        '<path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/>'
        '<path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/>'
        '<path d="m17 6-2.5-2.5"/><path d="m14 8.5-1-1"/>'
        '<path d="m7 18 2.5 2.5"/><path d="m3.5 14.5.5.5"/>',
    "sparkles":
        '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>'
        '<path d="M5 3v4"/><path d="M19 17v4"/>'
        '<path d="M3 5h4"/><path d="M17 19h4"/>',
    "circle-dot":
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="1"/>',
}

# Aliases
_PATHS["bot"] = _PATHS["cpu"]
_PATHS["network"] = _PATHS["users"]
_PATHS["analyse"] = _PATHS["bar-chart-2"]
_PATHS["interview"] = _PATHS["mic"]
_PATHS["pivot"] = _PATHS["compass"]
_PATHS["mock"] = _PATHS["mic"]
_PATHS["skills"] = _PATHS["brain"]
_PATHS["salary"] = _PATHS["dollar-sign"]
_PATHS["validate"] = _PATHS["shield"]
_PATHS["architecture"] = _PATHS["layers"]
_PATHS["cv"] = _PATHS["file-text"]
_PATHS["roi"] = _PATHS["target"]
_PATHS["ai"] = _PATHS["cpu"]
_PATHS["zwilling"] = _PATHS["user"]
_PATHS["ic"] = _PATHS["mic"]


# ─────────────────────────────────────────────────────────────────────────────
# Core render function
# ─────────────────────────────────────────────────────────────────────────────

def icon(
    name: str,
    size: int = 16,
    color: str = "currentColor",
    stroke_width: float = 2.0,
    css_class: str = "",
    style: str = "",
) -> str:
    """
    Return an inline SVG Lucide icon as an HTML string.

    Args:
        name:         Lucide icon name (e.g. "check", "briefcase", "target")
        size:         px size for width and height (default 16)
        color:        stroke color — CSS variable or hex (default currentColor)
        stroke_width: stroke-width (default 2.0, LinkedIn uses 1.5-2.0)
        css_class:    additional CSS class(es)
        style:        inline style string

    Returns:
        HTML string: <svg ...>...</svg>
    """
    paths = _PATHS.get(name, _PATHS.get("circle-dot", ""))
    cls = f"li-icon {css_class}".strip()
    base_style = f"vertical-align:middle;flex-shrink:0;{style}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'class="{cls}" style="{base_style}">'
        f'{paths}'
        f'</svg>'
    )


def icon_box(
    name: str,
    bg: str = "#EEF3FB",
    color: str = "#0A66C2",
    size: int = 18,
    box_size: int = 36,
    radius: int = 8,
) -> str:
    """
    Render an icon inside a LinkedIn-style colored square box.
    Used for section headers (replaces li-tool-icon).
    """
    svg = icon(name, size=size, color=color)
    return (
        f'<div style="width:{box_size}px;height:{box_size}px;border-radius:{radius}px;'
        f'background:{bg};display:flex;align-items:center;justify-content:center;'
        f'flex-shrink:0">{svg}</div>'
    )


def status_icon(status: str, size: int = 14) -> str:
    """
    Return a semantic status icon.
    status: "success" | "warning" | "error" | "info" | "neutral"
    """
    _map = {
        "success": ("check-circle", "#057642"),
        "warning": ("alert-triangle", "#A05A00"),
        "error":   ("alert-circle",  "#B71C1C"),
        "info":    ("info",           "#0A66C2"),
        "neutral": ("circle-dot",     "#5F6B7A"),
    }
    icon_name, color = _map.get(status, ("circle-dot", "#5F6B7A"))
    return icon(icon_name, size=size, color=color)


def check_icon(size: int = 14) -> str:
    return icon("check", size, "#057642")

def x_icon(size: int = 14) -> str:
    return icon("x", size, "#B71C1C")

def warn_icon(size: int = 14) -> str:
    return icon("alert-triangle", size, "#A05A00")


def get_icon_css() -> str:
    """Return the CSS block to inject once at app startup."""
    return ICON_CSS


# ─────────────────────────────────────────────────────────────────────────────
# Icon box configs for li-tool-header sections
# ─────────────────────────────────────────────────────────────────────────────

SECTION_ICONS: dict[str, tuple[str, str, str]] = {
    # key: (icon_name, bg_color, icon_color)
    "skills":      ("brain",        "#EEF3FB", "#0A66C2"),
    "salary":      ("dollar-sign",  "#FFF8E7", "#A05A00"),
    "validate":    ("shield",       "#FEF0F0", "#B24020"),
    "architecture":("layers",       "#EEF3FB", "#0A66C2"),
    "cv":          ("file-text",    "#F3EEF9", "#7A2A8A"),
    "linkedin":    ("linkedin",     "#EEF3FB", "#0A66C2"),
    "roi":         ("target",       "#EEF3FB", "#0A66C2"),
    "ai":          ("cpu",          "#EEF3FB", "#0A66C2"),
    "mock":        ("mic",          "#F3EEF9", "#7A2A8A"),
    "zwilling":    ("user",         "#F0F4FF", "#0A66C2"),
    "interview":   ("mic",          "#EEF3FB", "#0A66C2"),
    "search":      ("search",       "#EEF3FB", "#0A66C2"),
    "market":      ("trending-up",  "#F0FFF4", "#057642"),
    "network":     ("users",        "#EEF3FB", "#0A66C2"),
    "pipeline":    ("bar-chart-2",  "#EEF3FB", "#0A66C2"),
    "dna":         ("dna",          "#F3EEF9", "#7A2A8A"),
    "cohort":      ("users",        "#EEF3FB", "#0A66C2"),
    "momentum":    ("flame",        "#FFF8E7", "#A05A00"),
    "brief":       ("compass",      "#EEF3FB", "#0A66C2"),
    "hm":          ("user",         "#EEF3FB", "#0A66C2"),
}


def section_icon_box(key: str) -> str:
    """Render a section-header icon box by semantic key."""
    name, bg, color = SECTION_ICONS.get(key, ("circle-dot", "#EEF3FB", "#0A66C2"))
    return icon_box(name, bg=bg, color=color)
