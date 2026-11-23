"""The parts of the design the Streamlit widgets cannot draw, as CSS: cards, header, tabs, pills, print."""


# The parts of the design the Streamlit widgets cannot draw: score block,
# pills, chips and the small fact rows. Same tokens as the design page.
PAGE_CSS = """
<style>
/* the design's type scale (14 px base) and its 1320 px page */
html{font-size:14px}
.block-container{max-width:1320px;padding-top:1.2rem}
/* header: brand and data line left, switches right; tabs under them */
.st-key-header{border-bottom:1px solid rgba(128,128,128,.3);padding-bottom:4px;margin-bottom:14px}
.st-key-header [data-testid="stWidgetLabel"] p{font-size:11px;color:#b3b3b3;font-weight:500}
.brand{display:flex;flex-direction:column;gap:2px}
.brand .logo{font-weight:800;font-size:16px;line-height:1.2}
.brand .fresh{font-size:11px;color:#b3b3b3;font-weight:500}
/* the tab row: the tabs take their own width, the controls the rest; each tab its own width */
.st-key-navrow > [data-testid="stLayoutWrapper"]:first-child{flex:0 0 auto !important;width:auto !important}
.st-key-navrow > [data-testid="stLayoutWrapper"]:last-child{flex:1 1 0% !important;width:auto !important;min-width:0}
.st-key-tabs{flex-wrap:nowrap}
.st-key-tabs > [data-testid="stLayoutWrapper"]{flex:0 0 auto !important;width:auto !important}
.st-key-tabs [data-testid="stPageLink-NavLink"]{background:none !important;border:0 !important;
  border-bottom:3px solid transparent !important;border-radius:0 !important;padding:6px 10px !important;
  margin:0 !important;color:#d9d9d9;white-space:nowrap}
.st-key-tabs [data-testid="stPageLink-NavLink"] span,.st-key-tabs [data-testid="stPageLink-NavLink"] p{
  font-size:14px;font-weight:600;color:inherit}
.st-key-tabs [data-testid="stPageLink-NavLink"][aria-current="page"],
.st-key-tabs [data-testid="stPageLink-NavLink"][aria-disabled="true"]{color:inherit;border-bottom-color:#8ac646 !important;opacity:1}
/* page title and the cards */
.ptitle{font-size:22px;font-weight:800;margin:0 0 4px;line-height:1.2}
.psub{font-size:12px;color:#b3b3b3;margin:0 0 14px}
[class*="st-key-card-"]{background:rgba(0,0,0,.2);border:1px solid rgba(128,128,128,.3);border-radius:8px;
  padding:14px 16px;margin-bottom:12px}
.ch{margin:0 0 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#b3b3b3;font-weight:700;
  display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.ch .r{text-transform:none;letter-spacing:0;font-weight:500;font-size:12px}
.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#b3b3b3;font-weight:700}
.camname{font-size:34px;font-weight:800;line-height:1.1;margin:2px 0 6px}
.facts{font-size:12px;color:#b3b3b3}
.hero{font-size:56px;font-weight:800;line-height:1;display:inline-block}
.hero .unit{font-size:16px;color:#b3b3b3;margin-left:6px;font-weight:600}
.status{display:inline-block;margin-left:14px;padding:3px 10px;border-radius:4px;font-size:12px;
  font-weight:800;letter-spacing:.04em;text-transform:uppercase;border:1px solid #9a9a9a;color:#b3b3b3}
.status.good{color:#8ac646;border-color:#8ac646}.status.warn{color:#f2b134;border-color:#f2b134}
.status.crit{color:#ef6b64;border-color:#ef6b64}
.meter{height:8px;border-radius:4px;background:rgba(128,128,128,.25);margin:10px 0 8px;overflow:hidden}
.meter .fill{display:block;height:100%;background:#8ac646}
.meter.warn .fill{background:#f2b134}.meter.crit .fill{background:#ef6b64}
.score-sub{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#b3b3b3;margin-bottom:10px}
.score-sub .up{color:#8ac646}.score-sub .down{color:#ef6b64}
.items .row{display:grid;grid-template-columns:1fr 160px 90px;gap:10px;align-items:center;
  font-size:13px;padding:5px 0;border-top:1px solid rgba(128,128,128,.3)}
.items .track{height:6px;border-radius:3px;background:rgba(128,128,128,.25);overflow:hidden}
.items .track i{display:block;height:100%;background:#ef6b64}
.items .n{text-align:right;font-weight:700}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:700;border:1px solid}
.pill.ok{color:#8ac646;border-color:#8ac646}.pill.viol{color:#f28b8b;border-color:#ef6b64}
.pill.unk{color:#b3b3b3;border-color:#9a9a9a}
.chip{display:inline-block;font-size:11px;padding:1px 6px;border:1px solid #ef6b64;border-radius:3px;
  margin:2px 4px 0 0;color:#f28b8b}
.chip.o{border-color:#8ac646;color:#8ac646}.chip.u{border-color:#9a9a9a;color:#b3b3b3}
.meta{display:flex;gap:14px 28px;flex-wrap:wrap;font-size:12px;color:#b3b3b3;margin-bottom:8px}
.meta b{display:block;font-size:15px;color:inherit;filter:brightness(1.35)}
.ramp{display:inline-block;width:60px;height:8px;border-radius:4px;vertical-align:middle;
  background:linear-gradient(90deg,rgba(232,83,79,0),#e8534f)}
/* the strip: one bar per image, red = a violation, green = all compliant, grey = empty */
.strip{display:flex;gap:3px;overflow-x:auto;padding:4px 0 2px}
.strip i{flex:0 0 12px;height:30px;border-radius:2px;background:rgba(128,128,128,.35)}
.strip .g{background:#8ac646}.strip .r{background:#ef6b64}
.strip .sel{outline:2px solid currentColor;outline-offset:1px;border-radius:4px}
/* print: the browser's "Save as PDF" stands in for the report service until it exists
   (see request_report). The controls go, a title line comes, the palette is light. */
.print-only{display:none}
@media print{
  .st-key-header,[data-testid="stHeader"],
  .st-key-toolbar,.st-key-see_images,.stButton,[data-testid="stSlider"],.strip,[data-testid="stToast"],
  [data-testid="stTextInput"]{display:none !important}
  [class*="st-key-card-"]{background:#fff;border-color:#c9ccd3;break-inside:avoid}
  .print-only{display:block;font-weight:700;font-size:14px;margin:0 0 12px}
  [data-testid="stMain"]{left:0 !important;width:100% !important}
  .stApp,[data-testid="stMain"]{background:#fff !important}
  .stApp,.stApp p,.stApp span,.stApp div,.stApp b,.stApp h1,.stApp h2,.stApp h3,.stApp td,.stApp th{color:#141920 !important}
}
/* the toolbar and the "see images" buttons sit in one row: compact type */
.st-key-toolbar [data-testid="stButtonGroup"] button{padding:2px 7px}
.st-key-toolbar .stButton > button p{font-size:13px;white-space:nowrap}
.st-key-see_images .stButton > button{padding:1px 10px;min-height:0}
.st-key-see_images .stButton > button p{font-size:12px;white-space:nowrap}
.st-key-report_tools .stButton > button{padding:2px 8px;min-height:0}
.st-key-report_tools .stButton > button p{font-size:11px;white-space:nowrap}
/* the small ＋ － Fit and ◀ ▶ buttons of the design */
.st-key-zoom .stButton > button,.st-key-stepper .stButton > button{padding:2px 10px;min-height:0;line-height:1.5}
.st-key-zoom .stButton > button p,.st-key-stepper .stButton > button p{font-size:12px}
</style>
"""


# The light palette's surfaces (the dark ones are translucent blacks).
LIGHT_PAGE_CSS = """
<style>
[class*="st-key-card-"]{background:#fbfbfc;border-color:rgba(20,25,32,.11)}
.st-key-header{border-bottom-color:rgba(20,25,32,.11)}
.st-key-tabs [data-testid="stPageLink-NavLink"]{color:#333a46}
</style>
"""
