"""The parts of the design the Streamlit widgets cannot draw, as CSS: header, bands, tiles, tables, meters."""


# The parts of the design the Streamlit widgets cannot draw: the
# attention strip, the tiles, pills, tables and meters. Same tokens as
# the design page.
PAGE_CSS = """
<style>
/* the design's type scale (14 px base) and its 1320 px page */
html{font-size:14px}
.block-container{max-width:1320px;padding-top:1.4rem}
/* top bar: title with the green separator, the data line under it */
.st-key-header{border-bottom:1px solid rgba(128,128,128,.3);padding-bottom:4px;margin-bottom:12px}
.brand{display:flex;flex-direction:column;gap:2px;padding:4px 0 2px}
.brand .logo{font-size:18px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;line-height:1.2}
.brand .logo .sep{color:#8ac646;margin:0 6px}
.brand .sub{font-size:12px;color:#b3b3b3}
.badge{display:inline-block;padding:2px 9px;border-radius:4px;font-size:12px;font-weight:700;
  border:1px solid #f2b134;color:#f2b134;background:rgba(242,177,52,.16);margin-left:8px}
/* section head: small uppercase title left, its caption right */
.sh{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;
  margin:18px 0 10px;padding:9px 14px;background:rgba(0,0,0,.18);border-radius:6px}
.sh h2{margin:0;font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:inherit}
.sh span{font-size:12px;color:#b3b3b3}
.h3{font-size:14px;font-weight:700;margin:0 0 4px}
.cap{font-size:12px;color:#b3b3b3;margin:0 0 8px}
/* cards: containers keyed card-* */
[class*="st-key-card-"]{background:rgba(0,0,0,.18);border:1px solid rgba(128,128,128,.3);border-radius:6px;
  padding:14px 16px}
.attention{display:flex;flex-wrap:wrap;gap:8px 28px;padding:12px 16px;border:1px solid rgba(128,128,128,.3);
  border-radius:6px;background:rgba(0,0,0,.18);font-size:13px;margin-bottom:12px}
.attention span{display:inline-flex;gap:8px;align-items:center}
.sdot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#8ac646;flex:none}
.sdot.warn{background:#f2b134}.sdot.crit{background:#ef6b64}.sdot.info{background:#b3b3b3}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:8px}
@media(max-width:1100px){.tiles{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.tiles{grid-template-columns:1fr}}
.tile{border:1px solid rgba(128,128,128,.3);border-radius:6px;padding:14px 16px;background:rgba(0,0,0,.18);
  display:flex;flex-direction:column;gap:6px;min-width:0}
.tile .label{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#b3b3b3}
.tile .value{font-size:30px;font-weight:700;line-height:1.1}
.tile .value small{font-size:14px;font-weight:600;color:#b3b3b3;margin-left:4px}
.tile .sub{font-size:12px;color:#b3b3b3;display:flex;gap:8px;align-items:center}
.table-wrap{overflow-x:auto;margin-bottom:6px}
.htable{width:100%;border-collapse:collapse;font-size:13px}
.htable th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#b3b3b3;
  padding:8px 10px;border-bottom:1px solid rgba(128,128,128,.35);white-space:nowrap}
.htable td{padding:9px 10px;border-bottom:1px solid rgba(128,128,128,.2);vertical-align:top}
.htable .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.htable .muted{color:#b3b3b3}.htable .id{display:block;font-size:11px;color:#b3b3b3}
.htable .time{white-space:nowrap}
.htable .msg{max-width:560px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{display:inline-block;padding:2px 9px;border-radius:10px;font-size:12px;font-weight:700;border:1px solid;
  white-space:nowrap}
.pill.ok{color:#8ac646;border-color:#8ac646}.pill.warn{color:#f2b134;border-color:#f2b134}
.pill.crit{color:#ef6b64;border-color:#ef6b64}.pill.info{color:#b3b3b3;border-color:#b3b3b3}
.meter{display:grid;grid-template-columns:72px 1fr 48px;gap:10px;align-items:center;font-size:13px;margin:10px 0}
.meter .track{position:relative;height:10px;border-radius:5px;background:rgba(128,128,128,.25)}
.meter .fill{display:block;height:100%;border-radius:5px;background:#8ac646}
.meter .fill.warn{background:#f2b134}.meter .fill.crit{background:#ef6b64}
.meter .mark{position:absolute;top:-3px;width:2px;height:16px;background:#f2b134}
.meter .mark.crit{background:#ef6b64}
.meter .val{text-align:right;font-weight:700}
</style>
"""
