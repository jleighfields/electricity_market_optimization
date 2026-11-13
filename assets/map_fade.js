// Plant markers faint when zoomed out, solid when zoomed in (anywidget ESM, loaded by _MapFade
// in lp_electricity_markets.py). Plotly can't do this, so we give the MapLibre circle layers of
// traces tagged meta="fade" a zoom expression for their opacity.
const FAINT = 0.4, ZOOM_OUT = 3.5, ZOOM_IN = 6.5;  // fraction of full opacity at ZOOM_OUT
function fade(gd) {
  const map = gd._fullLayout?.map?._subplot?.map;
  if (!map || !map.getStyle()) return;
  const uids = new Set((gd._fullData || []).filter(t => t.meta === "fade").map(t => t.uid));
  for (const layer of map.getStyle().layers) {
    const uid = layer.id.match(/^plotly-trace-layer-(.+)-circle$/)?.[1];
    if (!uid || !uids.has(uid)) continue;
    const full = map.getPaintProperty(layer.id, "circle-opacity");
    if (typeof full !== "number") continue;  // already an expression
    map.setPaintProperty(layer.id, "circle-opacity",
      ["interpolate", ["linear"], ["zoom"], ZOOM_OUT, full * FAINT, ZOOM_IN, full]);
  }
}
function scan() {
  // marimo draws each chart inside the shadow DOM of a <marimo-plotly> element
  for (const host of document.querySelectorAll("marimo-plotly")) {
    for (const gd of host.shadowRoot?.querySelectorAll(".js-plotly-plot") ?? []) {
      fade(gd);  // cheap when already applied; also catches Plotly redraws (legend clicks)
    }
  }
}
function render() {
  if (window.__mapFade) return;
  window.__mapFade = setInterval(scan, 500);
}
export default { render };
