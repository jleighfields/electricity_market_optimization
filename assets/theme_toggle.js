// Light/dark button on the title slide of each deck (anywidget ESM, loaded by _ThemeToggle).
function render({ el }) {
  const btn = document.createElement("button");
  btn.textContent = "◐ Light / dark";
  btn.style.cssText = "font-size: 14px; padding: 4px 12px; border-radius: 6px; " +
    "border: 1px solid #A9B4C8; background: transparent; color: inherit; cursor: pointer";
  btn.onclick = () => {
    const dark = !document.body.classList.contains("dark");
    for (const node of [document.documentElement, document.body]) {
      node.classList.toggle("dark", dark); node.classList.toggle("dark-theme", dark);
      node.classList.toggle("light", !dark); node.classList.toggle("light-theme", !dark);
      node.style.colorScheme = dark ? "dark" : "light";
    }
  };
  el.appendChild(btn);
}
export default { render };
