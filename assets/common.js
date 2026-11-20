// llmlearn 公共小工具：所有课程页面共用（无任何外部依赖）
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches;
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// 可复现的随机数（mulberry32）与标准正态
function rng(seed = 42) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  u.normal = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  return u;
}
const randMat = (r, c, R, scale = 1) => Array.from({ length: r }, () => Array.from({ length: c }, () => R.normal() * scale));

// 线性代数
const matmul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const T = A => A[0].map((_, j) => A.map(r => r[j]));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function softmax(v, temp = 1) {
  const m = Math.max(...v.filter(Number.isFinite));
  const e = v.map(x => Number.isFinite(x) ? Math.exp((x - m) / temp) : 0);
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

// 数值 → 颜色。diverging：负蓝正橙；否则 0→透明 1→橙
function heatColor(v, max = 1, diverging = true) {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  if (!diverging || v >= 0) return `rgba(194,65,12,${a.toFixed(3)})`;
  return `rgba(29,78,216,${a.toFixed(3)})`;
}

// 渲染热力图表格
function heatmap(el, M, { rows = [], cols = [], fmt = v => v.toFixed(2), diverging = true, max, hlRow = -1, showVal = true, onRow } = {}) {
  const mx = max ?? Math.max(...M.flat().filter(Number.isFinite).map(Math.abs), 1e-9);
  let h = cols.length ? `<tr><th></th>${cols.map(c => `<th>${c}</th>`).join('')}</tr>` : '';
  M.forEach((r, i) => {
    h += `<tr class="${i === hlRow ? 'hl' : ''}" data-i="${i}" style="${onRow ? 'cursor:pointer' : ''}">${rows.length ? `<th class="r">${rows[i]}</th>` : ''}` +
      r.map(v => {
        if (!Number.isFinite(v)) return `<td style="color:var(--muted)">−∞</td>`;
        const a = Math.abs(v) / mx;
        return `<td title="${fmt(v)}" style="background:${heatColor(v, mx, diverging)};color:${a > .55 ? '#fff' : 'inherit'}">${showVal ? fmt(v) : ''}</td>`;
      }).join('') + '</tr>';
  });
  el.innerHTML = `<table class="hm">${h}</table>`;
  if (onRow) el.querySelectorAll('tr[data-i]').forEach(tr => tr.onclick = () => onRow(+tr.dataset.i));
}

// 横向条形图（概率分布等）
function bars(el, labels, values, { fmt = v => (v * 100).toFixed(1) + '%', max, dim = [] } = {}) {
  const mx = max ?? Math.max(...values, 1e-9);
  el.innerHTML = labels.map((l, i) => `<div class="bar ${dim[i] ? 'dim' : ''}"><span>${l}</span><div class="track"><div class="fill" style="width:${Math.max(0, values[i]) / mx * 100}%"></div></div><span>${fmt(values[i])}</span></div>`).join('');
}

// 简易折线图（SVG）
function lineChart(el, series, { w = 460, h = 220, ymin, ymax, xlabel = '', ylabel = '', colors } = {}) {
  const all = series.flatMap(s => s.data);
  const lo = ymin ?? Math.min(...all), hi = ymax ?? Math.max(...all);
  const n = Math.max(...series.map(s => s.data.length), 2);
  const px = i => 40 + i / (n - 1) * (w - 55), py = v => h - 28 - (v - lo) / ((hi - lo) || 1) * (h - 45);
  const cs = colors || [cssVar('--accent'), cssVar('--blue'), cssVar('--green')];
  let s = `<line x1="40" y1="${h - 28}" x2="${w - 10}" y2="${h - 28}" stroke="${cssVar('--line')}"/><line x1="40" y1="10" x2="40" y2="${h - 28}" stroke="${cssVar('--line')}"/>`;
  s += `<text x="34" y="18" font-size="10" text-anchor="end" fill="${cssVar('--muted')}">${hi.toFixed(2)}</text><text x="34" y="${h - 28}" font-size="10" text-anchor="end" fill="${cssVar('--muted')}">${lo.toFixed(2)}</text>`;
  s += `<text x="${w - 10}" y="${h - 8}" font-size="11" text-anchor="end" fill="${cssVar('--muted')}">${xlabel}</text><text x="44" y="12" font-size="11" fill="${cssVar('--muted')}">${ylabel}</text>`;
  series.forEach((se, k) => {
    if (se.data.length < 2) return;
    s += `<polyline fill="none" stroke="${cs[k % cs.length]}" stroke-width="2" points="${se.data.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')}"/>`;
    if (se.name) s += `<text x="${w - 12}" y="${22 + k * 14}" font-size="11" text-anchor="end" fill="${cs[k % cs.length]}">${se.name}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" style="width:100%;max-width:${w}px">${s}</svg>`;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
