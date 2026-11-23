import { P, parseSubnet, quantizeInt4, packInt4, packTernary, ternarize, FlyWasm } from './flyq.js';

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || undefined;
const fmt = (x) => x >= 1e6 ? (x / 1e6).toFixed(2) + 'M' : x >= 1e3 ? (x / 1e3).toFixed(1) + 'k' : String(x);
const C = { acc: '#1fe5a5', hot: '#ff2e88', inh: '#ff3d4f', grid: '#1a2d27', mut: '#83998f', warn: '#f2c14e' };

const state = { theta: 3, gain: 'drive', playing: true, speed: 50, tick: 0 };
let sub, q4, module, A, B, ternary, polarity;
const rate = { a: [], b: [] };           // spikes per 5 ms bins
const acc = { a: { spikes: 0, ops: 0, ms: 0, ticks: 0 }, b: { spikes: 0, ops: 0, ms: 0, ticks: 0 } };

async function boot() {
  const [wasm, bin, sw] = await Promise.all([
    fetch('flyq.wasm').then((r) => r.arrayBuffer()), fetch('assets/subnet.bin').then((r) => r.arrayBuffer()),
    fetch('assets/sweep.json').then((r) => r.json())]);
  module = await WebAssembly.compile(wasm);
  sub = parseSubnet(bin); q4 = quantizeInt4(sub.counts);
  polarity = new Int8Array(sub.n);
  for (let u = 0; u < sub.n; u++) { let s = 0; for (let e = sub.indptr[u]; e < sub.indptr[u + 1]; e++) s += Math.sign(sub.counts[e]); polarity[u] = s >= 0 ? 1 : -1; }
  buildSweep(sw); buildEsp();
  await rebuild();
  wire(); requestAnimationFrame(loop);
  // ?warm=N pre-runs N frames (README screenshots); ?t=1&g=unit starts from a config
  const q = new URLSearchParams(location.search);
  if (q.has('t')) setCfgNow(+q.get('t'), q.get('g') || 'unit');
  if (q.has('warm')) { await rebuild(); for (let i = 0; i < +q.get('warm'); i++) frame(); }
}

const inst = async () => new FlyWasm((await WebAssembly.instantiate(module, {})).exports);

async function rebuild() {
  // fresh instances: the wasm bump allocator never frees, so a new instance is the cheap "free"
  const fa = await inst(), fb = await inst();
  A = { fw: fa, h: fa.makeNet(sub, 'int4', { indptr: sub.indptr, indices: sub.indices, packed: packInt4(q4) }) };
  ternary = ternarize(sub, q4, state.theta, state.gain);
  B = { fw: fb, h: fb.makeNet(sub, 'ternary', { indptr: ternary.indptr, indices: ternary.indices, packed: packTernary(ternary.codes), lut: ternary.lut }) };
  state.tick = 0; rate.a = []; rate.b = [];
  for (const k of ['a', 'b']) Object.assign(acc[k], { spikes: 0, ops: 0, ms: 0, ticks: 0 });
  for (const id of ['r4', 'r2']) { const c = $(id); c.width = c.clientWidth * devicePixelRatio; c.height = 260 * devicePixelRatio; c.getContext('2d').clearRect(0, 0, c.width, c.height); }
  drawQuant(); renderStats();
}

function step(net, a, ticks) {
  const t0 = performance.now(); const r = net.fw.run(net.h, ticks); a.ms += performance.now() - t0;
  a.spikes += r.counts.reduce((x, y) => x + y, 0); a.ops += r.edgeOps; a.ticks += ticks; return r;
}

function scroll(canvas, res, ticks) {
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height, tpp = 4 / 1;   // 4 ticks per device-px column
  const dx = Math.max(1, Math.round(ticks / tpp * devicePixelRatio / devicePixelRatio));
  ctx.globalCompositeOperation = 'copy'; ctx.drawImage(canvas, -dx, 0); ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = css('--jade-code') || '#060c0a'; ctx.fillRect(W - dx, 0, dx, H);
  const row = H / sub.n;
  for (const ev of res.events) {
    const t = ev >>> 16, n = ev & 0xffff, x = W - dx + (t / ticks) * dx;
    ctx.fillStyle = polarity[n] > 0 ? C.acc : C.inh; ctx.fillRect(x, n * row, Math.max(1.5, dx / ticks * 1.5), Math.max(1.2, row));
  }
}

let last = 0;
function loop(ts) {
  requestAnimationFrame(loop);
  if (!state.playing || !A) return;
  if (ts - last < 16) return; last = ts;
  frame();
}
function frame() {
  const S = state.speed, ra = step(A, acc.a, S), rb = step(B, acc.b, S);
  scroll($('r4'), ra, S); scroll($('r2'), rb, S);
  for (const [k, r] of [['a', ra], ['b', rb]]) for (let i = 0; i < S; i += 50) rate[k].push(r.counts.slice(i, i + 50).reduce((x, y) => x + y, 0));
  for (const k of ['a', 'b']) if (rate[k].length > 400) rate[k].splice(0, rate[k].length - 400);
  state.tick += S; drawPop(); renderStats();
}

function stat(k, v) { return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
function renderStats() {
  const hz = (a) => (a.spikes / sub.n / Math.max(1e-9, a.ticks * 1e-4)).toFixed(0) + ' Hz';
  const us = (a) => a.ticks ? (a.ms * 1000 / a.ticks).toFixed(1) + ' µs' : '–';
  $('s4').innerHTML = stat('spikes', fmt(acc.a.spikes)) + stat('mean rate', hz(acc.a)) + stat('edge-ops', fmt(acc.a.ops)) + stat('wasm / tick', us(acc.a));
  $('s2').innerHTML = stat('spikes', fmt(acc.b.spikes)) + stat('mean rate', hz(acc.b)) + stat('edge-ops', fmt(acc.b.ops)) + stat('wasm / tick', us(acc.b));
  const v = $('verdict');
  if (acc.a.ticks < 1500) { v.className = 'verdict'; v.textContent = 'warming up…'; return; }
  const r = acc.b.spikes / Math.max(1, acc.a.spikes), o = acc.b.ops / Math.max(1, acc.a.ops), t = acc.a.ms > 0 ? (acc.b.ms / acc.b.ticks) / (acc.a.ms / acc.a.ticks) : NaN;
  const base = `ternary / INT4 → spikes <b>×${r.toFixed(2)}</b> · edge-ops <b>×${o.toFixed(2)}</b> · time/tick <b>${isNaN(t) ? '–' : '×' + t.toFixed(2)}</b> · wiring kept <b>${(100 * ternary.kept / sub.nnz).toFixed(0)}%</b>. `;
  if (r > 1.1) { v.className = 'verdict bad'; v.innerHTML = base + 'Ternary is <b>hyper-active</b>: flattening every synapse to ±1 weakened the strong inhibitory connections more than the excitatory ones.'; }
  else if (r < 0.9) { v.className = 'verdict'; v.innerHTML = base + 'Ternary is <b>under-driven</b>: pruning removed more drive than the gain puts back. Lower θ or raise the gain.'; }
  else { v.className = 'verdict ok'; v.innerHTML = base + 'Activity is <b>matched</b> to INT4' + (o < 0.9 ? ' with far less synaptic work. This is the fix.' : ', but it keeps most edges, so there is no saving yet.'); }
}

function line(ctx, data, W, H, max, col) {
  ctx.strokeStyle = col; ctx.lineWidth = 1.6 * devicePixelRatio; ctx.beginPath();
  data.forEach((v, i) => { const x = i / 399 * W, y = H - 6 - (v / max) * (H - 12); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
}
function drawPop() {
  const c = $('pop'), dpr = devicePixelRatio; c.width = c.clientWidth * dpr; c.height = 150 * dpr;
  const ctx = c.getContext('2d'), max = Math.max(10, ...rate.a, ...rate.b) * 1.1;
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; for (let i = 1; i < 4; i++) { ctx.beginPath(); ctx.moveTo(0, c.height * i / 4); ctx.lineTo(c.width, c.height * i / 4); ctx.stroke(); }
  line(ctx, rate.a, c.width, c.height, max, C.acc); line(ctx, rate.b, c.width, c.height, max, C.hot);
  ctx.fillStyle = C.mut; ctx.font = `${11 * dpr}px monospace`; ctx.fillText(max.toFixed(0) + ' spikes / 5 ms', 6 * dpr, 14 * dpr);
}

/* ---------- quantize lab ---------- */
function drawQuant() {
  const c = $('hist'), dpr = devicePixelRatio; c.width = c.clientWidth * dpr; c.height = 200 * dpr; const ctx = c.getContext('2d');
  const bins = 16, h = new Array(bins + 1).fill(0);
  for (const w of sub.counts) h[Math.min(bins, Math.abs(w))]++;
  const mx = Math.log10(Math.max(...h) + 1), bw = c.width / bins;
  for (let b = 1; b <= bins; b++) {
    const bh = Math.log10(h[b] + 1) / mx * (c.height - 28 * dpr), kept = b >= state.theta;
    ctx.fillStyle = kept ? C.acc : '#24433a'; ctx.fillRect((b - 1) * bw + 2, c.height - 18 * dpr - bh, bw - 4, bh);
    ctx.fillStyle = C.mut; ctx.font = `${10 * dpr}px monospace`; ctx.fillText(b === bins ? '16+' : b, (b - 1) * bw + bw / 2 - 5, c.height - 5 * dpr);
  }
  ctx.strokeStyle = C.hot; ctx.lineWidth = 2 * dpr; ctx.setLineDash([5 * dpr, 4 * dpr]); const x = (state.theta - 1) * bw; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height - 16 * dpr); ctx.stroke(); ctx.setLineDash([]);
  $('histnote').innerHTML = `|synapse count| on a log scale. Green bars are kept by θ=${state.theta}; dark bars are pruned. Kept <b>${(100 * ternary.kept / sub.nnz).toFixed(1)}%</b> of ${fmt(sub.nnz)} edges. Median count is 2; INT4 divides counts by ${P.QUANT_DIV}.`;
  const mv = (x) => (x / 65536).toFixed(3) + ' mV', rows = [];
  for (const w of [1, 2, 3, 5, 8, 16, 40]) {
    let q = Math.round(w / P.QUANT_DIV); if (q === 0) q = 1; q = Math.min(7, q);
    const kept = w >= state.theta;
    rows.push(`<tr><td>+${w} syn</td><td>${q}</td><td>${mv(q * P.W_SCALE)}</td><td>${kept ? '+1' : '<span style="color:var(--jade-muted)">dropped</span>'}</td><td>${kept ? mv(ternary.lut[1]) : '–'}</td></tr>`);
  }
  $('ex').tBodies[0].innerHTML = rows.join('');
  $('lutnote').innerHTML = `LUT = [0, ${ternary.lut[1]}, ${ternary.lut[2]}, 0] (Q16.16) → gains E×${ternary.gains[0].toFixed(2)} I×${ternary.gains[1].toFixed(2)}.`;
  packs();
}
function bitsOf(v, n) { return v.toString(2).padStart(n, '0'); }
function packs() {
  const k = 8, q = Array.from(q4.slice(0, k)), t = Array.from(ternary.codes.slice(0, k));
  const i4 = packInt4(Int8Array.from(q)), t2 = packTernary(Int8Array.from(t));
  const cell = (txt, cls) => `<span class="bit ${cls}">${txt}</span>`;
  const q4html = [...i4].map((b, i) => `${cell(bitsOf(b, 8), '')}<span class="note">= ${q[2 * i]}, ${q[2 * i + 1]}</span>`).join(' ');
  const t2html = [...t2].map((b, i) => `${cell(bitsOf(b, 8), '')}<span class="note">= ${t.slice(4 * i, 4 * i + 4).map((x) => (x > 0 ? '+' : x < 0 ? '−' : '0')).join(' ')}</span>`).join(' ');
  $('packs').innerHTML = `<div class="note">INT4: first ${k} edges → ${i4.length} bytes</div><div class="bytes">${q4html}</div><div class="note">Ternary (θ=${state.theta}): first ${k} kept edges → ${t2.length} bytes</div><div class="bytes">${t2html}</div>`;
}

/* ---------- sweep + esp ---------- */
function buildSweep(sw) {
  state.sweep = sw; const r = sw.rows, b = sw.baseline;
  $('swmeta').textContent = `INT4 baseline: ${(b.total_spikes / 1e6).toFixed(2)}M spikes, ${b.wall.toFixed(1)} s · winner by criteria: ${sw.winner ? sw.winner.cfg : 'none'}`;
  $('sw').tBodies[0].innerHTML = r.map((x) => `<tr data-t="${x.theta}" data-g="${x.gain}" class="${sw.winner && x.cfg === sw.winner.cfg ? 'win' : x.spike_ratio > 1.1 ? 'bad' : ''}"><td>${x.cfg}</td><td>${(100 * x.edge_frac).toFixed(0)}%</td><td>${x.spike_ratio.toFixed(2)}</td><td>${x.edge_ops_ratio.toFixed(2)}</td><td>${x.wall.toFixed(1)}</td><td>${x.speedup_vs_int4.toFixed(2)}×</td><td>${x.rate_spearman.toFixed(2)}</td><td>${x.final_snapshot_jaccard.toFixed(2)}</td></tr>`).join('');
  $('sw').tBodies[0].onclick = (e) => { const tr = e.target.closest('tr'); if (!tr) return; setCfg(+tr.dataset.t, tr.dataset.g); location.hash = '#lab'; };
  const thetas = [...new Set(r.map((x) => x.theta))], gains = ['unit', 'twn', 'drive'], cols = [C.mut, C.warn, C.acc];
  const chart = (id, key, ref, label) => {
    const c = $(id), dpr = devicePixelRatio; c.width = c.clientWidth * dpr; c.height = 230 * dpr; const ctx = c.getContext('2d'), W = c.width, H = c.height, pad = 30 * dpr;
    const all = r.map((x) => x[key]), lo = 0, hi = Math.max(...all, ref) * 1.1, X = (i) => pad + i / (thetas.length - 1) * (W - pad - 10 * dpr), Y = (v) => H - pad - (v - lo) / (hi - lo) * (H - pad - 10 * dpr);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.mut; ctx.font = `${10 * dpr}px monospace`;
    for (let g = 0; g <= 4; g++) { const v = lo + (hi - lo) * g / 4; ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), 0, Y(v) + 3 * dpr); }
    thetas.forEach((t, i) => ctx.fillText('θ' + t, Math.min(X(i) - 8 * dpr, W - 26 * dpr), H - 8 * dpr));
    ctx.strokeStyle = C.hot; ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.beginPath(); ctx.moveTo(pad, Y(ref)); ctx.lineTo(W, Y(ref)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.hot; ctx.fillText(label, W - 80 * dpr, Y(ref) - 4 * dpr);
    gains.forEach((g, gi) => { ctx.strokeStyle = cols[gi]; ctx.lineWidth = 2 * dpr; ctx.beginPath(); thetas.forEach((t, i) => { const row = r.find((x) => x.theta === t && x.gain === g); i ? ctx.lineTo(X(i), Y(row[key])) : ctx.moveTo(X(i), Y(row[key])); }); ctx.stroke(); ctx.fillStyle = cols[gi]; ctx.fillText(g, W - 40 * dpr, Y(r.find((x) => x.theta === thetas.at(-1) && x.gain === g)[key]) - 4 * dpr * (gi - 1)); });
  };
  chart('c-spk', 'spike_ratio', 1, 'INT4 = 1.0'); chart('c-spd', 'speedup_vs_int4', 1, 'INT4 = 1.0×');
}
function buildEsp() {
  const rows = [['Subnetwork', 'N = 4,096 neurons · 75,746 kept edges (θ=3, drive)'], ['Flash / rodata', '148 KiB indices (uint16) + 18 KiB weights (2-bit) + 16 KiB indptr'],
    ['RAM (state)', '60 KiB: V, G (int32), refractory (u8), queue (u16), stimulus (int32)'], ['Speed on host (same C)', 'bit-exact vs reference engine; on-device µs/tick not measured yet'],
    ['Whole brain at θ=3', '≈4.9 M kept edges → ≳15 MB of indices alone (3-byte), + ≈1.8 MB state: needs a 16–32 MB PSRAM class board, so not an 8 MB ESP32-S3 (estimate)']];
  $('espt').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td style="text-align:left">${b}</td></tr>`).join('');
}

function setCfgNow(t, g) { state.theta = t; state.gain = g; $('theta').value = t; $('thv').textContent = t; $('gain').value = g; }
function setCfg(t, g) { state.theta = t; state.gain = g; $('theta').value = t; $('thv').textContent = t; $('gain').value = g; rebuild(); }
function wire() {
  $('play').onclick = () => { state.playing = !state.playing; $('play').textContent = state.playing ? '⏸ Pause' : '▶ Play'; };
  $('reset').onclick = () => rebuild();
  $('p-paradox').onclick = () => setCfg(1, 'unit'); $('p-fix').onclick = () => setCfg(3, 'drive');
  $('theta').oninput = (e) => { $('thv').textContent = e.target.value; }; $('theta').onchange = (e) => setCfg(+e.target.value, state.gain);
  $('gain').onchange = (e) => setCfg(state.theta, e.target.value);
  $('speed').oninput = (e) => { state.speed = +e.target.value; $('spv').textContent = e.target.value; };
  addEventListener('resize', () => { drawQuant(); buildSweep(state.sweep); });
}
boot().catch((e) => { document.body.insertAdjacentHTML('afterbegin', `<pre style="color:#ff3d4f;padding:20px;position:relative;z-index:9">FlyQ demo failed to start: ${e.message}</pre>`); console.error(e); });
