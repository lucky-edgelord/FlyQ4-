// FlyQ in the browser: model maths (quantizers, packers) + WASM engine wrapper.
// Pure ES module, no dependencies. Works in browsers and in Node (see tests/test_wasm.mjs).

export const P = {
  W_SCALE: Math.round(0.275 * 65536),
  V_REST: Math.round(-52.0 * 65536),
  V_TH: Math.round(-45.0 * 65536),
  ALPHA_M: Math.round((1 - Math.exp(-0.1 / 20)) * 65536),
  G_SHIFT: 6, REFRAC: 22, STIM_Q16: Math.round(0.55 * 65536),
  QUANT_DIV: 2.0,           // p95(|w|)/6 on the full 138,639-neuron FlyWire v783 graph
};

/** numpy.round semantics (half to even) */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

export function parseSubnet(buf) {
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'FQS1') throw new Error('bad subnet magic');
  const n = dv.getUint32(4, true), nnz = dv.getUint32(8, true), nStim = dv.getUint32(12, true);
  let o = 16;
  const indptr = new Uint32Array(buf.slice(o, o + 4 * (n + 1))); o += 4 * (n + 1);
  const indices = new Uint16Array(buf.slice(o, o + 2 * nnz)); o += 2 * nnz;
  const counts = new Int16Array(buf.slice(o, o + 2 * nnz));
  return { n, nnz, nStim, indptr, indices, counts };
}

/** FlyQ4 rule: INT4 [-8,+7], a real synapse never rounds to 0 */
export function quantizeInt4(counts, div = P.QUANT_DIV) {
  const q = new Int8Array(counts.length);
  for (let i = 0; i < counts.length; i++) {
    const w = counts[i];
    let v = roundHalfEven(w / div);
    if (v === 0) v = w > 0 ? 1 : -1;
    q[i] = Math.max(-8, Math.min(7, v));
  }
  return q;
}

export function packInt4(q) {
  const out = new Uint8Array((q.length + 1) >> 1);
  for (let i = 0; i < q.length; i++) out[i >> 1] |= (q[i] & 15) << ((i & 1) * 4);
  return out;
}
export function packTernary(t) {
  const out = new Uint8Array((t.length + 3) >> 2);
  for (let i = 0; i < t.length; i++) {
    const c = t[i] > 0 ? 1 : t[i] < 0 ? 2 : 0;
    out[i >> 2] |= c << ((i & 3) * 2);
  }
  return out;
}

export const GAINS = ['unit', 'twn', 'drive'];

/** Deadzone ternary + per-polarity gain. Mirrors flyq4/quant.py::ternarize exactly. */
export function ternarize(sub, q4, theta, gain = 'unit') {
  const { n, indptr, indices, counts } = sub;
  const keep = new Uint8Array(counts.length);
  const newPtr = new Uint32Array(n + 1);
  let kept = 0;
  for (let u = 0; u < n; u++) {
    for (let e = indptr[u]; e < indptr[u + 1]; e++) if (Math.abs(counts[e]) >= theta) { keep[e] = 1; kept++; }
    newPtr[u + 1] = kept;
  }
  const idx = new Uint16Array(kept), codes = new Int8Array(kept);
  let k = 0, posAll = 0, negAll = 0, posK = 0, negK = 0, magAllPos = 0, magAllNeg = 0, magKPos = 0, magKNeg = 0;
  for (let e = 0; e < counts.length; e++) {
    const m = Math.abs(q4[e]);
    if (counts[e] > 0) { posAll++; magAllPos += m; } else { negAll++; magAllNeg += m; }
    if (keep[e]) {
      idx[k] = indices[e]; codes[k] = counts[e] > 0 ? 1 : -1; k++;
      if (counts[e] > 0) { posK++; magKPos += m; } else { negK++; magKNeg += m; }
    }
  }
  let aE = 1, aI = 1;
  if (gain === 'twn') { aE = posK ? magKPos / posK : 1; aI = negK ? magKNeg / negK : 1; }
  else if (gain === 'drive') { aE = magAllPos / Math.max(1, posK); aI = magAllNeg / Math.max(1, negK); }
  const lut = [0, Math.round(aE * P.W_SCALE), -Math.round(aI * P.W_SCALE), 0];
  return { indptr: newPtr, indices: idx, codes, lut, gains: [aE, aI], kept };
}

/** WASM engine: two independent nets (INT4 and ternary) share one module instance. */
export class FlyWasm {
  static async load(wasmBytesOrResponse) {
    const src = wasmBytesOrResponse;
    const mod = src instanceof ArrayBuffer || ArrayBuffer.isView(src)
      ? await WebAssembly.instantiate(src, {}) : await WebAssembly.instantiateStreaming(src, {});
    return new FlyWasm((mod.instance || mod).exports);
  }
  constructor(x) { this.x = x; }
  get mem() { return this.x.memory.buffer; }

  /** @returns a handle { ptr, n } */
  makeNet(sub, kind, graph) {
    const x = this.x, n = sub.n;
    const net = x.net_create(n, P.V_REST, P.V_TH, P.ALPHA_M, P.W_SCALE, P.G_SHIFT, P.REFRAC);
    const put = (arr) => { const p = x.alloc(arr.byteLength); new Uint8Array(this.mem, p, arr.byteLength).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)); return p; };
    const pPtr = put(graph.indptr), pIdx = put(graph.indices), pW = put(graph.packed);
    x.net_bind(net, kind === 'ternary' ? 1 : 0, pPtr, pIdx, pW);
    if (kind === 'ternary') x.net_set_lut(net, ...graph.lut);
    for (let i = 0; i < sub.nStim; i++) x.net_set_stim(net, i, P.STIM_Q16);
    const h = { ptr: net, n, kind, nStim: sub.nStim };
    this.reset(h);
    return h;
  }
  reset(h) { this.x.net_reset(h.ptr, h.nStim); }

  /** run `steps` ticks, returns { counts:Uint32Array, events:Uint32Array(t<<16|neuron), edgeOps } */
  run(h, steps, recCap = 60000) {
    const x = this.x;
    // scratch buffers are allocated once (the bump allocator never frees)
    const sc = this._sc || (this._sc = {});
    if (!sc.pc || sc.steps < steps) { sc.pc = x.alloc(4 * steps); sc.steps = steps; }
    if (!sc.pr || sc.cap < recCap) { sc.pr = x.alloc(4 * recCap); sc.cap = recCap; }
    const pc = sc.pc, pr = sc.pr;
    const nrec = x.net_run(h.ptr, steps, pc, pr, recCap);
    const counts = new Uint32Array(this.mem.slice(pc, pc + 4 * steps));
    const events = new Uint32Array(this.mem.slice(pr, pr + 4 * nrec));
    return { counts, events, edgeOps: x.net_take_edge_ops(h.ptr) };
  }
  totals(h) { const p = this.x.net_totals(h.ptr); return new Uint32Array(this.mem.slice(p, p + 4 * h.n)); }
  voltages(h) { const p = this.x.net_V(h.ptr); return new Int32Array(this.mem.slice(p, p + 4 * h.n)); }
}
