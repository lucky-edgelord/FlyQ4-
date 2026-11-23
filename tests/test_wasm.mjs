// WASM engine + JS quantizers must reproduce the C engine's spike trains tick for tick.
//   node tests/test_wasm.mjs
import fs from 'node:fs';
import { parseSubnet, quantizeInt4, packInt4, ternarize, packTernary, FlyWasm } from '../docs/flyq.js';
const root = new URL('..', import.meta.url).pathname;
const sub = parseSubnet(fs.readFileSync(root + 'docs/assets/subnet.bin').buffer.slice(0));
const { ticks, ref } = JSON.parse(fs.readFileSync(root + 'tests/wasm_ref.json', 'utf8'));
const fw = await FlyWasm.load(fs.readFileSync(root + 'docs/flyq.wasm'));
const q4 = quantizeInt4(sub.counts);
let bad = 0;
function check(name, h) {
  const r = fw.run(h, ticks), want = ref[name];
  const same = r.counts.length === want.counts.length && r.counts.every((c, i) => c === want.counts[i]);
  const eo = r.edgeOps === (want.edge_ops >>> 0);
  console.log(`${same && eo ? 'PASS' : 'FAIL'} ${name.padEnd(18)} spikes=${r.counts.reduce((a, b) => a + b, 0)} (want ${want.total})`);
  if (!(same && eo)) bad++;
}
check('int4', fw.makeNet(sub, 'int4', { indptr: sub.indptr, indices: sub.indices, packed: packInt4(q4) }));
for (const [th, g] of [[1, 'unit'], [3, 'drive'], [2, 'twn']]) {
  const t = ternarize(sub, q4, th, g);
  const want = ref[`ternary_${th}_${g}`];
  if (JSON.stringify(t.lut) !== JSON.stringify(want.lut)) { console.log('FAIL lut', t.lut, want.lut); bad++; }
  check(`ternary_${th}_${g}`, fw.makeNet(sub, 'ternary', { indptr: t.indptr, indices: t.indices, packed: packTernary(t.codes), lut: t.lut }));
}
process.exit(bad ? 1 : 0);
