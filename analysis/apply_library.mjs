// Experiment 02: apply the spotlighting-datamarking library, the real npm package, to the
// exact prompt pieces exported by analysis/build_library_experiment.py.
//
// Placement mirrors Microsoft's spotlighting, so only the defense differs between conditions:
//   system rules + "\n" + <library prompt>, blank line, tool prompt, query, blank line, <defended emails>
// The query and the tool prompt are untouched. Library defaults are used as shipped:
// alphanumeric markers of 7-12 characters, p = 0.5, minGap = 1, sanitize on, sandwich on.
// Markers are random per call, so every generated marker is saved next to its prompt.
//
// Run from the repo root:  node analysis/apply_library.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { DataMarkingViaSpotlighting } from 'spotlighting-datamarking';

const EXP_DIR = 'runs/02-library-defenses';
const lib = new DataMarkingViaSpotlighting();
const pkg = JSON.parse(readFileSync('node_modules/spotlighting-datamarking/package.json', 'utf8'));

// Same layout as llmail_prompt.assemble (Phi3LLM.call_model in Microsoft's agent).
function assemble(parts, systemExtra = '', query = parts.query, emails = parts.formatted_emails) {
  return `${parts.system_prompt}${systemExtra}\n\n${parts.tool_prompt}` + '\n' + query + '\n\n' + emails;
}

const withPrompt = (p, r) => ({
  prompt: assemble(p, '\n' + r.prompt.trimEnd(), p.query, r.markedText),
  marker: r.dataMarker ?? null,
});

const MODES = {
  lib_sanitize: (p) => ({ prompt: assemble(p, '', p.query, lib.sanitizeText(p.formatted_emails)), marker: null }),
  lib_markdata: (p) => withPrompt(p, lib.markData(p.formatted_emails)),
  lib_randommark: (p) => withPrompt(p, lib.randomlyMarkData(p.formatted_emails)),
  lib_base64: (p) => withPrompt(p, lib.base64EncodeData(p.formatted_emails)),
};

const rows = readFileSync(`${EXP_DIR}/parts.jsonl`, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const out = [];
for (const r of rows) {
  if (assemble(r.parts) !== r.prompt_none) {
    throw new Error(`JS layout differs from Python layout for ${r.base_id}`);
  }
  const base = { base_id: r.base_id, kind: r.kind, tool_name: r.tool_name, recorded_label: r.recorded_label ?? null };
  const conds = [];
  if (r.kind === 'clean') conds.push(['none', { prompt: r.prompt_none, marker: null }]);
  conds.push(['ms_spotlight', { prompt: r.prompt_ms_spotlight, marker: '0a8cb271' }]);
  for (const [name, fn] of Object.entries(MODES)) conds.push([name, fn(r.parts)]);
  for (const [condition, { prompt, marker }] of conds) {
    out.push({
      id: `${r.base_id}__${condition}`, ...base, condition,
      data_marker: marker,
      defense_source: condition.startsWith('lib_') ? `spotlighting-datamarking@${pkg.version}` : (condition === 'ms_spotlight' ? 'microsoft/llmail-inject-challenge' : null),
      prompt,
    });
  }
}
writeFileSync(`${EXP_DIR}/prompts.jsonl`, out.map((o) => JSON.stringify(o)).join('\n') + '\n');
const counts = {};
for (const o of out) counts[`${o.kind} / ${o.condition}`] = (counts[`${o.kind} / ${o.condition}`] || 0) + 1;
console.log(`JS layout matches Python for all ${rows.length} base items`);
console.log(`wrote ${out.length} prompts to ${EXP_DIR}/prompts.jsonl using spotlighting-datamarking@${pkg.version}`);
console.table(counts);
