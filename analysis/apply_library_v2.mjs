// Experiment 02b: corrected variants of the spotlighting-datamarking library.
//
// Built against the LOCAL library checkout, which has the `tokenizer` option; the copy in
// node_modules is the published 2.0.0-alpha and does not. Override with LIB_PATH if your
// checkout lives elsewhere.
//
// Variants, on the same inputs as experiment 02 (runs/02-library-defenses/parts.jsonl):
//   v2_uni_spaces         markData, short Unicode marker (1-2 private-use chars)
//   v2_uni_random_phi3    randomlyMarkData at Phi-3's token boundaries, short Unicode marker
//   v2_alnum_random_phi3  randomlyMarkData at Phi-3's token boundaries, stock alphanumeric marker
//   v2_uni_words          randomlyMarkData at word boundaries only, short Unicode marker
//
// The last one is the "spaces only" idea: it is expected to leave unspaced attack text unmarked,
// which is why it is measured rather than shipped.
//
// Run from the repo root:  node analysis/apply_library_v2.mjs [--limit N]
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { AutoTokenizer } from '@huggingface/transformers';

const EXP_DIR = 'runs/02-library-defenses';

// Which copy of the library runs, in order:
//   1. LIB_PATH, if set
//   2. the installed spotlighting-datamarking, if it has the tokenizer option (2.1.0-alpha+)
//   3. a sibling checkout at ../spotlighting-datamarking
// Whatever wins is recorded in library_provenance.json, so a result can always be traced back.
const HAS_OPTION = 'assertTokenizer';
async function resolveLibrary() {
  const candidates = [];
  if (process.env.LIB_PATH) candidates.push(['LIB_PATH', new URL(process.env.LIB_PATH, import.meta.url)]);
  candidates.push(['installed package', new URL('../node_modules/spotlighting-datamarking/index.js', import.meta.url)]);
  candidates.push(['sibling checkout', new URL('../../spotlighting-datamarking/index.js', import.meta.url)]);

  const tried = [];
  for (const [how, url] of candidates) {
    const file = fileURLToPath(url); // pathname keeps %20 for spaces in folder names
    if (!existsSync(file)) {
      tried.push(`${how}: not found at ${file}`);
      continue;
    }
    const source = readFileSync(file, 'utf8');
    if (!source.includes(HAS_OPTION)) {
      tried.push(`${how}: ${file} has no tokenizer option`);
      continue;
    }
    const pkgFile = fileURLToPath(new URL('package.json', url));
    const pkg = existsSync(pkgFile) ? JSON.parse(readFileSync(pkgFile, 'utf8')) : {};
    return {
      how,
      file,
      version: pkg.version ?? 'unknown',
      name: pkg.name ?? 'spotlighting-datamarking',
      sha256: createHash('sha256').update(source).digest('hex'),
      module: await import(url.href),
    };
  }
  throw new Error(`no library with the tokenizer option. Tried:\n  ${tried.join('\n  ')}`);
}

const library = await resolveLibrary();
const { DataMarkingViaSpotlighting } = library.module;
const provenance = {
  resolved_by: library.how,
  path: library.file,
  package: library.name,
  version: library.version,
  index_js_sha256: library.sha256,
  recorded_utc: new Date().toISOString(),
};
writeFileSync(`${EXP_DIR}/library_provenance.json`, JSON.stringify(provenance, null, 2) + '\n');
console.log(`library: ${provenance.package}@${provenance.version} via ${provenance.resolved_by}`);
console.log(`         ${provenance.path}`);
console.log(`         sha256 ${provenance.index_js_sha256.slice(0, 16)}...`);
if (process.argv.includes('--provenance-only')) process.exit(0);

const limitArg = process.argv.indexOf('--limit');
const limit = limitArg === -1 ? 0 : Number(process.argv[limitArg + 1]);

// Same layout as llmail_prompt.assemble (Phi3LLM.call_model in Microsoft's agent).
function assemble(parts, systemExtra = '', query = parts.query, emails = parts.formatted_emails) {
  return `${parts.system_prompt}${systemExtra}\n\n${parts.tool_prompt}` + '\n' + query + '\n\n' + emails;
}
const withPrompt = (p, r) => ({
  prompt: assemble(p, '\n' + r.prompt.trimEnd(), p.query, r.markedText),
  marker: r.dataMarker ?? null,
  marked: r.markedText,
});

// Phi-3's own tokenizer: markers land only where Phi-3 itself splits.
const hf = await AutoTokenizer.from_pretrained('microsoft/Phi-3-medium-128k-instruct');
const phi3 = {
  encode: text => hf.encode(text, { add_special_tokens: false }),
  decode: ids => hf.decode(ids),
};
// Whitespace "tokenizer": every token is a word with its leading space, so boundaries are word gaps.
const wordTokenizer = () => {
  const vocab = new Map();
  const pieces = [];
  return {
    encode(text) {
      return (text.match(/\s*\S+|\s+/g) || []).map(piece => {
        if (!vocab.has(piece)) {
          vocab.set(piece, pieces.length);
          pieces.push(piece);
        }
        return vocab.get(piece);
      });
    },
    decode: ids => ids.map(id => pieces[id]).join(''),
  };
};

const uni = new DataMarkingViaSpotlighting(1, 2, 0.5, 1, 'unicode'); // short private-use marker
const alnum = new DataMarkingViaSpotlighting(); // stock: 7-12 alphanumeric chars

// `mode` says what the marked text should look like once the markers are taken back out:
//   'spaces' replaces every whitespace character with a marker, so whitespace is gone
//   'insert' only inserts markers, so the text must come back byte for byte
const VARIANTS = {
  v2_uni_spaces: { lib: uni, mode: 'spaces', build: p => uni.markData(p.formatted_emails) },
  v2_uni_random_phi3: { lib: uni, mode: 'insert', build: p => uni.randomlyMarkData(p.formatted_emails, { tokenizer: phi3 }) },
  v2_alnum_random_phi3: { lib: alnum, mode: 'insert', build: p => alnum.randomlyMarkData(p.formatted_emails, { tokenizer: phi3 }) },
  v2_uni_words: { lib: uni, mode: 'insert', build: p => uni.randomlyMarkData(p.formatted_emails, { tokenizer: wordTokenizer() }) },
};

function checkIntegrity(name, mode, lib, marked, marker, original) {
  const clean = lib.sanitizeText(original); // sanitising legitimately drops invisible characters
  const stripped = marker ? marked.split(marker).join('') : marked;
  const expected = mode === 'spaces' ? clean.replace(/\s/g, '') : clean;
  if (stripped !== expected) throw new Error(`${name} altered the email text`);
}

let rows = readFileSync(`${EXP_DIR}/parts.jsonl`, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
if (limit) rows = rows.slice(0, limit);

const out = [];
const timing = {};
const stats = {};
for (const [name, { lib, mode, build }] of Object.entries(VARIANTS)) {
  const started = Date.now();
  let markerChars = 0;
  for (const r of rows) {
    const { prompt, marker, marked } = withPrompt(r.parts, build(r.parts));
    checkIntegrity(`${name} (${r.base_id})`, mode, lib, marked, marker, r.parts.formatted_emails);
    markerChars += marker ? marker.length : 0;
    out.push({
      id: `${r.base_id}__${name}`,
      base_id: r.base_id,
      kind: r.kind,
      tool_name: r.tool_name,
      recorded_label: r.recorded_label ?? null,
      condition: name,
      data_marker: marker,
      defense_source: `${provenance.package}@${provenance.version} (${provenance.how})`,
      prompt,
    });
  }
  timing[name] = ((Date.now() - started) / 1000).toFixed(1) + 's';
  stats[name] = { 'avg marker chars': (markerChars / rows.length).toFixed(1) };
}

writeFileSync(`${EXP_DIR}/prompts_v2.jsonl`, out.map(o => JSON.stringify(o)).join('\n') + '\n');
console.log(`inputs: ${rows.length} (${rows.filter(r => r.kind === 'attack').length} attacks)`);
console.log(`wrote ${out.length} prompts to ${EXP_DIR}/prompts_v2.jsonl`);
console.log('every variant preserved the original email text exactly');
console.table(timing);
console.table(stats);
