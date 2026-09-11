import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const REPO = 'microsoft/llmail-inject-challenge';
const BASE_URL = `https://huggingface.co/datasets/${REPO}/resolve/main/data`;
// Node's fetch asks for gzip; Hugging Face then omits content-length on small files.
const HEADERS = { 'Accept-Encoding': 'identity' };
const ATTEMPTS = 3;

export const CORE_FILES = [
  'system_prompt.json',
  'scenarios.json',
  'levels_descriptions.json',
  'objectives_descriptions.json',
  'emails_for_fp_tests.json',
  'raw_submissions_phase1.jsonl',
  'raw_submissions_phase2.jsonl',
];

export const LABELLED_FILES = [
  'labelled_unique_submissions_phase1.json',
  'labelled_unique_submissions_phase2.json',
];

const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;
const sizeOf = (path) => (existsSync(path) ? statSync(path).size : -1);

async function remoteSize(url) {
  const res = await fetch(url, { method: 'HEAD', headers: HEADERS });
  if (!res.ok) throw new Error(`HEAD ${url} -> ${res.status}`);
  return Number(res.headers.get('content-length')) || null;
}

/** Fetch one file into dataDir. Skips a complete file, resumes a partial one. */
export async function downloadFile(dataDir, name, log = () => {}) {
  const url = `${BASE_URL}/${name}`;
  const path = join(dataDir, name);
  const part = `${path}.part`;
  const expected = await remoteSize(url);

  if (expected && sizeOf(path) === expected) {
    log(`${'skip'.padEnd(10)}${name} (${mb(expected)})`);
    return 'skipped';
  }

  let offset = Math.max(sizeOf(part), 0);
  if (expected && offset >= expected) {
    rmSync(part); // stale or oversized partial: start over
    offset = 0;
  }

  const headers = offset ? { ...HEADERS, Range: `bytes=${offset}-` } : HEADERS;
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  const resumed = res.status === 206;

  log(`${(resumed ? 'resume' : 'download').padEnd(10)}${name}${expected ? ` (${mb(expected)})` : ''}${resumed ? ` from ${mb(offset)}` : ''}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(part, { flags: resumed ? 'a' : 'w' }));

  const actual = sizeOf(part);
  if (expected && actual !== expected) throw new Error(`${name}: got ${actual} bytes, expected ${expected}`);
  renameSync(part, path);
  return 'downloaded';
}

/** @param {{ dataDir: string }} config */
export async function downloadDataset(config, { labelled = false, log = console.log } = {}) {
  const files = labelled ? [...CORE_FILES, ...LABELLED_FILES] : CORE_FILES;
  mkdirSync(config.dataDir, { recursive: true });
  log(`${REPO} -> ${config.dataDir}`);

  const results = {};
  for (const name of files) {
    for (let attempt = 1; ; attempt += 1) {
      try {
        results[name] = await downloadFile(config.dataDir, name, log);
        break;
      } catch (error) {
        if (attempt === ATTEMPTS) throw error;
        log(`${'retry'.padEnd(10)}${name}: ${error.message}`);
      }
    }
  }
  return results;
}
