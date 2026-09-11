import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, it } from 'node:test';

import { downloadDataset, downloadFile } from '../src/download.js';

const CONTENT = 'hello dataset world';
const FILE = 'system_prompt.json';

/** Fake of the Hugging Face resolve endpoint with Range support. */
function fakeFetch(calls) {
  return async (url, init = {}) => {
    calls.push({ method: init.method || 'GET', range: init.headers?.Range });
    const bytes = Buffer.from(CONTENT);
    if (init.method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(bytes.length) } });
    }
    const start = Number(/^bytes=(\d+)-$/.exec(init.headers?.Range || '')?.[1] ?? 0);
    return new Response(bytes.subarray(start), { status: start ? 206 : 200 });
  };
}

describe('downloadFile', () => {
  const realFetch = globalThis.fetch;
  let dir;
  let calls;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'llmail-download-'));
    calls = [];
    globalThis.fetch = fakeFetch(calls);
  });

  afterEach(() => {
    globalThis.fetch = realFetch;
    rmSync(dir, { recursive: true, force: true });
  });

  const read = () => readFileSync(join(dir, FILE), 'utf8');

  it('downloads a fresh file', async () => {
    assert.equal(await downloadFile(dir, FILE), 'downloaded');
    assert.equal(read(), CONTENT);
    assert.equal(existsSync(join(dir, `${FILE}.part`)), false);
  });

  it('skips a complete file after only a HEAD request', async () => {
    writeFileSync(join(dir, FILE), CONTENT);
    assert.equal(await downloadFile(dir, FILE), 'skipped');
    assert.deepEqual(calls.map((c) => c.method), ['HEAD']);
  });

  it('resumes a partial file with a Range request', async () => {
    writeFileSync(join(dir, `${FILE}.part`), CONTENT.slice(0, 5));
    assert.equal(await downloadFile(dir, FILE), 'downloaded');
    assert.equal(calls[1].range, 'bytes=5-');
    assert.equal(read(), CONTENT);
  });

  it('restarts when the partial file is larger than the remote', async () => {
    writeFileSync(join(dir, `${FILE}.part`), `${CONTENT}garbage`);
    assert.equal(await downloadFile(dir, FILE), 'downloaded');
    assert.equal(calls[1].range, undefined);
    assert.equal(read(), CONTENT);
  });

  it('re-downloads a file whose size does not match the remote', async () => {
    writeFileSync(join(dir, FILE), 'stale');
    assert.equal(await downloadFile(dir, FILE), 'downloaded');
    assert.equal(read(), CONTENT);
  });

  it('retries a failed file before giving up', async () => {
    const inner = globalThis.fetch;
    let failures = 1;
    globalThis.fetch = async (url, init) => {
      if (init?.method !== 'HEAD' && failures-- > 0) throw new Error('socket hang up');
      return inner(url, init);
    };
    const log = [];
    const results = await downloadDataset({ dataDir: dir }, { log: (line) => log.push(line) });
    assert.equal(results[FILE], 'downloaded');
    assert.ok(log.some((line) => line.startsWith(`retry     ${FILE}`)));
    assert.equal(read(), CONTENT);
  });
});
