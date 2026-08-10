import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * @typedef {{
 *   dataDir: string;
 *   lmstudioBaseUrl?: string;
 *   lmstudioModel?: string;
 *   lmstudioApiKey: string;
 * }} AppConfig
 */

export function loadDotenv(path = '.env') {
  if (!existsSync(path)) return;

  const text = readFileSync(path, 'utf8');
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;

    const index = trimmed.indexOf('=');
    const key = trimmed.slice(0, index).trim();
    let value = trimmed.slice(index + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = value;
  }
}

/**
 * @returns {AppConfig}
 */
export function loadConfig() {
  loadDotenv();

  return {
    dataDir: resolve(process.env.DATA_DIR || './data/data'),
    lmstudioBaseUrl: process.env.LMSTUDIO_BASE_URL,
    lmstudioModel: process.env.LMSTUDIO_MODEL,
    lmstudioApiKey: process.env.LMSTUDIO_API_KEY || 'lm-studio',
  };
}

export function parseArgs(argv) {
  /** @type {Record<string, string | boolean>} */
  const args = {};

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;

    const stripped = token.slice(2);
    const eqIndex = stripped.indexOf('=');
    if (eqIndex !== -1) {
      args[stripped.slice(0, eqIndex)] = stripped.slice(eqIndex + 1);
      continue;
    }

    const next = argv[i + 1];
    if (next && !next.startsWith('--')) {
      args[stripped] = next;
      i += 1;
    } else {
      args[stripped] = true;
    }
  }

  return args;
}

export function asPositiveInt(value, fallback) {
  if (value === undefined || value === true) return fallback;
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function normalizePhase(value = 'phase1') {
  const phase = String(value).toLowerCase();
  if (phase === '1' || phase === 'phase1') return 'phase1';
  if (phase === '2' || phase === 'phase2') return 'phase2';
  throw new Error(`Unsupported phase "${value}". Use phase1 or phase2.`);
}
