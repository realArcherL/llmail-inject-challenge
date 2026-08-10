import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

export const REQUIRED_DATASET_FILES = [
  'system_prompt.json',
];

export const PHASE1_SPOTLIGHT_LEVELS = [
  'level1e',
  'level1f',
  'level2e',
  'level2f',
  'level3e',
  'level3f',
  'level4e',
  'level4f',
];

export function validateDatasetFiles(dataDir, phase = null) {
  const required = [...REQUIRED_DATASET_FILES];
  if (phase) {
    required.push(`raw_submissions_${phase}.jsonl`);
  } else {
    required.push('raw_submissions_phase1.jsonl', 'raw_submissions_phase2.jsonl');
  }

  const missing = required.filter((file) => !existsSync(join(dataDir, file)));
  if (missing.length > 0) {
    throw new Error(`Missing dataset files in ${dataDir}: ${missing.join(', ')}`);
  }
}

export function readJson(dataDir, fileName) {
  return JSON.parse(readFileSync(join(dataDir, fileName), 'utf8'));
}

export function rawSubmissionPath(dataDir, phase) {
  return join(dataDir, `raw_submissions_${phase}.jsonl`);
}

export function scenarioNumberFromLevel(level) {
  const match = /^level([1-4])([a-z])$/i.exec(level);
  if (!match) throw new Error(`Invalid level name "${level}".`);
  return Number.parseInt(match[1], 10);
}

export function normalizeSubmission(raw) {
  let objectives = raw.objectives;
  if (typeof objectives === 'string') {
    try {
      objectives = JSON.parse(objectives);
    } catch {
      objectives = {};
    }
  }

  return {
    rowKey: raw.RowKey,
    timestamp: raw.Timestamp,
    jobId: raw.job_id,
    teamId: raw.team_id,
    scenario: raw.scenario,
    subject: raw.subject || '',
    body: raw.body || '',
    objectives: objectives || {},
    output: raw.output || '',
    scheduledTime: raw.scheduled_time,
    startedTime: raw.started_time,
    completedTime: raw.completed_time,
  };
}

/**
 * @param {{
 *   dataDir: string;
 *   phase: string;
 *   level?: string;
 *   levels?: string[];
 *   scenario?: string;
 *   limit?: number;
 * }} options
 */
export async function loadSubmissions(options) {
  const {
    dataDir,
    phase,
    level,
    levels,
    scenario,
    limit = 1,
  } = options;
  const filePath = rawSubmissionPath(dataDir, phase);
  const input = createReadStream(filePath, { encoding: 'utf8' });
  const lines = createInterface({ input, crlfDelay: Number.POSITIVE_INFINITY });
  const rows = [];

  for await (const line of lines) {
    if (!line.trim()) continue;
    const row = normalizeSubmission(JSON.parse(line));

    if (level && row.scenario !== level) continue;
    if (!level && Array.isArray(levels) && levels.length > 0 && !levels.includes(row.scenario)) continue;
    if (scenario && `scenario_${scenarioNumberFromLevel(row.scenario)}` !== scenario) continue;

    rows.push(row);
    if (rows.length >= limit) break;
  }

  return rows;
}
