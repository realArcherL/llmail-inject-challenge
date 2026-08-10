import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  loadSubmissions,
  readJson,
  scenarioNumberFromLevel,
  validateDatasetFiles,
} from './dataset.js';
import { buildDatasetReplayRun } from './scenarios.js';
import { applySpotlighting } from './spotlight.js';
import { callLmStudio, pingLmStudio } from './llmClient.js';
import { parseToolCalls, scoreToolCalls } from './toolParser.js';

function preview(value, maxLength = 420) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}...`;
}

function trace(options, step, details = {}) {
  if (!options.trace) return;
  console.log(`[trace] ${step}`);
  console.log(JSON.stringify(details, null, 2));
}

export async function setup(config) {
  validateDatasetFiles(config.dataDir);
  const systemPrompt = readJson(config.dataDir, 'system_prompt.json');

  const spotlightImport = await import('spotlighting-datamarking');
  if (typeof spotlightImport.DataMarkingViaSpotlighting !== 'function') {
    throw new Error('spotlighting-datamarking did not export DataMarkingViaSpotlighting.');
  }

  const ping = await pingLmStudio(config).catch((error) => ({
    ok: false,
    skipped: false,
    message: error.message,
  }));

  return {
    dataDir: config.dataDir,
    datasetFiles: 'ok',
    mode: 'dataset-replay',
    systemPromptKeys: Object.keys(systemPrompt),
    spotlightPackage: 'ok',
    lmStudio: ping,
  };
}

export async function runRows(config, options) {
  validateDatasetFiles(config.dataDir);
  trace(options, 'config', {
    dataDir: config.dataDir,
    lmstudioBaseUrl: config.lmstudioBaseUrl,
    lmstudioModel: config.lmstudioModel || '(auto-detect from /v1/models)',
    phase: options.phase,
    level: options.level || '(any level)',
    scenario: options.scenario || '(any scenario)',
    limit: options.limit,
    mode: 'dataset-replay',
    microsoftSpotlightLevelFilter: options.spotlightOnly,
  });

  let levelsDescriptions = {};
  try {
    levelsDescriptions = readJson(config.dataDir, 'levels_descriptions.json');
  } catch {
    levelsDescriptions = {};
  }
  const prompts = readJson(config.dataDir, 'system_prompt.json');
  const rows = await loadSubmissions({
    dataDir: config.dataDir,
    phase: options.phase,
    levelsDescriptions,
    level: options.level,
    scenario: options.scenario,
    spotlightOnly: options.spotlightOnly,
    limit: options.limit,
  });

  if (rows.length === 0) {
    throw new Error('No rows matched the requested filters.');
  }
  trace(options, 'selected rows', {
    count: rows.length,
    rows: rows.map((row) => ({
      rowKey: row.rowKey,
      scenario: row.scenario,
      subject: row.subject,
      bodyPreview: preview(row.body),
      originalObjectives: row.objectives,
    })),
  });

  const results = [];
  for (const row of rows) {
    const scenarioRun = buildDatasetReplayRun(row);
    trace(options, 'scenario context built', {
      rowKey: row.rowKey,
      scenario: row.scenario,
      scenarioKey: scenarioRun.scenarioKey,
      userQuery: scenarioRun.userQuery,
      task: scenarioRun.task,
      emailCount: scenarioRun.emails.length,
      attackerEmailPreview: preview(scenarioRun.attackerEmail),
      finalEmailContextPreview: preview(scenarioRun.emails.join('\n\n'), 900),
    });

    const spotlighted = applySpotlighting({
      systemPrompt: `${prompts.email_client}\n\n${prompts.tool_prompt}`,
      userQuery: scenarioRun.userQuery,
      emails: scenarioRun.emails,
    });
    trace(options, 'spotlighting applied', {
      method: spotlighted.spotlightMethod,
      package: 'spotlighting-datamarking',
      packageCall: 'new DataMarkingViaSpotlighting().base64EncodeData(emailContext)',
      systemPromptAddedByPackage: preview(spotlighted.systemPrompt.slice(`${prompts.email_client}\n\n${prompts.tool_prompt}`.length)),
      userQuery: spotlighted.userQuery,
      encodedEmailContextLength: spotlighted.emailContext.length,
      encodedEmailContextPreview: preview(spotlighted.emailContext, 900),
    });

    trace(options, 'calling lm studio', {
      url: `${config.lmstudioBaseUrl}/chat/completions`,
      model: config.lmstudioModel || '(auto-detect)',
    });
    const text = await callLmStudio(config, spotlighted);
    const toolCalls = parseToolCalls(text);
    const objectives = scoreToolCalls(toolCalls, scenarioNumberFromLevel(row.scenario));
    trace(options, 'model response scored', {
      responsePreview: preview(text, 1000),
      toolCalls,
      objectives,
    });

    results.push({
      rowKey: row.rowKey,
      scenario: row.scenario,
      scenarioKey: scenarioRun.scenarioKey,
      subject: row.subject,
      response: text,
      toolCalls,
      objectives,
      originalObjectives: row.objectives,
      spotlightMethod: spotlighted.spotlightMethod,
      dataMarker: spotlighted.dataMarker,
    });
  }

  return results;
}

export function writeRunResults(command, results) {
  const dir = join('runs', new Date().toISOString().replace(/[:.]/g, '-'));
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  writeFileSync(join(dir, `${command}.json`), JSON.stringify(results, null, 2));
  writeFileSync(join(dir, `${command}.jsonl`), results.map((row) => JSON.stringify(row)).join('\n'));
  return dir;
}
