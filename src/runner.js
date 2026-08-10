import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  loadSubmissions,
  readJson,
  scenarioNumberFromLevel,
  validateDatasetFiles,
} from './dataset.js';
import { buildScenarioRun } from './scenarios.js';
import { applySpotlighting } from './spotlight.js';
import { callLmStudio, pingLmStudio } from './llmClient.js';
import { parseToolCalls, scoreToolCalls } from './toolParser.js';

export async function setup(config) {
  validateDatasetFiles(config.dataDir);
  const levelsDescriptions = readJson(config.dataDir, 'levels_descriptions.json');
  const scenarios = readJson(config.dataDir, 'scenarios.json');
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
    scenarioCount: Object.keys(scenarios).length,
    phase1LevelCount: Object.keys(levelsDescriptions.phase1 || {}).length,
    systemPromptKeys: Object.keys(systemPrompt),
    spotlightPackage: 'ok',
    lmStudio: ping,
  };
}

export async function runRows(config, options) {
  validateDatasetFiles(config.dataDir);

  const levelsDescriptions = readJson(config.dataDir, 'levels_descriptions.json');
  const scenarios = readJson(config.dataDir, 'scenarios.json');
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

  const results = [];
  for (const row of rows) {
    const scenarioRun = buildScenarioRun(row, scenarios);
    const spotlighted = applySpotlighting({
      systemPrompt: `${prompts.email_client}\n\n${prompts.tool_prompt}`,
      userQuery: scenarioRun.userQuery,
      emails: scenarioRun.emails,
    });
    const text = await callLmStudio(config, spotlighted);
    const toolCalls = parseToolCalls(text);
    const objectives = scoreToolCalls(toolCalls, scenarioNumberFromLevel(row.scenario));

    results.push({
      rowKey: row.rowKey,
      scenario: row.scenario,
      scenarioKey: scenarioRun.scenarioKey,
      subject: row.subject,
      response: text,
      toolCalls,
      objectives,
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
