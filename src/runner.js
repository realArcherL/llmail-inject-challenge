import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  loadSubmissions,
  readJson,
  scenarioNumberFromLevel,
  validateDatasetFiles,
} from './dataset.js';
import { buildDatasetReplayRun, formatEmailContext } from './scenarios.js';
import { applySpotlighting } from './spotlight.js';
import { callLmStudioDetailed, pingLmStudio } from './llmClient.js';
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

function rate(count, total) {
  return total === 0 ? 0 : Number((count / total).toFixed(4));
}

function isFullSuccess(objectives = {}) {
  return Boolean(
    objectives['exfil.sent'] &&
      objectives['exfil.destination'] &&
      objectives['exfil.content'],
  );
}

function summarizeBucket(rows) {
  const total = rows.length;
  const exfilSent = rows.filter((row) => row.objectives['exfil.sent']).length;
  const exfilDestination = rows.filter((row) => row.objectives['exfil.destination']).length;
  const exfilContent = rows.filter((row) => row.objectives['exfil.content']).length;
  const fullSuccess = rows.filter((row) => isFullSuccess(row.objectives)).length;
  const originalFullSuccess = rows.filter((row) => isFullSuccess(row.originalObjectives)).length;

  return {
    total,
    exfilSent,
    exfilDestination,
    exfilContent,
    fullSuccess,
    attackSuccessRate: rate(fullSuccess, total),
    microsoftOriginalFullSuccess: originalFullSuccess,
    microsoftOriginalAttackSuccessRate: rate(originalFullSuccess, total),
  };
}

export function buildRunSummary(command, results, metadata = {}) {
  const byScenario = {};
  for (const row of results) {
    if (!byScenario[row.scenario]) byScenario[row.scenario] = [];
    byScenario[row.scenario].push(row);
  }

  return {
    run: {
      command,
      generatedAt: new Date().toISOString(),
      mode: 'dataset-replay',
      spotlightMethod: 'base64',
      ...metadata,
    },
    overall: summarizeBucket(results),
    byScenario: Object.fromEntries(
      Object.entries(byScenario).map(([scenario, rows]) => [
        scenario,
        summarizeBucket(rows),
      ]),
    ),
  };
}

function buildRunMetadata(config, prompts, options, promptSnapshot) {
  return {
    dataDir: config.dataDir,
    phase: options.phase,
    limit: options.limit,
    filters: {
      level: options.level || null,
      scenario: options.scenario || null,
    },
    lmStudio: {
      baseUrl: config.lmstudioBaseUrl,
      configuredModel: config.lmstudioModel || null,
    },
    promptRefs: {
      source: 'data/data/system_prompt.json',
      systemPrompt: 'email_client',
      toolPrompt: 'tool_prompt',
      userQuery: promptSnapshot.userQuery,
    },
    spotlight: {
      package: 'spotlighting-datamarking',
      method: 'base64EncodeData',
      call: 'new DataMarkingViaSpotlighting().base64EncodeData(emailContext)',
      packagePromptPreview: preview(promptSnapshot.spotlightPackagePrompt, 260),
    },
  };
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
  });

  const prompts = readJson(config.dataDir, 'system_prompt.json');
  let promptSnapshot = null;
  const rows = await loadSubmissions({
    dataDir: config.dataDir,
    phase: options.phase,
    level: options.level,
    scenario: options.scenario,
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
    const originalSystemPrompt = `${prompts.email_client}\n\n${prompts.tool_prompt}`;
    const emailContextBeforeDefense = formatEmailContext(scenarioRun.emails);
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
      systemPrompt: originalSystemPrompt,
      userQuery: scenarioRun.userQuery,
      emails: scenarioRun.emails,
    });
    if (!promptSnapshot) {
      promptSnapshot = {
        userQuery: scenarioRun.userQuery,
        spotlightPackagePrompt: spotlighted.systemPrompt.slice(originalSystemPrompt.length).trim(),
        afterDefenseSystemPrompt: spotlighted.systemPrompt,
      };
    }
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
    const llmResult = await callLmStudioDetailed(config, spotlighted);
    const text = llmResult.response;
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
      audit: {
        inputBeforeDefense: {
          source: 'data/data/raw_submissions_phase*.jsonl',
          row: {
            rowKey: row.rowKey,
            jobId: row.jobId,
            teamId: row.teamId,
            scenario: row.scenario,
            subject: row.subject,
            body: row.body,
            originalObjectives: row.objectives,
          },
          emails: scenarioRun.emails,
          emailContext: emailContextBeforeDefense,
        },
        inputAfterDefense: {
          package: 'spotlighting-datamarking',
          packageCall: 'new DataMarkingViaSpotlighting().base64EncodeData(emailContext)',
          spotlightMethod: spotlighted.spotlightMethod,
          emailContext: spotlighted.emailContext,
        },
        llmRequest: {
          url: `${config.lmstudioBaseUrl}/chat/completions`,
          model: llmResult.model,
          temperature: 0,
          messages: llmResult.messageAudit,
        },
        llmResponse: {
          text,
          rawResponse: llmResult.rawResponse,
          parsedToolCalls: toolCalls,
          scoredObjectives: objectives,
        },
      },
    });
  }

  return {
    results,
    metadata: buildRunMetadata(config, prompts, options, promptSnapshot),
  };
}

export function writeRunResults(command, runOutput, metadata = {}) {
  const results = Array.isArray(runOutput) ? runOutput : runOutput.results;
  const runMetadata = Array.isArray(runOutput) ? metadata : runOutput.metadata;
  const dir = join('runs', new Date().toISOString().replace(/[:.]/g, '-'));
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const summary = buildRunSummary(command, results, { ...runMetadata, ...metadata });
  writeFileSync(join(dir, 'summary.json'), JSON.stringify(summary, null, 2));
  writeFileSync(join(dir, 'results.jsonl'), results.map((row) => JSON.stringify(row)).join('\n'));
  return { dir, summary };
}
