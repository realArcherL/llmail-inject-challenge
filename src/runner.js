import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  loadSubmissions,
  PHASE1_SPOTLIGHT_LEVELS,
  readJson,
  scenarioNumberFromLevel,
  validateDatasetFiles,
} from './dataset.js';
import { buildDatasetReplayRun, formatEmailContext } from './scenarios.js';
import { applySpotlighting, withoutSpotlighting } from './spotlight.js';
import { callLmStudioDetailed, pingLmStudio } from './llmClient.js';
import { mergeToolCalls, parseNativeToolCalls, parseToolCalls, scoreToolCalls } from './toolParser.js';

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

function summarizeVariants(results) {
  const variants = ['unsafeBaseline', 'baseline', 'defended'];
  const byVariant = {};

  for (const variant of variants) {
    const rows = results
      .map((row) => row.variants?.[variant])
      .filter(Boolean)
      .map((variantResult) => ({
        scenario: variantResult.scenario,
        objectives: variantResult.objectives,
        originalObjectives: variantResult.originalObjectives,
      }));
    byVariant[variant] = summarizeBucket(rows);
  }

  const pairedTotal = results.filter(
    (row) => row.variants?.unsafeBaseline && row.variants?.baseline && row.variants?.defended,
  ).length;
  const unsafeSuccessBaselineFailure = results.filter(
    (row) =>
      isFullSuccess(row.variants?.unsafeBaseline?.objectives) &&
      !isFullSuccess(row.variants?.baseline?.objectives),
  ).length;
  const unsafeSuccessDefendedFailure = results.filter(
    (row) =>
      isFullSuccess(row.variants?.unsafeBaseline?.objectives) &&
      !isFullSuccess(row.variants?.defended?.objectives),
  ).length;
  const baselineSuccessDefendedFailure = results.filter(
    (row) =>
      isFullSuccess(row.variants?.baseline?.objectives) &&
      !isFullSuccess(row.variants?.defended?.objectives),
  ).length;
  const unsafeSuccessBaselineSuccessDefendedFailure = results.filter(
    (row) =>
      isFullSuccess(row.variants?.unsafeBaseline?.objectives) &&
      isFullSuccess(row.variants?.baseline?.objectives) &&
      !isFullSuccess(row.variants?.defended?.objectives),
  ).length;
  const bothSuccess = results.filter(
    (row) =>
      isFullSuccess(row.variants?.baseline?.objectives) &&
      isFullSuccess(row.variants?.defended?.objectives),
  ).length;
  const bothFailure = results.filter(
    (row) =>
      !isFullSuccess(row.variants?.baseline?.objectives) &&
      !isFullSuccess(row.variants?.defended?.objectives),
  ).length;
  const baselineFailureDefendedSuccess = results.filter(
    (row) =>
      !isFullSuccess(row.variants?.baseline?.objectives) &&
      isFullSuccess(row.variants?.defended?.objectives),
  ).length;

  return {
    ...byVariant,
    comparison: {
      pairedTotal,
      unsafeSuccessBaselineFailure,
      unsafeSuccessDefendedFailure,
      baselineSuccessDefendedFailure,
      unsafeSuccessBaselineSuccessDefendedFailure,
      bothSuccess,
      bothFailure,
      baselineFailureDefendedSuccess,
      llmailPromptBlockedAttackRate: rate(unsafeSuccessBaselineFailure, pairedTotal),
      spotlightBlockedAttackRate: rate(baselineSuccessDefendedFailure, pairedTotal),
      endToEndBlockedAttackRate: rate(unsafeSuccessDefendedFailure, pairedTotal),
    },
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
    overall: summarizeVariants(results),
    byScenario: Object.fromEntries(
      Object.entries(byScenario).map(([scenario, rows]) => [
        scenario,
        summarizeVariants(rows),
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
      levels: options.level ? null : options.levels || null,
      scenario: options.scenario || null,
    },
    lmStudio: {
      baseUrl: config.lmstudioBaseUrl,
      configuredModel: config.lmstudioModel || null,
    },
    prompts: {
      source: 'data/data/system_prompt.json',
      systemPrompt: 'email_client',
      toolPrompt: 'tool_prompt',
      userQuery: promptSnapshot.userQuery,
      unsafeBaseline: {
        systemPrompt: 'tool_prompt only',
      },
      baseline: {
        systemPrompt: 'email_client + tool_prompt',
      },
      defended: {
        systemPrompt: 'email_client + tool_prompt + spotlight package prompt',
      },
    },
    spotlight: {
      package: 'spotlighting-datamarking',
      method: 'base64EncodeData',
      call: 'new DataMarkingViaSpotlighting().base64EncodeData(emailContext)',
      packagePromptPreview: preview(promptSnapshot.spotlightPackagePrompt, 260),
    },
  };
}

async function runVariant({ config, row, scenarioRun, prompt, emailContextBeforeDefense, variant }) {
  const llmResult = await callLmStudioDetailed(config, prompt, variant);
  const text = llmResult.response;
  const textToolCalls = parseToolCalls(text);
  const nativeToolCalls = parseNativeToolCalls(llmResult.rawResponse);
  const toolCalls = mergeToolCalls(nativeToolCalls, textToolCalls);
  const objectives = scoreToolCalls(toolCalls, scenarioNumberFromLevel(row.scenario));

  return {
    variant,
    rowKey: row.rowKey,
    scenario: row.scenario,
    scenarioKey: scenarioRun.scenarioKey,
    subject: row.subject,
    response: text,
    toolCalls,
    objectives,
    originalObjectives: row.objectives,
    spotlightMethod: prompt.spotlightMethod,
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
        package: variant === 'defended' ? 'spotlighting-datamarking' : null,
        packageCall:
          variant === 'defended'
            ? 'new DataMarkingViaSpotlighting().base64EncodeData(emailContext)'
            : null,
        spotlightMethod: prompt.spotlightMethod,
        emailContext: prompt.emailContext,
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
        nativeToolCalls,
        textToolCalls,
        parsedToolCalls: toolCalls,
        scoredObjectives: objectives,
      },
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
  validateDatasetFiles(config.dataDir, options.phase);
  trace(options, 'config', {
    dataDir: config.dataDir,
    lmstudioBaseUrl: config.lmstudioBaseUrl,
    lmstudioModel: config.lmstudioModel || '(auto-detect from /v1/models)',
    phase: options.phase,
    level: options.level || '(any level)',
    levels: options.level ? null : options.levels || '(any level set)',
    scenario: options.scenario || '(any scenario)',
    limit: options.limit,
    mode: 'dataset-replay',
  });

  const prompts = readJson(config.dataDir, 'system_prompt.json');
  const levels = options.level
    ? undefined
    : options.levels || (options.phase === 'phase1' ? PHASE1_SPOTLIGHT_LEVELS : undefined);
  let promptSnapshot = null;
  const rows = await loadSubmissions({
    dataDir: config.dataDir,
    phase: options.phase,
    level: options.level,
    levels,
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
    const unsafeSystemPrompt = prompts.tool_prompt;
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
    const baselinePrompt = withoutSpotlighting({
      systemPrompt: originalSystemPrompt,
      userQuery: scenarioRun.userQuery,
      emails: scenarioRun.emails,
    });
    const unsafeBaselinePrompt = withoutSpotlighting({
      systemPrompt: unsafeSystemPrompt,
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
      variants: ['unsafeBaseline', 'baseline', 'defended'],
    });
    const unsafeBaseline = await runVariant({
      config,
      row,
      scenarioRun,
      prompt: unsafeBaselinePrompt,
      emailContextBeforeDefense,
      variant: 'unsafeBaseline',
    });
    const baseline = await runVariant({
      config,
      row,
      scenarioRun,
      prompt: baselinePrompt,
      emailContextBeforeDefense,
      variant: 'baseline',
    });
    const defended = await runVariant({
      config,
      row,
      scenarioRun,
      prompt: spotlighted,
      emailContextBeforeDefense,
      variant: 'defended',
    });
    trace(options, 'model responses scored', {
      unsafeBaseline: {
        responsePreview: preview(unsafeBaseline.response, 1000),
        toolCalls: unsafeBaseline.toolCalls,
        objectives: unsafeBaseline.objectives,
      },
      baseline: {
        responsePreview: preview(baseline.response, 1000),
        toolCalls: baseline.toolCalls,
        objectives: baseline.objectives,
      },
      defended: {
        responsePreview: preview(defended.response, 1000),
        toolCalls: defended.toolCalls,
        objectives: defended.objectives,
      },
    });

    results.push({
      rowKey: row.rowKey,
      scenario: row.scenario,
      scenarioKey: scenarioRun.scenarioKey,
      subject: row.subject,
      originalObjectives: row.objectives,
      variants: {
        unsafeBaseline,
        baseline,
        defended,
      },
    });
  }

  return {
    results,
    metadata: buildRunMetadata(config, prompts, { ...options, levels }, promptSnapshot),
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
