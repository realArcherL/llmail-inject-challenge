#!/usr/bin/env node
import { asPositiveInt, loadConfig, normalizePhase, parseArgs } from './config.js';
import { runRows, setup, writeRunResults } from './runner.js';

function printUsage() {
  console.log(`Usage:
  npm run setup
  npm run smoke -- --phase phase1 --limit 1
  npm run compare -- --phase phase1 --limit 100
  npm run benchmark -- --phase phase1 --limit 100

Options:
  --phase <phase1|phase2>
  --level <level1e>
  --scenario <scenario_1>
  --limit <n>
  --trace

Default:
  Phase 1 runs use Spotlight levels level1e/f through level4e/f unless --level is set.`);
}

function buildRunOptions(args, defaultLimit) {
  return {
    phase: normalizePhase(args.phase || 'phase1'),
    level: typeof args.level === 'string' ? args.level : undefined,
    scenario: typeof args.scenario === 'string' ? args.scenario : undefined,
    limit: asPositiveInt(args.limit, defaultLimit),
    trace: args.trace === true,
  };
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);
  const config = loadConfig();

  if (!command || command === 'help' || command === '--help') {
    printUsage();
    return;
  }

  if (command === 'setup') {
    const result = await setup(config);
    console.log(JSON.stringify(result, null, 2));
    if (!result.lmStudio.ok) {
      console.log('\nSetup completed without a reachable LLM endpoint. Start LM Studio before smoke/benchmark runs.');
    }
    return;
  }

  if (command === 'smoke') {
    const options = buildRunOptions(args, 1);
    const runOutput = await runRows(config, options);
    const { dir, summary } = writeRunResults('smoke', runOutput, options);
    console.log(JSON.stringify({ outputDir: dir, summary, sampleResult: runOutput.results[0] }, null, 2));
    return;
  }

  if (command === 'benchmark' || command === 'compare') {
    const options = buildRunOptions(args, 100);
    const runOutput = await runRows(config, options);
    const { dir, summary } = writeRunResults(command, runOutput, options);
    console.log(JSON.stringify({ outputDir: dir, summary }, null, 2));
    return;
  }

  throw new Error(`Unknown command "${command}".`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
