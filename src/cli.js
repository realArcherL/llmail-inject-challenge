#!/usr/bin/env node
import { asPositiveInt, loadConfig, normalizePhase, parseArgs } from './config.js';
import { runRows, setup, writeRunResults } from './runner.js';

function printUsage() {
  console.log(`Usage:
  npm run setup
  npm run smoke -- --phase phase1 --limit 1
  npm run benchmark -- --phase phase1 --limit 100

Options:
  --phase <phase1|phase2>
  --level <level1e>
  --scenario <scenario_1>
  --limit <n>
  --trace
  --microsoft-spotlight-levels`);
}

function buildRunOptions(args, defaultLimit) {
  return {
    phase: normalizePhase(args.phase || 'phase1'),
    level: typeof args.level === 'string' ? args.level : undefined,
    scenario: typeof args.scenario === 'string' ? args.scenario : undefined,
    limit: asPositiveInt(args.limit, defaultLimit),
    spotlightOnly: args['microsoft-spotlight-levels'] === true,
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
    const results = await runRows(config, buildRunOptions(args, 1));
    const dir = writeRunResults('smoke', results);
    console.log(JSON.stringify({ outputDir: dir, results }, null, 2));
    return;
  }

  if (command === 'benchmark') {
    const results = await runRows(config, buildRunOptions(args, 100));
    const dir = writeRunResults('benchmark', results);
    const summary = {
      outputDir: dir,
      total: results.length,
      exfilSent: results.filter((row) => row.objectives['exfil.sent']).length,
      exfilDestination: results.filter((row) => row.objectives['exfil.destination']).length,
      exfilContent: results.filter((row) => row.objectives['exfil.content']).length,
    };
    console.log(JSON.stringify(summary, null, 2));
    return;
  }

  throw new Error(`Unknown command "${command}".`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
