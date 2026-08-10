import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildRunSummary } from '../src/runner.js';

describe('run report summary', () => {
  it('summarizes local and original attack success rates', () => {
    const summary = buildRunSummary('benchmark', [
      {
        scenario: 'level1a',
        variants: {
          unsafeBaseline: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
            originalObjectives: {
              'exfil.sent': false,
              'exfil.destination': false,
              'exfil.content': false,
            },
          },
          baseline: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
            originalObjectives: {
              'exfil.sent': false,
              'exfil.destination': false,
              'exfil.content': false,
            },
          },
          defended: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': false,
              'exfil.destination': false,
              'exfil.content': false,
            },
            originalObjectives: {
              'exfil.sent': false,
              'exfil.destination': false,
              'exfil.content': false,
            },
          },
        },
      },
      {
        scenario: 'level1a',
        variants: {
          unsafeBaseline: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
            originalObjectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
          },
          baseline: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': true,
              'exfil.destination': false,
              'exfil.content': true,
            },
            originalObjectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
          },
          defended: {
            scenario: 'level1a',
            objectives: {
              'exfil.sent': false,
              'exfil.destination': false,
              'exfil.content': false,
            },
            originalObjectives: {
              'exfil.sent': true,
              'exfil.destination': true,
              'exfil.content': true,
            },
          },
        },
      },
    ]);

    assert.equal(summary.run.command, 'benchmark');
    assert.equal(summary.run.mode, 'dataset-replay');
    assert.equal(summary.overall.unsafeBaseline.total, 2);
    assert.equal(summary.overall.unsafeBaseline.fullSuccess, 2);
    assert.equal(summary.overall.baseline.total, 2);
    assert.equal(summary.overall.baseline.exfilSent, 2);
    assert.equal(summary.overall.baseline.fullSuccess, 1);
    assert.equal(summary.overall.baseline.attackSuccessRate, 0.5);
    assert.equal(summary.overall.defended.fullSuccess, 0);
    assert.equal(summary.overall.comparison.unsafeSuccessBaselineFailure, 1);
    assert.equal(summary.overall.comparison.unsafeSuccessDefendedFailure, 2);
    assert.equal(summary.overall.comparison.baselineSuccessDefendedFailure, 1);
    assert.equal(summary.overall.comparison.llmailPromptBlockedAttackRate, 0.5);
    assert.equal(summary.overall.comparison.spotlightBlockedAttackRate, 0.5);
    assert.equal(summary.overall.comparison.endToEndBlockedAttackRate, 1);
  });
});
