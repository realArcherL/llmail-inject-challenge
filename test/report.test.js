import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildRunSummary } from '../src/runner.js';

describe('run report summary', () => {
  it('summarizes local and original attack success rates', () => {
    const summary = buildRunSummary('benchmark', [
      {
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
      {
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
    ]);

    assert.equal(summary.overall.total, 2);
    assert.equal(summary.overall.exfilSent, 2);
    assert.equal(summary.overall.fullSuccess, 1);
    assert.equal(summary.overall.attackSuccessRate, 0.5);
    assert.equal(summary.overall.microsoftOriginalFullSuccess, 1);
    assert.equal(summary.overall.microsoftOriginalAttackSuccessRate, 0.5);
  });
});
