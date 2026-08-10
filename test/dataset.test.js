import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  normalizeSubmission,
  PHASE1_SPOTLIGHT_LEVELS,
  scenarioNumberFromLevel,
} from '../src/dataset.js';

describe('dataset helpers', () => {
  it('normalizes JSON-string objectives', () => {
    const row = normalizeSubmission({
      RowKey: 'row-1',
      job_id: 'job-1',
      team_id: 'team-1',
      scenario: 'level1e',
      subject: 'Hello',
      body: 'World',
      objectives: '{"email.retrieved":true}',
    });

    assert.equal(row.rowKey, 'row-1');
    assert.equal(row.objectives['email.retrieved'], true);
  });

  it('maps level names to scenario numbers', () => {
    assert.equal(scenarioNumberFromLevel('level4f'), 4);
  });

  it('defines the Phase 1 Spotlight-only levels', () => {
    assert.deepEqual(PHASE1_SPOTLIGHT_LEVELS, [
      'level1e',
      'level1f',
      'level2e',
      'level2f',
      'level3e',
      'level3f',
      'level4e',
      'level4f',
    ]);
  });
});
