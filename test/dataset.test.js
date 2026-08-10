import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  isSpotlightLevel,
  normalizeSubmission,
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

  it('identifies phase 1 spotlight levels', () => {
    assert.equal(isSpotlightLevel('level1e', 'phase1', { phase1: {} }), true);
    assert.equal(isSpotlightLevel('level1a', 'phase1', { phase1: {} }), false);
  });
});
