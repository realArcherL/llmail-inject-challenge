import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildScenarioRun } from '../src/scenarios.js';

describe('scenario context builder', () => {
  it('places scenario 1 attacker email last', () => {
    const run = buildScenarioRun(
      { scenario: 'level1e', subject: 'Attack', body: 'Body' },
      {
        scenario_1: {
          emails: ['benign'],
          position: 'last',
          user_query: 'Summarize',
          task: 'task',
        },
      },
    );

    assert.equal(run.emails.length, 2);
    assert.match(run.emails[1], /Subject of the email: Attack/);
  });

  it('places scenario 2 attacker email in the middle of nine local emails', () => {
    const emails = Array.from({ length: 12 }, (_, index) => `email-${index}`);
    const run = buildScenarioRun(
      { scenario: 'level2e', subject: 'Attack', body: 'Body' },
      {
        scenario_2: {
          emails,
          position: 'mid',
          user_query: 'Summarize',
          task: 'task',
        },
      },
    );

    assert.equal(run.emails.length, 10);
    assert.match(run.emails[4], /Subject of the email: Attack/);
  });
});
