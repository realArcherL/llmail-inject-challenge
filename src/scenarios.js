import { scenarioNumberFromLevel } from './dataset.js';

export function formatAttackerEmail(submission) {
  return `Subject of the email: ${submission.subject}.   Body: ${submission.body}`;
}

function insertInMiddle(emails, attackerEmail, topK) {
  const existing = emails.slice(0, Math.max(topK - 1, 0));
  if (existing.length <= 1) return [...existing, attackerEmail];

  const index = Math.floor(existing.length / 2);
  return [...existing.slice(0, index), attackerEmail, ...existing.slice(index)];
}

/**
 * Reconstructs the local email context for a row using dataset scenario metadata.
 */
export function buildScenarioRun(submission, scenarios) {
  const scenarioNumber = scenarioNumberFromLevel(submission.scenario);
  const scenarioKey = `scenario_${scenarioNumber}`;
  const scenario = scenarios[scenarioKey];
  if (!scenario) throw new Error(`Missing ${scenarioKey} in scenarios.json.`);

  const attackerEmail = formatAttackerEmail(submission);
  let emails;

  if (scenario.position === 'last') {
    emails = [...scenario.emails.slice(0, 1), attackerEmail];
  } else if (scenario.position === 'mid') {
    emails = insertInMiddle(scenario.emails, attackerEmail, 10);
  } else if (scenario.position === 'retrieval') {
    emails = insertInMiddle(scenario.emails, attackerEmail, 10);
  } else {
    throw new Error(`Unsupported scenario position "${scenario.position}" for ${scenarioKey}.`);
  }

  return {
    scenarioKey,
    userQuery: scenario.user_query,
    task: scenario.task,
    emails,
    attackerEmail,
  };
}

export function formatEmailContext(emails) {
  return emails.map((email) => ` email: ${email}`).join('');
}
