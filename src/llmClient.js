function trimTrailingSlash(value) {
  return value.replace(/\/+$/, '');
}

export function endpointConfigured(config) {
  return Boolean(config.lmstudioBaseUrl);
}

async function listModels(config) {
  if (!endpointConfigured(config)) {
    throw new Error('LMSTUDIO_BASE_URL is not configured.');
  }

  const response = await fetch(`${trimTrailingSlash(config.lmstudioBaseUrl)}/models`, {
    headers: { Authorization: `Bearer ${config.lmstudioApiKey}` },
  });

  if (!response.ok) {
    throw new Error(`LM Studio /models returned ${response.status} ${response.statusText}.`);
  }

  const json = await response.json();
  return Array.isArray(json.data) ? json.data : [];
}

export async function resolveLmStudioModel(config) {
  if (config.lmstudioModel) return config.lmstudioModel;

  const models = await listModels(config);
  const model = models.find((item) => typeof item?.id === 'string')?.id;
  if (!model) {
    throw new Error('LM Studio did not return any model ids from /models.');
  }
  return model;
}

export async function pingLmStudio(config) {
  if (!endpointConfigured(config)) {
    return {
      ok: false,
      skipped: true,
      message: 'LMSTUDIO_BASE_URL is not configured.',
    };
  }

  const model = await resolveLmStudioModel(config);
  return {
    ok: true,
    skipped: false,
    model,
    message: `LM Studio endpoint is reachable. Using model "${model}".`,
  };
}

export function buildLmStudioMessages(prompt) {
  return [
    { role: 'system', content: prompt.systemPrompt },
    { role: 'user', content: `${prompt.userQuery}\n\n${prompt.emailContext}` },
  ];
}

export function buildLmStudioMessageAudit(prompt, variant = 'defended') {
  const promptPrefix =
    variant === 'unsafeBaseline'
      ? 'unsafeBaseline'
      : variant === 'baseline'
        ? 'baseline'
        : 'defended';
  const emailContextRef =
    variant === 'defended'
      ? 'audit.inputAfterDefense.emailContext'
      : 'audit.inputBeforeDefense.emailContext';

  return [
    {
      role: 'system',
      contentRef: `summary.run.prompts.${promptPrefix}.systemPrompt`,
    },
    {
      role: 'user',
      contentParts: {
        userQueryRef: 'summary.run.prompts.userQuery',
        emailContextRef,
      },
    },
  ];
}

export async function callLmStudioDetailed(config, prompt, variant = 'defended') {
  if (!endpointConfigured(config)) {
    throw new Error('LMSTUDIO_BASE_URL is required for smoke/benchmark runs.');
  }

  const model = await resolveLmStudioModel(config);
  const messages = buildLmStudioMessages(prompt);
  const response = await fetch(`${trimTrailingSlash(config.lmstudioBaseUrl)}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.lmstudioApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LM Studio chat completion failed: ${response.status} ${response.statusText}\n${body}`);
  }

  const json = await response.json();
  return {
    model,
    messages,
    messageAudit: buildLmStudioMessageAudit(prompt, variant),
    response: json.choices?.[0]?.message?.content || '',
    rawResponse: json,
  };
}

export async function callLmStudio(config, prompt) {
  const result = await callLmStudioDetailed(config, prompt);
  return result.response;
}
