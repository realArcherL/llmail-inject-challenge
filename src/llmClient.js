function trimTrailingSlash(value) {
  return value.replace(/\/+$/, '');
}

export function endpointConfigured(config) {
  return Boolean(config.lmstudioBaseUrl && config.lmstudioModel);
}

export async function pingLmStudio(config) {
  if (!endpointConfigured(config)) {
    return {
      ok: false,
      skipped: true,
      message: 'LMSTUDIO_BASE_URL or LMSTUDIO_MODEL is not configured.',
    };
  }

  const response = await fetch(`${trimTrailingSlash(config.lmstudioBaseUrl)}/models`, {
    headers: { Authorization: `Bearer ${config.lmstudioApiKey}` },
  });

  if (!response.ok) {
    return {
      ok: false,
      skipped: false,
      message: `LM Studio /models returned ${response.status} ${response.statusText}.`,
    };
  }

  return { ok: true, skipped: false, message: 'LM Studio endpoint is reachable.' };
}

export async function callLmStudio(config, prompt) {
  if (!endpointConfigured(config)) {
    throw new Error('LMSTUDIO_BASE_URL and LMSTUDIO_MODEL are required for smoke/benchmark runs.');
  }

  const response = await fetch(`${trimTrailingSlash(config.lmstudioBaseUrl)}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.lmstudioApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: config.lmstudioModel,
      temperature: 0,
      messages: [
        { role: 'system', content: prompt.systemPrompt },
        { role: 'user', content: `${prompt.userQuery}\n\n${prompt.emailContext}` },
      ],
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LM Studio chat completion failed: ${response.status} ${response.statusText}\n${body}`);
  }

  const json = await response.json();
  return json.choices?.[0]?.message?.content || '';
}
