import { DataMarkingViaSpotlighting } from 'spotlighting-datamarking';
import { formatEmailContext } from './scenarios.js';

export function applySpotlighting({ systemPrompt, userQuery, emails }) {
  const marker = new DataMarkingViaSpotlighting();
  const emailContext = formatEmailContext(emails);
  const result = marker.base64EncodeData(emailContext);

  if (!result.markedText || !result.prompt) {
    throw new Error(
      'spotlighting-datamarking returned an invalid base64EncodeData result.',
    );
  }

  return {
    systemPrompt: `${systemPrompt}\n\n${result.prompt}`,
    userQuery,
    emailContext: `Emails:\n${result.markedText}`,
    spotlightMethod: 'base64',
    dataMarker: null,
  };
}
