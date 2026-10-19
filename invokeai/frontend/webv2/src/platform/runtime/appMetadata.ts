/** Build-time application metadata with no domain dependencies. */
export const APP_VERSION = import.meta.env.APP_VERSION;

export const DOCS_URL = 'https://v7.invoke.ai/';

/** Release notes for a server version string (`7.0.0`, `7.0.0-rc1`), or the release list when it is unknown. */
export const getReleaseNotesUrl = (version: string | null): string =>
  version
    ? `https://github.com/invoke-ai/InvokeAI/releases/tag/v${version}`
    : 'https://github.com/invoke-ai/InvokeAI/releases';
