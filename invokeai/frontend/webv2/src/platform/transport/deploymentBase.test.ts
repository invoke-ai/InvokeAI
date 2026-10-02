import { describe, expect, it } from 'vitest';

import { deriveDeploymentBaseUrl } from './deploymentBase';

describe('deriveDeploymentBaseUrl', () => {
  const fallback = 'https://app.example';

  it('derives a reverse-proxy prefix from a production asset URL', () => {
    expect(
      deriveDeploymentBaseUrl('https://app.example/invoke/assets/index-abc.js', fallback, 'https://app.example')
    ).toBe('https://app.example/invoke');
  });

  it('keeps root deployments at the page origin', () => {
    expect(deriveDeploymentBaseUrl('https://app.example/assets/index-abc.js', fallback, 'https://app.example')).toBe(
      fallback
    );
  });

  it('uses the final assets segment for nested deployment prefixes', () => {
    expect(
      deriveDeploymentBaseUrl(
        'https://app.example/team/assets/invoke/assets/index-abc.js',
        fallback,
        'https://app.example'
      )
    ).toBe('https://app.example/team/assets/invoke');
  });

  it('keeps development modules at the page origin', () => {
    expect(deriveDeploymentBaseUrl('http://localhost:5174/src/main.tsx', 'http://localhost:5174')).toBe(
      'http://localhost:5174'
    );
  });

  it('falls back to the app origin for foreign CDN assets', () => {
    expect(deriveDeploymentBaseUrl('https://cdn.example/invoke/assets/index.js', fallback, 'https://app.example')).toBe(
      fallback
    );
  });

  it('falls back for an invalid module URL', () => {
    expect(deriveDeploymentBaseUrl('not a url', fallback)).toBe(fallback);
  });
});
