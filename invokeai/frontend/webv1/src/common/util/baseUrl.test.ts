import { afterEach, describe, expect, it, vi } from 'vitest';

import { deriveDeploymentBaseUrl } from './baseUrl';

describe('deriveDeploymentBaseUrl', () => {
  const fallback = 'http://fallback.example';

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns origin + sub-path prefix when served from a sub-path', () => {
    expect(deriveDeploymentBaseUrl('https://example.com/invoke/assets/index-abc.js', fallback)).toBe(
      'https://example.com/invoke'
    );
  });

  it('returns just the origin when served from the domain root', () => {
    expect(deriveDeploymentBaseUrl('https://example.com/assets/index-abc.js', fallback)).toBe('https://example.com');
  });

  it('handles nested sub-paths', () => {
    expect(deriveDeploymentBaseUrl('https://example.com/a/b/c/assets/index-abc.js', fallback)).toBe(
      'https://example.com/a/b/c'
    );
  });

  it('preserves a non-default port', () => {
    expect(deriveDeploymentBaseUrl('http://localhost:8080/invoke/assets/index-abc.js', fallback)).toBe(
      'http://localhost:8080/invoke'
    );
  });

  it('returns origin (no prefix) in dev mode where there is no /assets/ segment', () => {
    expect(deriveDeploymentBaseUrl('http://localhost:5173/src/main.tsx', fallback)).toBe('http://localhost:5173');
  });

  it('falls back to the provided origin for an unparseable module url', () => {
    expect(deriveDeploymentBaseUrl('not-a-url', fallback)).toBe(fallback);
  });

  it('uses lastIndexOf so a prefix containing an /assets/ segment resolves to the deploy root', () => {
    expect(deriveDeploymentBaseUrl('https://example.com/foo/assets/bar/assets/index-abc.js', fallback)).toBe(
      'https://example.com/foo/assets/bar'
    );
  });

  it('falls back to the app origin when the bundle is served from a foreign (CDN) origin', () => {
    vi.stubGlobal('window', { location: { origin: 'https://app.example' } });
    // Bundle on a CDN origin -> sub-path detection there is meaningless; must target the app origin.
    expect(deriveDeploymentBaseUrl('https://cdn.example/invoke/assets/index-abc.js', fallback)).toBe(fallback);
  });

  it('still detects the sub-path when the bundle origin matches the app origin', () => {
    vi.stubGlobal('window', { location: { origin: 'https://example.com' } });
    expect(deriveDeploymentBaseUrl('https://example.com/invoke/assets/index-abc.js', fallback)).toBe(
      'https://example.com/invoke'
    );
  });
});
