import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./deploymentBase', () => ({
  getDeploymentBasePath: () => '/invoke',
  getDeploymentBaseUrl: () => 'https://app.example/invoke',
}));

import { absolutizeApiUrl, buildApiUrl, getBackendSocketPath, getBackendSocketUrl } from './http';

describe('deployment-aware backend URLs', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { location: { origin: 'https://app.example' } });
  });

  it('prefixes API requests with the deployment root', () => {
    expect(buildApiUrl('/api/v1/app/version')).toBe('https://app.example/invoke/api/v1/app/version');
  });

  it('prefixes backend-relative resource URLs and preserves absolute URLs', () => {
    expect(absolutizeApiUrl('/api/v1/images/i/image.png/full')).toBe(
      'https://app.example/invoke/api/v1/images/i/image.png/full'
    );
    expect(absolutizeApiUrl('https://cdn.example/image.png')).toBe('https://cdn.example/image.png');
  });

  it('uses the page origin with a deployment-prefixed Socket.IO path', () => {
    expect(getBackendSocketUrl()).toBe('https://app.example');
    expect(getBackendSocketPath()).toBe('/invoke/ws/socket.io');
  });
});
