import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ModelsDataPort } from './transport';

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('./transport', () => ({
  browserModelsDataPort: {
    buildUrl: (path: string) => path,
    request: mocks.request,
    requestJson: mocks.apiFetchJson,
  } satisfies ModelsDataPort,
}));

describe('installModel', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({
      id: 1,
      source: 'https://example.test/model.safetensors',
      status: 'waiting',
    });
  });

  it('sends remote source credentials in a header instead of the URL', async () => {
    const { installModel } = await import('./api');

    await installModel({
      accessToken: 'secret-token',
      config: { name: 'Model' },
      source: 'https://example.test/model',
    });

    const [url, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('/api/v2/models/install?source=https%3A%2F%2Fexample.test%2Fmodel');
    expect(url).not.toContain('secret-token');
    expect(init.headers).toEqual({ 'X-Model-Source-Access-Token': 'secret-token' });
    expect(JSON.parse(init.body as string)).toEqual({ name: 'Model' });
  });
});
