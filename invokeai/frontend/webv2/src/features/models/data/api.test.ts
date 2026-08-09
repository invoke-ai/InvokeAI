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

// relationshipsApi bypasses the port and uses the platform client directly.
vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...((await importOriginal()) as Record<string, unknown>),
  apiFetch: mocks.request,
  apiFetchJson: mocks.apiFetchJson,
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
    const controller = new AbortController();

    await installModel(
      {
        accessToken: 'secret-token',
        config: { name: 'Model' },
        source: 'https://example.test/model',
      },
      controller.signal
    );

    const [url, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('/api/v2/models/install?source=https%3A%2F%2Fexample.test%2Fmodel');
    expect(url).not.toContain('secret-token');
    expect(init.headers).toEqual({ 'X-Model-Source-Access-Token': 'secret-token' });
    expect(JSON.parse(init.body as string)).toEqual({ name: 'Model' });
    expect(init.signal).toBe(controller.signal);
  });
});

describe('account-owned model requests', () => {
  beforeEach(() => {
    mocks.request.mockReset().mockResolvedValue(new Response());
    mocks.apiFetchJson.mockReset().mockResolvedValue([]);
  });

  it('forwards one account signal through lookup, credentials, relationships, maintenance, and image requests', async () => {
    const {
      deleteOrphanedModels,
      getExternalProviderConfigs,
      getHuggingFaceModels,
      pruneCompletedModelInstalls,
      updateModelImage,
    } = await import('./api');
    const { addModelRelationship } = await import('./relationshipsApi');
    const controller = new AbortController();
    const image = new Blob(['image'], { type: 'image/png' });

    await getHuggingFaceModels('owner/repo', controller.signal);
    await getExternalProviderConfigs(controller.signal);
    await deleteOrphanedModels(['/models/orphan'], controller.signal);
    await addModelRelationship('model-a', 'model-b', controller.signal);
    await pruneCompletedModelInstalls(controller.signal);
    await updateModelImage('model-a', image, controller.signal);

    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(
      1,
      '/api/v2/models/hugging_face?hugging_face_repo=owner%2Frepo',
      { signal: controller.signal }
    );
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/app/external_providers/config', {
      signal: controller.signal,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(3, '/api/v2/models/sync/orphaned', {
      body: JSON.stringify({ paths: ['/models/orphan'] }),
      headers: { 'Content-Type': 'application/json' },
      method: 'DELETE',
      signal: controller.signal,
    });
    expect(mocks.request).toHaveBeenNthCalledWith(1, '/api/v1/model_relationships/', {
      body: JSON.stringify({ model_key_1: 'model-a', model_key_2: 'model-b' }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      signal: controller.signal,
    });
    expect(mocks.request).toHaveBeenNthCalledWith(2, '/api/v2/models/install', {
      method: 'DELETE',
      signal: controller.signal,
    });
    expect(mocks.request).toHaveBeenNthCalledWith(
      3,
      '/api/v2/models/i/model-a/image',
      expect.objectContaining({ method: 'PATCH', signal: controller.signal })
    );
  });
});
