import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModels: vi.fn(),
}));

vi.mock('./api', () => api);

describe('models store loading', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getModelsDir.mockReset().mockResolvedValue('/models');
    api.listMissingModels.mockReset().mockResolvedValue([]);
    api.listModels.mockReset();
  });

  it('returns its shared request and retries after an initial failure', async () => {
    api.listModels.mockRejectedValueOnce(new Error('temporary outage')).mockResolvedValueOnce([
      {
        base: 'sdxl',
        description: null,
        file_size: 1,
        format: 'checkpoint',
        hash: 'hash',
        key: 'model-key',
        name: 'Model',
        source: 'model.safetensors',
        type: 'main',
      },
    ]);
    const { ensureModelsLoaded, getModelsSnapshot } = await import('./modelsStore');

    const first = ensureModelsLoaded();
    expect(ensureModelsLoaded()).toBe(first);
    await first;
    expect(getModelsSnapshot()).toMatchObject({ error: 'temporary outage', status: 'error' });

    await ensureModelsLoaded();
    expect(getModelsSnapshot()).toMatchObject({ error: null, status: 'loaded' });
    expect(getModelsSnapshot().models.map((model) => model.key)).toEqual(['model-key']);
    expect(api.listModels).toHaveBeenCalledTimes(2);
  });
});
