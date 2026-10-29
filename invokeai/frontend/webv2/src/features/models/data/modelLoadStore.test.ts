import { beforeEach, describe, expect, it } from 'vitest';

import { getModelLoads, modelLoadActivitySink } from './modelLoadStore';

beforeEach(() => {
  modelLoadActivitySink.reset();
});

describe('model load activity', () => {
  it('owns backend payload interpretation and balances matching loads', () => {
    const payload = { config: { base: 'flux', name: 'FLUX Dev', type: 'main' }, submodel_type: 'transformer' };

    modelLoadActivitySink.started(payload);
    modelLoadActivitySink.started({ config: { name: 'VAE' } });

    expect(getModelLoads()).toEqual([{ label: 'FLUX Dev (flux, main, transformer)' }, { label: 'VAE' }]);

    modelLoadActivitySink.completed(payload);
    expect(getModelLoads()).toEqual([{ label: 'VAE' }]);
  });

  it('falls back deterministically for malformed completion payloads', () => {
    modelLoadActivitySink.started(null);
    modelLoadActivitySink.completed({});

    expect(getModelLoads()).toEqual([]);
  });
});
