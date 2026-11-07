import type { ModelConfig } from '@features/models/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, expect, it, vi } from 'vitest';

import { deleteModel, updateModel } from './api';
import { getModelsSnapshot, refreshModels, removeModelsFromStore, replaceModelInStore } from './modelsStore';

afterEach(() => {
  vi.unstubAllGlobals();
  accountLifecycle.invalidate();
});

it.each(['rename', 'delete'] as const)(
  'preserves a confirmed %s across deferred HTTP list responses',
  async (mutation) => {
    accountLifecycle.activate(`models-browser-${mutation}`);
    const original = { key: 'model-a', name: 'Before', base: 'sdxl', type: 'main' } as ModelConfig;
    let serverModels = [original];
    let finishOld!: () => void;
    let finishTrailing!: () => void;
    let listReads = 0;
    const mutationRequests: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = new URL(String(input), location.href);
        if (url.pathname.endsWith('/models/missing')) {
          return Promise.resolve(Response.json({ models: [] }));
        }
        if (url.pathname.endsWith('/models/models_dir')) {
          return Promise.resolve(Response.json('/models'));
        }
        if (url.pathname.endsWith('/models/')) {
          listReads += 1;
          const snapshot = [...serverModels];
          if (listReads === 2) {
            return new Promise<Response>((resolve) => {
              finishOld = () => resolve(Response.json({ models: snapshot }));
            });
          }
          if (listReads === 3) {
            return new Promise<Response>((resolve) => {
              finishTrailing = () => resolve(Response.json({ models: serverModels }));
            });
          }
          return Promise.resolve(Response.json({ models: snapshot }));
        }
        if (url.pathname.endsWith('/models/i/model-a')) {
          mutationRequests.push(init?.method ?? 'GET');
          if (init?.method === 'DELETE') {
            serverModels = [];
            return Promise.resolve(new Response(null, { status: 204 }));
          }
          const model = { ...original, name: 'Confirmed rename' };
          serverModels = [model];
          return Promise.resolve(Response.json(model));
        }
        throw new Error(`Unexpected model request: ${url.pathname}`);
      })
    );
    await refreshModels();
    const oldRead = refreshModels();
    if (mutation === 'rename') {
      replaceModelInStore(await updateModel(original.key, { name: 'Confirmed rename' }));
    } else {
      await deleteModel(original.key);
      removeModelsFromStore([original.key]);
    }
    const confirmed = getModelsSnapshot().models;
    finishOld();
    await oldRead;
    expect(getModelsSnapshot().models).toEqual(confirmed);
    expect(getModelsSnapshot().error).toBeNull();
    expect(listReads).toBe(3);
    expect(mutationRequests).toEqual([mutation === 'rename' ? 'PATCH' : 'DELETE']);
    serverModels = [...serverModels, { ...original, key: 'model-b', name: 'Unrelated server model' }];
    finishTrailing();
    await vi.waitFor(() => expect(getModelsSnapshot().modelsByKey.get('model-b')?.name).toBe('Unrelated server model'));
    expect(getModelsSnapshot().modelsByKey.get('model-a')?.name).toBe(
      mutation === 'rename' ? 'Confirmed rename' : undefined
    );
    expect(listReads).toBe(3);
  }
);
