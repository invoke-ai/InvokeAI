import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryCacheModule from './libraryCache';

const api = vi.hoisted(() => ({
  getLibraryWorkflowRecord: vi.fn(),
  listLibraryWorkflows: vi.fn(),
}));

vi.mock('./api', () => api);

let account: typeof accountLifecycleModule;
let cache: typeof libraryCacheModule;

const params = { category: 'user' as const, page: 1, perPage: 20 };

beforeEach(async () => {
  vi.resetModules();
  api.getLibraryWorkflowRecord.mockReset();
  api.listLibraryWorkflows.mockReset();
  cache = await import('./libraryCache');
  account = await import('@platform/state/accountLifecycle');
});

describe('workflow library account ownership', () => {
  it('clears pages and payloads synchronously on account invalidation', async () => {
    account.accountLifecycle.activate('user-a');
    api.listLibraryWorkflows.mockResolvedValue({ items: [], page: 1, pages: 1, per_page: 20, total: 0 });
    api.getLibraryWorkflowRecord.mockResolvedValue({ workflow: {}, workflow_id: 'workflow-a' });

    await cache.listLibraryWorkflowsCached(params);
    await cache.getLibraryWorkflowCached('workflow-a');
    expect(cache.getCachedWorkflowPage(params)).not.toBeNull();

    account.accountLifecycle.invalidate();

    expect(cache.getCachedWorkflowPage(params)).toBeNull();
    api.getLibraryWorkflowRecord.mockResolvedValue({ workflow: {}, workflow_id: 'after-clear' });
    await cache.getLibraryWorkflowCached('workflow-a');
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('rejects delayed page and payload completions from the prior epoch', async () => {
    account.accountLifecycle.activate('user-a');
    let resolvePage: ((value: unknown) => void) | undefined;
    let resolveWorkflow: ((value: Record<string, unknown>) => void) | undefined;
    api.listLibraryWorkflows.mockReturnValueOnce(
      new Promise((resolve) => {
        resolvePage = resolve;
      })
    );
    api.getLibraryWorkflowRecord.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveWorkflow = resolve;
      })
    );

    const oldPage = cache.listLibraryWorkflowsCached(params);
    const oldWorkflow = cache.getLibraryWorkflowCached('shared-id');
    const pageSignal = api.listLibraryWorkflows.mock.calls[0]?.[0]?.signal as AbortSignal;
    const workflowSignal = api.getLibraryWorkflowRecord.mock.calls[0]?.[1] as AbortSignal;

    expect(pageSignal.aborted).toBe(false);
    expect(workflowSignal.aborted).toBe(false);
    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    expect(pageSignal.aborted).toBe(true);
    expect(workflowSignal.aborted).toBe(true);
    resolvePage?.({ items: [{ workflow_id: 'a' }], page: 1, pages: 1, per_page: 20, total: 1 });
    resolveWorkflow?.({ workflow: { owner: 'a' }, workflow_id: 'shared-id' });

    await expect(oldPage).rejects.toThrow('no longer active');
    await expect(oldWorkflow).rejects.toThrow('no longer active');
    expect(cache.getCachedWorkflowPage(params)).toBeNull();

    api.getLibraryWorkflowRecord.mockResolvedValueOnce({ workflow: { owner: 'b' }, workflow_id: 'shared-id' });
    await expect(cache.getLibraryWorkflowCached('shared-id')).resolves.toEqual({ id: 'shared-id', owner: 'b' });
  });
});

describe('workflow library page cache keying', () => {
  beforeEach(() => {
    account.accountLifecycle.activate('user-a');
    api.listLibraryWorkflows.mockResolvedValue({ items: [], page: 1, pages: 1, per_page: 20, total: 0 });
  });

  it('keys distinct tag filters into distinct cache entries, fetching each once', async () => {
    const upscaling = { ...params, tags: ['upscaling'] };
    const lora = { ...params, tags: ['lora'] };

    await cache.listLibraryWorkflowsCached(upscaling);
    await cache.listLibraryWorkflowsCached(lora);

    expect(api.listLibraryWorkflows).toHaveBeenCalledTimes(2);
    expect(cache.getCachedWorkflowPage(upscaling)).not.toBeNull();
    expect(cache.getCachedWorkflowPage(lora)).not.toBeNull();
  });

  it('hits the cache for a repeat request with the same tags regardless of order', async () => {
    const first = { ...params, tags: ['lora', 'upscaling'] };
    const second = { ...params, tags: ['upscaling', 'lora'] };

    await cache.listLibraryWorkflowsCached(first);

    expect(api.listLibraryWorkflows).toHaveBeenCalledTimes(1);
    expect(cache.getCachedWorkflowPage(second)).not.toBeNull();
    expect(cache.getCachedWorkflowPage(second)).toBe(cache.getCachedWorkflowPage(first));
  });

  it('does not collide an untagged request with a tagged one for the same page', async () => {
    const untagged = params;
    const tagged = { ...params, tags: ['lora'] };

    await cache.listLibraryWorkflowsCached(untagged);

    expect(cache.getCachedWorkflowPage(tagged)).toBeNull();

    await cache.listLibraryWorkflowsCached(tagged);

    expect(api.listLibraryWorkflows).toHaveBeenCalledTimes(2);
  });
});

describe('workflow library cache invalidation listeners', () => {
  it('notifies registered listeners when the cache is invalidated', () => {
    const listener = vi.fn();
    const unsubscribe = cache.onWorkflowLibraryCacheInvalidated(listener);

    cache.invalidateWorkflowLibraryCache();
    expect(listener).toHaveBeenCalledTimes(1);

    cache.invalidateWorkflowLibraryCache('workflow-1');
    expect(listener).toHaveBeenLastCalledWith('workflow-1');

    unsubscribe();
    cache.invalidateWorkflowLibraryCache();
    expect(listener).toHaveBeenCalledTimes(2);
  });
});
