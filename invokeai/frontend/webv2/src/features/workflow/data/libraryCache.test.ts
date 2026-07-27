import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryCacheModule from './libraryCache';

const api = vi.hoisted(() => ({
  getLibraryWorkflow: vi.fn(),
  listLibraryWorkflows: vi.fn(),
}));

vi.mock('./api', () => api);

let account: typeof accountLifecycleModule;
let cache: typeof libraryCacheModule;

const params = { category: 'user' as const, page: 1, perPage: 20 };

beforeEach(async () => {
  vi.resetModules();
  api.getLibraryWorkflow.mockReset();
  api.listLibraryWorkflows.mockReset();
  cache = await import('./libraryCache');
  account = await import('@platform/state/accountLifecycle');
});

describe('workflow library account ownership', () => {
  it('clears pages and payloads synchronously on account invalidation', async () => {
    account.accountLifecycle.activate('user-a');
    api.listLibraryWorkflows.mockResolvedValue({ items: [], page: 1, pages: 1, per_page: 20, total: 0 });
    api.getLibraryWorkflow.mockResolvedValue({ id: 'workflow-a' });

    await cache.listLibraryWorkflowsCached(params);
    await cache.getLibraryWorkflowCached('workflow-a');
    expect(cache.getCachedWorkflowPage(params)).not.toBeNull();

    account.accountLifecycle.invalidate();

    expect(cache.getCachedWorkflowPage(params)).toBeNull();
    api.getLibraryWorkflow.mockResolvedValue({ id: 'after-clear' });
    await cache.getLibraryWorkflowCached('workflow-a');
    expect(api.getLibraryWorkflow).toHaveBeenCalledTimes(2);
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
    api.getLibraryWorkflow.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveWorkflow = resolve;
      })
    );

    const oldPage = cache.listLibraryWorkflowsCached(params);
    const oldWorkflow = cache.getLibraryWorkflowCached('shared-id');
    const pageSignal = api.listLibraryWorkflows.mock.calls[0]?.[0]?.signal as AbortSignal;
    const workflowSignal = api.getLibraryWorkflow.mock.calls[0]?.[1] as AbortSignal;

    expect(pageSignal.aborted).toBe(false);
    expect(workflowSignal.aborted).toBe(false);
    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    expect(pageSignal.aborted).toBe(true);
    expect(workflowSignal.aborted).toBe(true);
    resolvePage?.({ items: [{ workflow_id: 'a' }], page: 1, pages: 1, per_page: 20, total: 1 });
    resolveWorkflow?.({ owner: 'a' });

    await expect(oldPage).rejects.toThrow('no longer active');
    await expect(oldWorkflow).rejects.toThrow('no longer active');
    expect(cache.getCachedWorkflowPage(params)).toBeNull();

    api.getLibraryWorkflow.mockResolvedValueOnce({ owner: 'b' });
    await expect(cache.getLibraryWorkflowCached('shared-id')).resolves.toEqual({ owner: 'b' });
  });
});
