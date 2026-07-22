import type { WidgetContributionSource, WidgetSearchProvider } from '@workbench/widgetContracts';

import { describe, expect, it, vi } from 'vitest';

import { createExtensionSearchProvider } from './extensionPaletteAdapters';
import { getPaletteProviderQueryKey } from './providerQueryKey';

const source = (instanceId: string): WidgetContributionSource => ({
  instanceId,
  projectId: 'project-1',
  region: 'left',
  typeId: 'generate',
});

const extensionProvider = (instanceId: string): WidgetSearchProvider => ({
  id: 'shared-provider-id',
  label: 'Shared',
  search: () => [{ commandId: 'run', id: 'shared-result-id', title: 'Result' }],
  source: source(instanceId),
});

describe('extension palette adapters', () => {
  it('keeps providers, query keys, scopes, and results collision-free per widget source', async () => {
    const execute = vi.fn();
    const first = createExtensionSearchProvider(extensionProvider('one'), execute);
    const second = createExtensionSearchProvider(extensionProvider('two'), execute);
    const [firstResult] = await first.search({ text: 'query' });
    const [secondResult] = await second.search({ text: 'query' });

    expect(first.providerKey).not.toBe(second.providerKey);
    expect(getPaletteProviderQueryKey(first, { text: 'query' })).not.toEqual(
      getPaletteProviderQueryKey(second, { text: 'query' })
    );
    expect(firstResult?.id).not.toBe(secondResult?.id);
  });

  it('passes only the stripped query text through to the extension search contract', async () => {
    const search = vi.fn(() => [{ commandId: 'run', id: 'shared-result-id', title: 'Result' }]);
    const provider = createExtensionSearchProvider({ ...extensionProvider('one'), search }, vi.fn());

    await provider.search({ range: { from: '2026-07-14', to: '2026-07-21' }, text: 'sunset' });

    expect(search).toHaveBeenCalledWith('sunset');
    expect(provider.supportsCreatedAtRange).toBeUndefined();
  });

  it('includes immutable context in provider query keys', () => {
    const provider = createExtensionSearchProvider(extensionProvider('one'), vi.fn());
    const changedProject = { ...provider, contextKey: 'project-2' };
    const changedBoard = { ...provider, contextKey: 'board-2' };
    const changedQueueScope = { ...provider, contextKey: JSON.stringify({ originPrefix: 'project:2' }) };
    const changedHistory = { ...provider, contextKey: JSON.stringify([{ positivePrompt: 'new' }]) };

    for (const changed of [changedProject, changedBoard, changedQueueScope, changedHistory]) {
      expect(getPaletteProviderQueryKey(changed, { text: 'same query' })).not.toEqual(
        getPaletteProviderQueryKey(provider, { text: 'same query' })
      );
    }
  });

  it('differentiates query keys by resolved date range', () => {
    const provider = createExtensionSearchProvider(extensionProvider('one'), vi.fn());

    expect(getPaletteProviderQueryKey(provider, { text: 'from:7d' })).not.toEqual(
      getPaletteProviderQueryKey(provider, { range: { from: '2026-07-14' }, text: '' })
    );
    expect(getPaletteProviderQueryKey(provider, { range: { from: '2026-07-14' }, text: '' })).not.toEqual(
      getPaletteProviderQueryKey(provider, { range: { from: '2026-07-15' }, text: '' })
    );
  });
});
