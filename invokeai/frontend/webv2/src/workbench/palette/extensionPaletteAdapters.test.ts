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
    const [firstResult] = await first.search('query');
    const [secondResult] = await second.search('query');

    expect(first.providerKey).not.toBe(second.providerKey);
    expect(getPaletteProviderQueryKey(first, 'query')).not.toEqual(getPaletteProviderQueryKey(second, 'query'));
    expect(firstResult?.id).not.toBe(secondResult?.id);
  });

  it('includes immutable context in provider query keys', () => {
    const provider = createExtensionSearchProvider(extensionProvider('one'), vi.fn());
    const changedProject = { ...provider, contextKey: 'project-2' };
    const changedBoard = { ...provider, contextKey: 'board-2' };
    const changedQueueScope = { ...provider, contextKey: JSON.stringify({ originPrefix: 'project:2' }) };
    const changedHistory = { ...provider, contextKey: JSON.stringify([{ positivePrompt: 'new' }]) };

    for (const changed of [changedProject, changedBoard, changedQueueScope, changedHistory]) {
      expect(getPaletteProviderQueryKey(changed, 'same query')).not.toEqual(
        getPaletteProviderQueryKey(provider, 'same query')
      );
    }
  });
});
