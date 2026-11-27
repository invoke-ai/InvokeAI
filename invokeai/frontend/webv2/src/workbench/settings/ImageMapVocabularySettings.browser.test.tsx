import type { ImageMapVocab } from '@workbench/image-map/vocabulary';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const mocks = vi.hoisted(() => ({
  canManage: true,
  indexCounts: null as { total: number; embedded: number; pending: number; failed: number } | null,
  refreshImageIndexStatus: vi.fn(),
  updateImageMapVocab: vi.fn((_terms: string[]) => Promise.resolve(null as unknown as ImageMapVocab)),
  vocab: null as ImageMapVocab | null,
}));

vi.mock('@features/identity', () => ({
  useCapabilities: () => ({ canManageImageMapVocabulary: mocks.canManage }),
}));

vi.mock('@workbench/image-map/imageMapStore', () => ({
  imageMapStore: {
    useSelector: (selector: (snapshot: object) => unknown) =>
      selector({ indexCounts: mocks.indexCounts, indexUpdatedAt: null }),
  },
  refreshImageIndexStatus: () => mocks.refreshImageIndexStatus(),
}));

vi.mock('@workbench/image-map/vocabulary', () => ({
  imageMapVocabKeys: { all: ['image-map', 'vocab'] },
  imageMapVocabQueryOptions: () => ({
    queryFn: () => Promise.resolve(mocks.vocab),
    queryKey: ['image-map', 'vocab'],
    staleTime: Infinity,
  }),
  updateImageMapVocab: (terms: string[]) => mocks.updateImageMapVocab(terms),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      ({
        'common.add': 'Add',
        'common.retry': 'Retry',
        'settings.imageMapVocabulary.addHelp': 'Terms are lowercased and applied the next time labels are computed.',
        'settings.imageMapVocabulary.addLabel': 'Add terms',
        'settings.imageMapVocabulary.addPlaceholder': 'Add terms (comma or newline separated)',
        'settings.imageMapVocabulary.adminOnly': 'Only an administrator can change the vocabulary.',
        'settings.imageMapVocabulary.alreadyInList': 'Every entered term is already in the list.',
        'settings.imageMapVocabulary.buildFailed': `The vocabulary could not be prepared: ${String(options?.message)}`,
        'settings.imageMapVocabulary.count': `${String(options?.count)} of ${String(options?.max)} supplementary terms`,
        'settings.imageMapVocabulary.indexOff': 'The image index is not running.',
        'settings.imageMapVocabulary.loadFailed': 'Could not load the vocabulary.',
        'settings.imageMapVocabulary.noTermsYet': 'No supplementary terms yet.',
        'settings.imageMapVocabulary.rebuilding': 'Updating cluster labels with the new vocabulary…',
        'settings.imageMapVocabulary.rebuildingQueued': `Waiting for image indexing to finish (${String(options?.progress)}) before updating cluster labels…`,
        'settings.imageMapVocabulary.removeTerm': `Remove ${String(options?.term)}`,
        'settings.imageMapVocabulary.saveFailed': 'Could not save the vocabulary.',
        'settings.imageMapVocabulary.termTooLong': `Terms must be ${String(options?.max)} characters or fewer.`,
        'settings.imageMapVocabulary.tooManyTerms': `The list is limited to ${String(options?.max)} terms.`,
        'widgets.imageMap.progress.counts': `${String(options?.processed)} of ${String(options?.total)} images`,
      })[key] ?? key,
  }),
}));

import { ImageMapVocabularySettings } from './ImageMapVocabularySettings';

const BASE_VOCAB: ImageMapVocab = {
  error: null,
  maxTerms: 500,
  maxTermLength: 64,
  state: 'ready',
  terms: [],
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const flushQueries = async (): Promise<void> => {
  // The query resolves in a microtask; let effects and state settle.
  await act(async () => {
    await Promise.resolve();
  });
};

const render = async (vocab: Partial<ImageMapVocab>): Promise<void> => {
  mocks.vocab = { ...BASE_VOCAB, ...vocab };

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <ImageMapVocabularySettings />
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
  await flushQueries();
};

const input = (): HTMLInputElement | null =>
  host!.querySelector<HTMLInputElement>('input[type="text"], input:not([type])');

const chipCloseTriggers = (): HTMLElement[] =>
  Array.from(host!.querySelectorAll<HTMLElement>('[aria-label^="Remove "]'));

beforeEach(() => {
  mocks.canManage = true;
  mocks.indexCounts = null;
  mocks.refreshImageIndexStatus.mockClear();
  mocks.updateImageMapVocab.mockClear();
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  queryClient = null;
  host?.remove();
  host = null;
  root = null;
});

describe('ImageMapVocabularySettings', () => {
  it('splits a pasted list, normalizes it, and saves the merged terms', async () => {
    mocks.updateImageMapVocab.mockImplementation((terms: string[]) =>
      Promise.resolve({ ...BASE_VOCAB, state: 'building' as const, terms })
    );
    await render({ terms: ['zebra'] });

    await userEvent.fill(input()!, '  Golden   Retriever, OKAPI, zebra');
    await userEvent.keyboard('{Enter}');
    await flushQueries();

    // "zebra" is already stored and drops out; the rest arrive normalized.
    expect(mocks.updateImageMapVocab).toHaveBeenCalledWith(['zebra', 'golden retriever', 'okapi']);
    // The draft clears only after a successful save.
    expect(input()?.value).toBe('');
    // The save's response reports the background rebuild.
    expect(host?.textContent).toContain('Updating cluster labels with the new vocabulary…');
  });

  it('removes a chip by saving the list without it', async () => {
    mocks.updateImageMapVocab.mockImplementation((terms: string[]) =>
      Promise.resolve({ ...BASE_VOCAB, state: 'building' as const, terms })
    );
    await render({ terms: ['aardvark', 'zebra'] });

    await userEvent.click(chipCloseTriggers()[0]!);
    await flushQueries();

    expect(mocks.updateImageMapVocab).toHaveBeenCalledWith(['zebra']);
  });

  it('shows a read-only list to a non-admin', async () => {
    mocks.canManage = false;
    await render({ terms: ['zebra'] });

    expect(host?.textContent).toContain('zebra');
    expect(host?.textContent).toContain('Only an administrator can change the vocabulary.');
    expect(input()).toBeNull();
    expect(chipCloseTriggers()).toHaveLength(0);
  });
});
