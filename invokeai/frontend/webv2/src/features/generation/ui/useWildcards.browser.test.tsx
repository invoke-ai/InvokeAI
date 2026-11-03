import type * as WildcardsModule from '@features/generation/data/wildcards';

import { dynamicPromptsKeys } from '@features/generation/data/dynamicPromptsQueries';
import { useWildcards, WildcardDeleteError, WildcardWriteError } from '@features/generation/ui/useWildcards';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const createWildcard = vi.fn(() => Promise.resolve());
const updateWildcard = vi.fn(() => Promise.resolve());
// Records each call; the real invalidation still runs against the test's query client.
const invalidateWildcardDependents = vi.fn();
// Answers as the server does: every id it was sent was the caller's.
const deleteWildcards = vi.fn((ids: readonly string[], _signal?: AbortSignal) => Promise.resolve([...ids]));
const listWildcards = vi.fn<() => Promise<WildcardsModule.WildcardRecord[]>>();

vi.mock('@features/generation/data/wildcards', async (importOriginal) => {
  // Partial: the real module's other exports (constants, mappers) stay available to what imports them.
  const actual = await importOriginal<typeof WildcardsModule>();

  return {
    ...actual,
    createWildcard: (...args: unknown[]) => createWildcard(...(args as [])),
    deleteWildcard: vi.fn(),
    deleteWildcards: (ids: readonly string[], signal?: AbortSignal) => deleteWildcards(ids, signal),
    invalidateWildcardDependents: (queryClient: QueryClient) => {
      invalidateWildcardDependents(queryClient);

      return actual.invalidateWildcardDependents(queryClient);
    },
    updateWildcard: (...args: unknown[]) => updateWildcard(...(args as [])),
    wildcardsQueryOptions: () => ({ queryFn: () => listWildcards(), queryKey: actual.wildcardKeys.all }),
  };
});

/** Holds the catalog's refetches until released, as a refresh retrying against an outage would. */
const holdCatalogRefetch = (answer: () => WildcardsModule.WildcardRecord[] = () => []): (() => void) => {
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });

  listWildcards.mockImplementation(() => held.then(answer));

  return release;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient;
const catalogRef = createRef<ReturnType<typeof useWildcards>>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ ref }: { ref: Ref<ReturnType<typeof useWildcards>> }) => {
  const value = useWildcards();

  useImperativeHandle(ref, () => value, [value]);

  return null;
};

beforeEach(async () => {
  vi.clearAllMocks();
  listWildcards.mockImplementation(() => Promise.resolve([]));
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient();

  await act(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <Probe ref={catalogRef} />
      </QueryClientProvider>
    );
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('applyWrites', () => {
  const WRITES = [
    { name: 'a', values: ['1'] },
    { id: 'w2', name: 'b', values: ['2'] },
    { name: 'c', values: ['3'] },
  ];

  it('invalidates once for the whole run, not once per write', async () => {
    let done = 0;

    await act(async () => {
      done = await catalogRef.current!.applyWrites(WRITES, captureAccountScope());
    });

    expect(done).toBe(3);
    expect(createWildcard).toHaveBeenCalledTimes(2);
    expect(updateWildcard).toHaveBeenCalledTimes(1);
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('still invalidates, and carries the count, when a write fails', async () => {
    createWildcard.mockImplementationOnce(() => Promise.resolve());
    updateWildcard.mockImplementationOnce(() => Promise.reject(new Error('conflict')));

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.applyWrites(WRITES, captureAccountScope()).catch((error: unknown) => error);
    });

    expect(caught).toBeInstanceOf(WildcardWriteError);
    expect((caught as WildcardWriteError).done).toBe(1);
    expect((caught as WildcardWriteError).cause).toMatchObject({ message: 'conflict' });
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('invalidates when the very first write loses its answer, since the server may have committed it', async () => {
    createWildcard.mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')));

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.applyWrites(WRITES, captureAccountScope()).catch((error: unknown) => error);
    });

    expect((caught as WildcardWriteError).done).toBe(0);
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('does not invalidate when there was nothing to write', async () => {
    await act(async () => {
      await catalogRef.current!.applyWrites([], captureAccountScope());
    });

    expect(invalidateWildcardDependents).not.toHaveBeenCalled();
  });
});

describe('single writes', () => {
  it('invalidates when a create loses its answer, and reports that failure without waiting on the refresh', async () => {
    createWildcard.mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')));
    holdCatalogRefetch();

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.create({ name: 'a', values: ['1'] }).catch((error: unknown) => error);
    });

    expect(caught).toMatchObject({ message: 'Failed to fetch' });
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('reports only the write failure when the refresh behind it fails too', async () => {
    const unhandled = vi.fn();
    window.addEventListener('unhandledrejection', unhandled);
    createWildcard.mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')));
    vi.spyOn(queryClient, 'invalidateQueries').mockRejectedValueOnce(new Error('refresh failed'));

    let caught: unknown;

    try {
      await act(async () => {
        caught = await catalogRef.current!.create({ name: 'a', values: ['1'] }).catch((error: unknown) => error);
        // Let a rejection nobody handled reach the window.
        await new Promise((resolve) => {
          globalThis.setTimeout(resolve, 20);
        });
      });
    } finally {
      window.removeEventListener('unhandledrejection', unhandled);
    }

    expect(caught).toMatchObject({ message: 'Failed to fetch' });
    expect(unhandled).not.toHaveBeenCalled();
  });
});

describe('removeMany', () => {
  const ids = (count: number) => Array.from({ length: count }, (_, index) => `w${String(index)}`);

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('sends a large selection in sequential batches of at most 1000 and invalidates once', async () => {
    let deleted: string[] = [];

    await act(async () => {
      deleted = await catalogRef.current!.removeMany(ids(2500));
    });

    expect(deleteWildcards.mock.calls.map(([batch]) => batch.length)).toEqual([1000, 1000, 500]);
    expect(deleted).toEqual(ids(2500));
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('sends 720 ids in one request', async () => {
    await act(async () => {
      await catalogRef.current!.removeMany(ids(720));
    });

    expect(deleteWildcards).toHaveBeenCalledOnce();
  });

  it('refreshes the catalog when the server deleted none of them, as when another tab already had', async () => {
    deleteWildcards.mockImplementationOnce(() => Promise.resolve([]));

    await act(async () => {
      await catalogRef.current!.removeMany(ids(3));
    });

    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('drops the deleted row and cached expansions when the server committed the delete but its answer was lost', async () => {
    let server: WildcardsModule.WildcardRecord[] = [
      { id: 'w0', name: 'animal', values: ['cat'] },
      { id: 'w1', name: 'color', values: ['red'] },
    ];
    listWildcards.mockImplementation(() => Promise.resolve(server));
    const expansionKey = dynamicPromptsKeys.expansion({ prompt: 'a __animal__' });
    queryClient.setQueryData(expansionKey, { prompts: ['a cat'] });
    await act(() => queryClient.refetchQueries({ queryKey: ['generation', 'wildcards'] }));
    expect(catalogRef.current!.wildcards.map(({ id }) => id)).toEqual(['w0', 'w1']);

    deleteWildcards.mockImplementationOnce((batch) => {
      server = server.filter(({ id }) => !batch.includes(id));

      return Promise.reject(new TypeError('Failed to fetch'));
    });
    const releaseRefetch = holdCatalogRefetch(() => server);

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.removeMany(['w0']).catch((error: unknown) => error);
    });

    // The failure is reported while the refresh is still out.
    expect(caught).toBeInstanceOf(WildcardDeleteError);
    expect((caught as WildcardDeleteError).deletedIds).toEqual([]);
    expect(queryClient.getQueryState(expansionKey)?.isInvalidated).toBe(true);
    expect(catalogRef.current!.wildcards.map(({ id }) => id)).toEqual(['w0', 'w1']);

    await act(async () => {
      releaseRefetch();
      await vi.waitFor(() => expect(queryClient.isFetching({ queryKey: ['generation', 'wildcards'] })).toBe(0));
    });

    expect(catalogRef.current!.wildcards.map(({ id }) => id)).toEqual(['w1']);
  });

  it('invalidates after a partial failure and names the ids deleted before it', async () => {
    deleteWildcards.mockImplementationOnce((batch) => Promise.resolve([...batch]));
    deleteWildcards.mockImplementationOnce(() => Promise.reject(new Error('offline')));

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.removeMany(ids(1500)).catch((error: unknown) => error);
    });

    expect(caught).toBeInstanceOf(WildcardDeleteError);
    expect((caught as WildcardDeleteError).deletedIds).toEqual(ids(1000));
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('stops, and leaves the next account alone, once the account it started under is gone', async () => {
    accountLifecycle.activate('wildcards-delete-a', ':user:wildcards-delete-a');
    deleteWildcards.mockImplementationOnce((batch) => {
      accountLifecycle.activate('wildcards-delete-b', ':user:wildcards-delete-b');

      return Promise.resolve([...batch]);
    });

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.removeMany(ids(1500)).catch((error: unknown) => error);
    });

    expect(caught).toBeInstanceOf(Error);
    expect(deleteWildcards).toHaveBeenCalledOnce();
    expect(invalidateWildcardDependents).not.toHaveBeenCalled();
  });
});
