import { useWildcards, WildcardWriteError } from '@features/generation/ui/useWildcards';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const createWildcard = vi.fn(() => Promise.resolve());
const updateWildcard = vi.fn(() => Promise.resolve());
const invalidateWildcardDependents = vi.fn(() => Promise.resolve());

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: (...args: unknown[]) => createWildcard(...(args as [])),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: (...args: unknown[]) => invalidateWildcardDependents(...(args as [])),
  updateWildcard: (...args: unknown[]) => updateWildcard(...(args as [])),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const catalogRef = createRef<ReturnType<typeof useWildcards>>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ ref }: { ref: Ref<ReturnType<typeof useWildcards>> }) => {
  const value = useWildcards();

  useImperativeHandle(ref, () => value, [value]);

  return null;
};

beforeEach(async () => {
  vi.clearAllMocks();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
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

  // Every single-write method invalidates the wildcard cache *and* every
  // dynamic-prompt expansion, and awaits the refetch. Doing that between each
  // write of a forty-file import meant re-reading a list that grew by one, forty
  // times over.
  it('invalidates once for the whole run, not once per write', async () => {
    await act(async () => {
      await catalogRef.current!.applyWrites(WRITES);
    });

    expect(createWildcard).toHaveBeenCalledTimes(2);
    expect(updateWildcard).toHaveBeenCalledTimes(1);
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('reports how many landed', async () => {
    let done = 0;

    await act(async () => {
      done = await catalogRef.current!.applyWrites(WRITES);
    });

    expect(done).toBe(3);
  });

  // The ones already written are real, so the caches are stale whether or not
  // the run finished — and the caller needs the count to say how far it got.
  it('still invalidates, and carries the count, when a write fails', async () => {
    createWildcard.mockImplementationOnce(() => Promise.resolve());
    updateWildcard.mockImplementationOnce(() => Promise.reject(new Error('conflict')));

    let caught: unknown;

    await act(async () => {
      caught = await catalogRef.current!.applyWrites(WRITES).catch((error: unknown) => error);
    });

    expect(caught).toBeInstanceOf(WildcardWriteError);
    expect((caught as WildcardWriteError).done).toBe(1);
    expect((caught as WildcardWriteError).cause).toMatchObject({ message: 'conflict' });
    expect(invalidateWildcardDependents).toHaveBeenCalledTimes(1);
  });

  it('does not invalidate when the very first write fails', async () => {
    createWildcard.mockImplementationOnce(() => Promise.reject(new Error('nope')));

    await act(async () => {
      await catalogRef.current!.applyWrites(WRITES).catch(() => undefined);
    });

    expect(invalidateWildcardDependents).not.toHaveBeenCalled();
  });
});
