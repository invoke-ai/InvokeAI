import { ApiError } from '@platform/transport/http';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ServerOutputRead, ServerOutputRow } from './transcript';

import { createServerOutputReader, SERVER_OUTPUT_READER_TIMING, type DocumentVisibility } from './reader';
import { iterateRows } from './transcript';

interface PendingRead {
  cursor: string | null;
  signal: AbortSignal;
  resolve: (value: ServerOutputRead) => void;
  reject: (error: unknown) => void;
}

const row = (id: number, text = `line ${id}`): ServerOutputRow => ({ id, styles: [], text, truncated: false });

const response = (cursor: string, overrides: Partial<ServerOutputRead> = {}): ServerOutputRead => ({
  cursor,
  floor_id: 1,
  has_more: false,
  reset: null,
  rows: [],
  run_id: 'run',
  snapshot_truncated: false,
  ...overrides,
});

const createVisibility = () => {
  let hidden = false;
  const listeners = new Set<() => void>();
  const visibility: DocumentVisibility = {
    isHidden: () => hidden,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  return {
    setHidden: (value: boolean) => {
      hidden = value;
      for (const listener of listeners) {
        listener();
      }
    },
    visibility,
  };
};

let reads: PendingRead[];
let lifetime: AbortController;
let page: ReturnType<typeof createVisibility>;
let onRefused: ReturnType<typeof vi.fn<(reason: string) => void>>;

const createReader = () =>
  createServerOutputReader({
    lifetime: lifetime.signal,
    onRefused,
    read: (cursor, signal) =>
      new Promise((resolve, reject) => {
        reads.push({ cursor, reject, resolve, signal });
      }),
    visibility: page.visibility,
  });

const settle = async (index: number, value: ServerOutputRead | Error) => {
  const pending = reads[index]!;
  if (value instanceof Error) {
    pending.reject(value);
  } else {
    pending.resolve(value);
  }
  await vi.advanceTimersByTimeAsync(0);
};

const heldIds = (reader: ReturnType<typeof createReader>) =>
  [...iterateRows(reader.getSnapshot().transcript)].map((held) => held.id);

beforeEach(() => {
  vi.useFakeTimers();
  reads = [];
  lifetime = new AbortController();
  page = createVisibility();
  onRefused = vi.fn<(reason: string) => void>();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('server output reader', () => {
  it('keeps one read in flight and schedules the next after it settles', async () => {
    const reader = createReader();
    reader.start();

    expect(reads).toHaveLength(1);
    expect(reads[0]!.cursor).toBeNull();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(reads).toHaveLength(1);

    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    expect(reader.getSnapshot().phase).toBe('live');
    await vi.advanceTimersByTimeAsync(SERVER_OUTPUT_READER_TIMING.activeMs - 1);
    expect(reads).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(reads).toHaveLength(2);
    expect(reads[1]!.cursor).toBe('c1');
  });

  it('drains pages at once, then backs off while idle and speeds up again on output', async () => {
    const reader = createReader();
    reader.start();

    await settle(0, response('c1', { has_more: true, reset: 'initial', rows: [row(1)] }));
    expect(reads).toHaveLength(2);
    expect(reads[1]!.cursor).toBe('c1');

    await settle(1, response('c2', { rows: [row(2)] }));
    const delays: number[] = [];
    for (let index = 2; index < 7; index += 1) {
      const before = Date.now();
      await vi.advanceTimersToNextTimerAsync();
      delays.push(Date.now() - before);
      expect(reads).toHaveLength(index + 1);
      await settle(index, response('c2'));
    }
    expect(delays).toEqual([250, 500, 1_000, 2_000, 2_000]);

    await vi.advanceTimersToNextTimerAsync();
    await settle(7, response('c3', { rows: [row(3)] }));
    const before = Date.now();
    await vi.advanceTimersToNextTimerAsync();
    expect(Date.now() - before).toBe(250);
    expect(heldIds(reader)).toEqual([1, 2, 3]);
  });

  it('reconnects from the last applied cursor after transient failures, keeping held rows', async () => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    await vi.advanceTimersToNextTimerAsync();

    await settle(1, new TypeError('Failed to fetch'));
    expect(reader.getSnapshot().phase).toBe('reconnecting');
    expect(heldIds(reader)).toEqual([1]);

    const delays: number[] = [];
    for (let index = 2; index < 7; index += 1) {
      const before = Date.now();
      await vi.advanceTimersToNextTimerAsync();
      delays.push(Date.now() - before);
      expect(reads[index]!.cursor).toBe('c1');
      await settle(index, new ApiError('Bad gateway', 502));
    }
    expect(delays).toEqual([1_000, 2_000, 4_000, 8_000, 15_000]);

    await vi.advanceTimersToNextTimerAsync();
    await settle(7, response('c2', { rows: [row(2)] }));
    expect(reader.getSnapshot().phase).toBe('live');
    expect(heldIds(reader)).toEqual([1, 2]);
  });

  it.each([
    [401, 'refused'],
    [403, 'refused'],
    [404, 'unavailable'],
  ] as const)('clears and stops on %i', async (status, phase) => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    await vi.advanceTimersToNextTimerAsync();

    await settle(1, new ApiError('Refused', status));

    expect(reader.getSnapshot()).toMatchObject({ phase, transcript: { rowCount: 0 } });
    expect(onRefused).toHaveBeenCalledExactlyOnceWith(phase === 'refused' ? 'forbidden' : 'unavailable');
    await vi.advanceTimersByTimeAsync(60_000);
    expect(reads).toHaveLength(2);
  });

  it('starts over from a fresh snapshot when the server rejects the cursor', async () => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    await vi.advanceTimersToNextTimerAsync();

    await settle(1, new ApiError('Malformed cursor', 422));
    expect(reads).toHaveLength(3);
    expect(reads[2]!.cursor).toBeNull();

    await settle(2, response('c9', { reset: 'initial', rows: [row(7)] }));
    expect(heldIds(reader)).toEqual([7]);
    expect(reader.getSnapshot().phase).toBe('live');
  });

  it('treats a rejected snapshot read as transient rather than looping on fresh snapshots', async () => {
    const reader = createReader();
    reader.start();

    await settle(0, new ApiError('Malformed cursor', 422));
    expect(reader.getSnapshot().phase).toBe('reconnecting');
    await vi.advanceTimersByTimeAsync(SERVER_OUTPUT_READER_TIMING.errorMs - 1);
    expect(reads).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(reads).toHaveLength(2);
    expect(reads[1]!.cursor).toBeNull();
  });

  it('starts its error backoff over after a successful read', async () => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));

    for (let index = 1; index <= 3; index += 1) {
      await vi.advanceTimersToNextTimerAsync();
      await settle(index, new TypeError('Failed to fetch'));
    }
    await vi.advanceTimersToNextTimerAsync();
    await settle(4, response('c2', { rows: [row(2)] }));
    await vi.advanceTimersToNextTimerAsync();

    await settle(5, new TypeError('Failed to fetch'));
    const before = Date.now();
    await vi.advanceTimersToNextTimerAsync();
    expect(Date.now() - before).toBe(SERVER_OUTPUT_READER_TIMING.errorMs);
  });

  it('aborts on stop and ignores a response that arrives afterwards', async () => {
    const reader = createReader();
    const stop = reader.start();

    stop();
    expect(reads[0]!.signal.aborted).toBe(true);
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    await vi.advanceTimersByTimeAsync(60_000);

    expect(reads).toHaveLength(1);
    expect(reader.getSnapshot().transcript.rowCount).toBe(0);

    // Starting again resumes from what it had applied.
    reader.start();
    expect(reads).toHaveLength(2);
    expect(reads[1]!.cursor).toBeNull();
  });

  it('pauses while the document is hidden and reads again as soon as it is shown', async () => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));

    page.setHidden(true);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(reads).toHaveLength(1);

    page.setHidden(false);
    expect(reads).toHaveLength(2);
    expect(reads[1]!.cursor).toBe('c1');
  });

  it('clears and ignores a delayed response once its account lifetime ends', async () => {
    const reader = createReader();
    reader.start();
    await settle(0, response('c1', { reset: 'initial', rows: [row(1)] }));
    await vi.advanceTimersToNextTimerAsync();

    lifetime.abort();
    expect(reads[1]!.signal.aborted).toBe(true);
    expect(reader.getSnapshot().transcript.rowCount).toBe(0);

    await settle(1, response('c2', { rows: [row(2)] }));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(reader.getSnapshot().transcript.rowCount).toBe(0);
    expect(reads).toHaveLength(2);
  });
});
