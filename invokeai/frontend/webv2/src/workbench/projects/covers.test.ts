import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as coversModule from './covers';

/** Cover-index failure degrades to missing thumbnails. */

const api = vi.hoisted(() => {
  const clientState = new Map<string, string>();

  return {
    __clientState: clientState,
    getClientStateValue: vi.fn((key: string) => Promise.resolve(clientState.get(key) ?? null)),
    setClientStateValue: vi.fn((key: string, value: string) => {
      clientState.set(key, value);

      return Promise.resolve();
    }),
  };
});

vi.mock('./api', () => api);

let covers: typeof coversModule;
let lifecycle: typeof accountLifecycleModule;

const createDeferred = <T = void>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

beforeEach(async () => {
  vi.resetModules();
  api.__clientState.clear();
  vi.resetAllMocks();

  lifecycle = await import('@platform/state/accountLifecycle');
  covers = await import('./covers');
});

describe('parseProjectCovers', () => {
  it('treats unreadable or wrongly-shaped blobs as empty', () => {
    expect(covers.parseProjectCovers(null)).toEqual({});
    expect(covers.parseProjectCovers('')).toEqual({});
    expect(covers.parseProjectCovers('not json')).toEqual({});
    expect(covers.parseProjectCovers('[]')).toEqual({});
    expect(covers.parseProjectCovers('"a string"')).toEqual({});
  });
});

describe('loadProjectCovers', () => {
  it('reads the index once and shares concurrent calls', async () => {
    api.__clientState.set('webv2:project-covers', '{"p1":"a.png"}');

    await Promise.all([covers.loadProjectCovers(), covers.loadProjectCovers()]);

    expect(api.getClientStateValue).toHaveBeenCalledTimes(1);
    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
  });

  it('does not let a later load overwrite a cover mutation after the account index loaded', async () => {
    const staleRead = createDeferred<string | null>();

    api.getClientStateValue.mockResolvedValueOnce('{"p1":"old.png"}').mockImplementationOnce(() => staleRead.promise);

    await covers.loadProjectCovers();

    const laterLoad = covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'new.png');
    staleRead.resolve('{"p1":"old.png"}');
    await laterLoad;

    await vi.waitFor(() => {
      expect(covers.getProjectCoverImageName('p1')).toBe('new.png');
      expect(api.__clientState.get('webv2:project-covers')).toBe('{"p1":"new.png"}');
    });
    expect(api.getClientStateValue).toHaveBeenCalledTimes(1);
  });
});

/** Do not replace the KV blob before the initial read succeeds. */
describe('recordProjectCover before the index has loaded', () => {
  it('shows the cover immediately without writing over an index it has not read', async () => {
    api.__clientState.set('webv2:project-covers', '{"other":"b.png"}');

    covers.recordProjectCover('p1', 'a.png');

    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
    expect(api.setClientStateValue).not.toHaveBeenCalled();

    // Drain triggered reads and writes before the test ends.
    await covers.loadProjectCovers();

    expect(api.setClientStateValue).toHaveBeenCalledTimes(1);
  });

  it('merges the pending record onto the server blob instead of replacing it', async () => {
    api.__clientState.set('webv2:project-covers', '{"other":"b.png"}');

    covers.recordProjectCover('p1', 'a.png');
    await covers.loadProjectCovers();

    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ other: 'b.png', p1: 'a.png' });
    expect(covers.getProjectCoverImageName('other')).toBe('b.png');
  });

  it('holds the pending record when the read fails rather than persisting a blank index', async () => {
    api.__clientState.set('webv2:project-covers', '{"other":"b.png"}');
    api.getClientStateValue.mockRejectedValueOnce(new Error('offline'));

    covers.recordProjectCover('p1', 'a.png');
    await covers.loadProjectCovers();

    expect(api.setClientStateValue).not.toHaveBeenCalled();
    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ other: 'b.png' });

    // The next successful read drains what was held.
    await covers.loadProjectCovers();

    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ other: 'b.png', p1: 'a.png' });
  });
});

describe('the tracked-cover cap', () => {
  /** Eviction is by insertion order, so the entry being written has to be re-inserted. */
  it('never evicts the project it was just asked to record', async () => {
    const initial = Object.fromEntries(Array.from({ length: 500 }, (_, index) => [`p${index}`, `${index}.png`]));

    api.__clientState.set('webv2:project-covers', JSON.stringify(initial));
    await covers.loadProjectCovers();

    // p0 is the oldest entry, so a naive slice would drop the very write below.
    covers.recordProjectCover('p0', 'new.png');

    expect(covers.getProjectCoverImageName('p0')).toBe('new.png');
    await vi.waitFor(() => {
      expect(JSON.parse(api.__clientState.get('webv2:project-covers')!).p0).toBe('new.png');
    });
  });
});

describe('recordProjectCover', () => {
  it('persists a new cover and reads it back', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');

    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
    await vi.waitFor(() => {
      expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ p1: 'a.png' });
    });
  });

  it('serializes rapid mutations and persists the newest complete blob last', async () => {
    const firstWrite = createDeferred();
    const secondWrite = createDeferred();
    const completedWrites: string[] = [];

    api.setClientStateValue
      .mockImplementationOnce(async (key: string, value: string) => {
        await firstWrite.promise;
        api.__clientState.set(key, value);
        completedWrites.push(value);
      })
      .mockImplementationOnce(async (key: string, value: string) => {
        await secondWrite.promise;
        api.__clientState.set(key, value);
        completedWrites.push(value);
      });

    await covers.loadProjectCovers();
    const listener = vi.fn();
    const unsubscribe = covers.subscribeProjectCovers(listener);
    covers.recordProjectCover('p1', 'a.png');
    covers.recordProjectCover('p1', 'b.png');

    await vi.waitFor(() => expect(api.setClientStateValue).toHaveBeenCalledTimes(1));
    expect(JSON.parse(api.setClientStateValue.mock.calls[0]![1])).toEqual({ p1: 'a.png' });

    firstWrite.resolve();
    await vi.waitFor(() => expect(api.setClientStateValue).toHaveBeenCalledTimes(2));
    expect(JSON.parse(api.setClientStateValue.mock.calls[1]![1])).toEqual({ p1: 'b.png' });

    secondWrite.resolve();
    await vi.waitFor(() => {
      expect(completedWrites.map((value) => JSON.parse(value))).toEqual([{ p1: 'a.png' }, { p1: 'b.png' }]);
      expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ p1: 'b.png' });
      expect(listener).toHaveBeenCalledTimes(3);
    });
    unsubscribe();
  });

  it('discards queued old-account writes and ignores stale completions', async () => {
    const oldAccountWrite = createDeferred();
    const newAccountWrite = createDeferred();
    let didCompleteOldWrite = false;
    let didFailNewWrite = false;

    api.setClientStateValue
      .mockImplementationOnce(async (key: string, value: string) => {
        await oldAccountWrite.promise;
        api.__clientState.set(key, value);
        didCompleteOldWrite = true;
      })
      .mockImplementationOnce(async () => {
        await newAccountWrite.promise;
        didFailNewWrite = true;
        throw new Error('offline');
      });

    lifecycle.accountLifecycle.activate('user-a');
    await covers.loadProjectCovers();
    covers.recordProjectCover('old-project', 'old.png');
    await vi.waitFor(() => expect(api.setClientStateValue).toHaveBeenCalledTimes(1));
    covers.recordProjectCover('old-project-2', 'old-2.png');
    expect(api.setClientStateValue).toHaveBeenCalledTimes(1);

    lifecycle.accountLifecycle.activate('user-b');
    await covers.loadProjectCovers();
    const listener = vi.fn();
    const unsubscribe = covers.subscribeProjectCovers(listener);
    covers.recordProjectCover('new-project', 'new.png');
    await vi.waitFor(() => expect(api.setClientStateValue).toHaveBeenCalledTimes(2));

    oldAccountWrite.resolve();
    newAccountWrite.resolve();
    await vi.waitFor(() => {
      expect(didCompleteOldWrite).toBe(true);
      expect(didFailNewWrite).toBe(true);
      expect(api.setClientStateValue).toHaveBeenCalledTimes(2);
      expect(listener).toHaveBeenCalledTimes(1);
      expect(covers.getProjectCoverImageName('old-project')).toBeUndefined();
      expect(covers.getProjectCoverImageName('new-project')).toBe('new.png');
    });

    covers.recordProjectCover('new-project', 'new.png');
    await vi.waitFor(() => {
      expect(api.setClientStateValue).toHaveBeenCalledTimes(3);
      expect(JSON.parse(api.setClientStateValue.mock.calls[2]![1])).toEqual({ 'new-project': 'new.png' });
      expect(listener).toHaveBeenCalledTimes(3);
    });
    unsubscribe();
  });
});

describe('forgetProjectCover', () => {
  it('drops a deleted project so the blob does not accumulate dead ids', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');
    covers.forgetProjectCover('p1');

    await vi.waitFor(() => {
      expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({});
    });
  });
});
