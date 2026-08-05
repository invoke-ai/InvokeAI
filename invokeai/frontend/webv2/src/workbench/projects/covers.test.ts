import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as coversModule from './covers';

/**
 * The cover index: a per-user blob that lets the library grid show thumbnails
 * without fetching every project's document. It is an index, not a truth, so
 * every failure here has to degrade to "no cover" rather than to an error.
 */

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

beforeEach(async () => {
  vi.resetModules();
  api.__clientState.clear();
  api.getClientStateValue.mockClear();
  api.setClientStateValue.mockClear();

  covers = await import('./covers');
});

describe('parseProjectCovers', () => {
  it('reads a well-formed blob', () => {
    expect(covers.parseProjectCovers('{"p1":"a.png"}')).toEqual({ p1: 'a.png' });
  });

  it('treats unreadable or wrongly-shaped blobs as empty', () => {
    expect(covers.parseProjectCovers(null)).toEqual({});
    expect(covers.parseProjectCovers('')).toEqual({});
    expect(covers.parseProjectCovers('not json')).toEqual({});
    expect(covers.parseProjectCovers('[]')).toEqual({});
    expect(covers.parseProjectCovers('"a string"')).toEqual({});
  });

  it('drops entries that are not non-empty strings', () => {
    expect(covers.parseProjectCovers('{"p1":"a.png","p2":7,"p3":null,"p4":""}')).toEqual({ p1: 'a.png' });
  });
});

describe('loadProjectCovers', () => {
  it('reads the index once and shares concurrent calls', async () => {
    api.__clientState.set('webv2:project-covers', '{"p1":"a.png"}');

    await Promise.all([covers.loadProjectCovers(), covers.loadProjectCovers()]);

    expect(api.getClientStateValue).toHaveBeenCalledTimes(1);
    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
  });

  it('degrades to no covers when the backend is unreachable', async () => {
    api.getClientStateValue.mockRejectedValueOnce(new Error('offline'));

    await expect(covers.loadProjectCovers()).resolves.toBeUndefined();
    expect(covers.getProjectCoverImageName('p1')).toBeUndefined();
  });
});

describe('recordProjectCover', () => {
  it('persists a new cover and reads it back', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');

    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({ p1: 'a.png' });
  });

  /** Autosave runs constantly; a cover changes when a generation lands. */
  it('does not write when nothing changed', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');
    api.setClientStateValue.mockClear();
    covers.recordProjectCover('p1', 'a.png');

    expect(api.setClientStateValue).not.toHaveBeenCalled();
  });

  it('clears an entry when a project loses its cover', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');
    covers.recordProjectCover('p1', null);

    expect(covers.getProjectCoverImageName('p1')).toBeUndefined();
    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({});
  });

  it('notifies subscribers so the library can re-derive', async () => {
    await covers.loadProjectCovers();

    const listener = vi.fn();
    const unsubscribe = covers.subscribeProjectCovers(listener);

    covers.recordProjectCover('p1', 'a.png');
    unsubscribe();
    covers.recordProjectCover('p2', 'b.png');

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('survives a failed write without throwing at the caller', async () => {
    await covers.loadProjectCovers();
    api.setClientStateValue.mockRejectedValueOnce(new Error('offline'));

    expect(() => covers.recordProjectCover('p1', 'a.png')).not.toThrow();
    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
  });
});

describe('forgetProjectCover', () => {
  it('drops a deleted project so the blob does not accumulate dead ids', async () => {
    await covers.loadProjectCovers();
    covers.recordProjectCover('p1', 'a.png');
    covers.forgetProjectCover('p1');

    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!)).toEqual({});
  });
});

describe('getProjectCoverUrl', () => {
  it('points at the thumbnail route and escapes the name', () => {
    expect(covers.getProjectCoverUrl('a b.png')).toContain('/api/v1/images/i/a%20b.png/thumbnail');
  });
});
