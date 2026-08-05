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

/**
 * The index is one blob and every write replaces it, so a record made before
 * the first read must not reach the KV — it would delete every project's cover
 * but its own. Autosave reaches `recordProjectCover` from the editor, which
 * loads the index only when the project switcher opens, so this ordering is the
 * ordinary one for anyone who reloads straight into a project.
 */
describe('recordProjectCover before the index has loaded', () => {
  it('shows the cover immediately without writing over an index it has not read', async () => {
    api.__clientState.set('webv2:project-covers', '{"other":"b.png"}');

    covers.recordProjectCover('p1', 'a.png');

    expect(covers.getProjectCoverImageName('p1')).toBe('a.png');
    expect(api.setClientStateValue).not.toHaveBeenCalled();

    // Recording kicks off the read it was waiting for; drain it so the write
    // lands inside this test rather than after it.
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

  it('writes nothing when the pending record is what the server already said', async () => {
    api.__clientState.set('webv2:project-covers', '{"p1":"a.png"}');

    covers.recordProjectCover('p1', 'a.png');
    await covers.loadProjectCovers();

    expect(api.setClientStateValue).not.toHaveBeenCalled();
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
    expect(JSON.parse(api.__clientState.get('webv2:project-covers')!).p0).toBe('new.png');
  });

  it('drops the oldest entry when a new project pushes past the cap', async () => {
    const initial = Object.fromEntries(Array.from({ length: 500 }, (_, index) => [`p${index}`, `${index}.png`]));

    api.__clientState.set('webv2:project-covers', JSON.stringify(initial));
    await covers.loadProjectCovers();
    covers.recordProjectCover('fresh', 'new.png');

    const persisted = JSON.parse(api.__clientState.get('webv2:project-covers')!);

    expect(Object.keys(persisted)).toHaveLength(500);
    expect(persisted.fresh).toBe('new.png');
    expect(persisted.p0).toBeUndefined();
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
