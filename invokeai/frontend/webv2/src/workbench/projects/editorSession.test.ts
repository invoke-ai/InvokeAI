import type { ExclusiveLockResult } from '@platform/browser/webLocks';

import { describe, expect, it, vi } from 'vitest';

import {
  createEditorSessionLivenessCheck,
  createEditorSessionProvider,
  EDITOR_SESSION_STORAGE_KEY,
  isEditorSessionLive,
} from './editorSession';

const createStorage = (initial?: string) => {
  const values = new Map<string, string>();
  if (initial) {
    values.set(EDITOR_SESSION_STORAGE_KEY, initial);
  }
  return {
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => values.set(key, value)),
  };
};

const createLockPort = () => {
  const held = new Set<string>();
  return vi.fn<(name: string) => Promise<ExclusiveLockResult>>((name) => {
    if (held.has(name)) {
      return Promise.resolve({ kind: 'contended' });
    }
    held.add(name);
    return Promise.resolve({
      kind: 'acquired',
      release: () => {
        held.delete(name);
        return Promise.resolve();
      },
    });
  });
};

const instant = (): Promise<void> => Promise.resolve();

describe('editor session identity', () => {
  it('keeps the persisted session when its lock is let go within the retries, as on a reload', async () => {
    const waits: number[] = [];
    let attempts = 0;
    const acquireLock = vi.fn<(name: string) => Promise<ExclusiveLockResult>>(() => {
      attempts += 1;
      return Promise.resolve(
        attempts < 3 ? { kind: 'contended' } : { kind: 'acquired', release: () => Promise.resolve() }
      );
    });
    const storage = createStorage('previous-life');
    const provider = createEditorSessionProvider(
      storage,
      acquireLock,
      () => 'fresh',
      (ms) => {
        waits.push(ms);
        return Promise.resolve();
      }
    );

    const session = await provider();

    expect(session.id).toBe('previous-life');
    expect(waits).toEqual([100, 100]);
    expect(acquireLock.mock.calls.map(([name]) => name)).toEqual(
      Array(3).fill(expect.stringContaining('previous-life'))
    );
  });

  it('takes a session of its own once the persisted one stays held, as in a duplicated tab', async () => {
    const waits: number[] = [];
    const acquireLock = vi.fn<(name: string) => Promise<ExclusiveLockResult>>((name) =>
      Promise.resolve(
        name.endsWith('held-elsewhere') ? { kind: 'contended' } : { kind: 'acquired', release: () => Promise.resolve() }
      )
    );
    const provider = createEditorSessionProvider(
      createStorage('held-elsewhere'),
      acquireLock,
      () => 'own',
      (ms) => {
        waits.push(ms);
        return Promise.resolve();
      }
    );

    const session = await provider();

    expect(session.id).toBe('own');
    expect(waits).toHaveLength(5);
    expect(acquireLock).toHaveBeenCalledTimes(7);
  });

  it('reuses an unclaimed persisted identity across a reload', async () => {
    const storage = createStorage('session-a');
    const acquireLock = createLockPort();
    const provider = createEditorSessionProvider(storage, acquireLock, () => 'unused');

    const session = await provider();

    expect(session.id).toBe('session-a');
    const sameTab = await provider();
    expect(sameTab.id).toBe('session-a');
    await session.release();
    await sameTab.release();
  });

  it('keeps the lock while any holder in the tab still uses the session', async () => {
    const acquireLock = createLockPort();
    const provider = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'rotated', instant);
    const exitingEditor = await provider();
    const nextEditor = await provider();

    await exitingEditor.release();
    const duplicatedTab = await createEditorSessionProvider(
      createStorage('copied'),
      acquireLock,
      () => 'dup',
      instant
    )();
    expect(nextEditor.id).toBe('copied');
    expect(duplicatedTab.id).toBe('dup');

    await nextEditor.release();
    const afterLastRelease = await createEditorSessionProvider(
      createStorage('copied'),
      acquireLock,
      () => 'x',
      instant
    )();
    expect(afterLastRelease.id).toBe('copied');
    await duplicatedTab.release();
    await afterLastRelease.release();
  });

  it('rotates a copied identity when another live tab holds its claim', async () => {
    const acquireLock = createLockPort();
    const first = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'session-a', instant);
    const second = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'session-b', instant);

    const firstSession = await first();
    const secondSession = await second();

    expect(firstSession.id).toBe('copied');
    expect(secondSession.id).toBe('session-b');
    await firstSession.release();
    await secondSession.release();
  });

  it('uses a fresh page identity when locking is unavailable', async () => {
    const storage = createStorage('copied');
    const acquireLock = vi.fn(() => Promise.resolve({ kind: 'unavailable' as const }));
    const provider = createEditorSessionProvider(storage, acquireLock, () => 'fallback');

    const session = await provider();

    expect(session.id).toBe('fallback');
    expect(storage.setItem).toHaveBeenCalledWith(EDITOR_SESSION_STORAGE_KEY, 'fallback');
    await session.release();
  });

  it('reclaims after release instead of returning an unlocked cached identity', async () => {
    const acquireLock = createLockPort();
    const storage = createStorage('copied');
    const provider = createEditorSessionProvider(storage, acquireLock, () => 'rotated', instant);
    const first = await provider();
    await first.release();
    const peer = await createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'peer', instant)();

    const reclaimed = await provider();

    expect(reclaimed.id).toBe('rotated');
    await peer.release();
    await reclaimed.release();
  });

  it('stops publishing a session as soon as release begins', async () => {
    let finishRelease: () => void = () => undefined;
    const released = new Promise<void>((resolve) => {
      finishRelease = resolve;
    });
    let call = 0;
    const acquireLock = vi.fn<(name: string) => Promise<ExclusiveLockResult>>(() => {
      call += 1;
      if (call === 1) {
        return Promise.resolve({ kind: 'acquired', release: () => released });
      }
      if (call === 2) {
        return Promise.resolve({ kind: 'contended' });
      }
      return Promise.resolve({ kind: 'acquired', release: () => Promise.resolve() });
    });
    const provider = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'rotated', instant);
    const first = await provider();

    const releasing = first.release();
    const replacement = await provider();

    expect(replacement).not.toBe(first);
    // A new claim, which re-takes the page's own session once the released one has let go.
    expect(replacement.id).toBe('copied');
    finishRelease();
    await releasing;
    await replacement.release();
  });

  it('reads whether a page still holds a session from its lock, and assumes none when it cannot tell', async () => {
    const queried: string[] = [];
    const queryLock = (name: string) => {
      queried.push(name);
      return Promise.resolve(name.endsWith(':held') ? true : name.endsWith(':free') ? false : null);
    };

    await expect(isEditorSessionLive('held', queryLock)).resolves.toBe(true);
    await expect(isEditorSessionLive('free', queryLock)).resolves.toBe(false);
    await expect(isEditorSessionLive('unknown', queryLock)).resolves.toBe(false);
    expect(queried).toEqual(['held', 'free', 'unknown'].map((id) => `invokeai:v7:webv2:editor-session:${id}`));
  });

  it("never takes this page's own session for another page's, and asks about each other session once", async () => {
    const asked: string[] = [];
    const isOtherPageLive = createEditorSessionLivenessCheck('own', (editorSessionId) => {
      asked.push(editorSessionId);
      return Promise.resolve(true);
    });

    await expect(isOtherPageLive('own')).resolves.toBe(false);
    await expect(isOtherPageLive('other')).resolves.toBe(true);
    await expect(isOtherPageLive('other')).resolves.toBe(true);
    expect(asked).toEqual(['other']);
  });
});
