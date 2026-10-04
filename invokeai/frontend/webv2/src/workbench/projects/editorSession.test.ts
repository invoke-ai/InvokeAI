import type { ExclusiveLockResult } from '@platform/browser/webLocks';

import { describe, expect, it, vi } from 'vitest';

import { createEditorSessionProvider, EDITOR_SESSION_STORAGE_KEY } from './editorSession';

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

describe('editor session identity', () => {
  it('reuses an unclaimed persisted identity across a reload', async () => {
    const storage = createStorage('session-a');
    const acquireLock = createLockPort();
    const provider = createEditorSessionProvider(storage, acquireLock, () => 'unused');

    const session = await provider();

    expect(session.id).toBe('session-a');
    expect(await provider()).toBe(session);
    await session.release();
  });

  it('rotates a copied identity when another live tab holds its claim', async () => {
    const acquireLock = createLockPort();
    const first = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'session-a');
    const second = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'session-b');

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
    const provider = createEditorSessionProvider(storage, acquireLock, () => 'rotated');
    const first = await provider();
    await first.release();
    const peer = await createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'peer')();

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
    const provider = createEditorSessionProvider(createStorage('copied'), acquireLock, () => 'rotated');
    const first = await provider();

    const releasing = first.release();
    const replacement = await provider();

    expect(replacement).not.toBe(first);
    expect(replacement.id).toBe('rotated');
    finishRelease();
    await releasing;
    await replacement.release();
  });
});
