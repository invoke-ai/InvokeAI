import { describe, expect, it } from 'vitest';

import { acquireExclusiveLock, isLockHeld, waitForSharedLock } from './webLocks';

describe('Web Locks adapter', () => {
  it('holds one exclusive owner and releases it idempotently', async () => {
    const name = `invokeai:web-lock-test:${crypto.randomUUID()}`;
    const first = await acquireExclusiveLock(name);
    expect(first.kind).toBe('acquired');

    const second = await acquireExclusiveLock(name);
    expect(second).toEqual({ kind: 'contended' });

    if (first.kind === 'acquired') {
      await first.release();
      await first.release();
    }
    const third = await acquireExclusiveLock(name);
    expect(third.kind).toBe('acquired');
    if (third.kind === 'acquired') {
      await third.release();
    }
  });

  it('tells whether a lock is held without taking it', async () => {
    const name = `invokeai:web-lock-test:${crypto.randomUUID()}`;
    await expect(isLockHeld(name)).resolves.toBe(false);
    const held = await acquireExclusiveLock(name);

    await expect(isLockHeld(name)).resolves.toBe(true);
    // Asking did not queue a request that would contend with the holder or outlive it.
    if (held.kind === 'acquired') {
      await held.release();
    }
    await expect(isLockHeld(name)).resolves.toBe(false);
    const again = await acquireExclusiveLock(name);
    expect(again.kind).toBe('acquired');
    if (again.kind === 'acquired') {
      await again.release();
    }
  });

  it('waits for a held lock until its holder lets it go, and stops waiting once aborted', async () => {
    const name = `invokeai:web-lock-test:${crypto.randomUUID()}`;
    const held = await acquireExclusiveLock(name);
    const waiting = waitForSharedLock(name, new AbortController().signal);
    const abort = new AbortController();
    const abandoned = waitForSharedLock(name, abort.signal);
    let isGranted = false;
    void waiting.then(() => {
      isGranted = true;
    });

    abort.abort();
    await expect(abandoned).resolves.toEqual({ kind: 'unavailable' });
    await new Promise((resolve) => {
      setTimeout(resolve, 20);
    });
    expect(isGranted).toBe(false);
    if (held.kind === 'acquired') {
      await held.release();
    }
    const next = await waiting;
    expect(next.kind).toBe('acquired');
    if (next.kind === 'acquired') {
      await next.release();
    }
  });

  it('cannot tell without the Web Locks API', async () => {
    await expect(isLockHeld('invokeai:web-lock-test:none', {} as LockManager)).resolves.toBeNull();
  });
});
