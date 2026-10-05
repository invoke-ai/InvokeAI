import { describe, expect, it } from 'vitest';

import { acquireExclusiveLock } from './webLocks';

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
});
