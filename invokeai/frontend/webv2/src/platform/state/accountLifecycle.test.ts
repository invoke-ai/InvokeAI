import { describe, expect, it, vi } from 'vitest';

import { createAccountLifecycle } from './accountLifecycle';

describe('account lifecycle', () => {
  it('rotates immutable epochs for every authenticated lifetime, including the same account', () => {
    const lifecycle = createAccountLifecycle();
    const boot = lifecycle.capture();
    const first = lifecycle.activate('user-a');

    expect(first).toMatchObject({ accountId: 'user-a', epoch: 1, storageSuffix: '' });
    expect(lifecycle.isCurrent(boot)).toBe(false);
    expect(boot.signal.aborted).toBe(true);

    lifecycle.invalidate();
    const second = lifecycle.activate('user-a', ':user:user-a');

    expect(second).toMatchObject({ accountId: 'user-a', epoch: 3, storageSuffix: ':user:user-a' });
    expect(lifecycle.isCurrent(first)).toBe(false);
    expect(first.signal.aborted).toBe(true);
    expect(lifecycle.isCurrent(second)).toBe(true);
    expect(second.signal.aborted).toBe(false);
    expect(Object.isFrozen(second)).toBe(true);
  });

  it('invalidates before clearing resources and keeps clearing after one resource fails', () => {
    const lifecycle = createAccountLifecycle();
    const observed: boolean[] = [];
    const first = lifecycle.activate('user-a');
    const finalClear = vi.fn();

    lifecycle.register({
      clear: () => {
        observed.push(lifecycle.isCurrent(first));
      },
      name: 'observer',
    });
    lifecycle.register({
      clear: () => {
        throw new Error('broken clear');
      },
      name: 'broken',
    });
    lifecycle.register({ clear: finalClear, name: 'final' });

    lifecycle.invalidate();

    expect(observed).toEqual([false]);
    expect(finalClear).toHaveBeenCalledOnce();
    expect(lifecycle.capture()).toMatchObject({ accountId: null, epoch: 2, storageSuffix: '' });
  });

  it('unregisters lifetime-owned resources', () => {
    const lifecycle = createAccountLifecycle();
    const clear = vi.fn();
    const unregister = lifecycle.register({ clear, name: 'ephemeral' });

    unregister();
    lifecycle.invalidate();

    expect(clear).not.toHaveBeenCalled();
  });

  it('replaces same-name registrations without letting an old disposer remove the replacement', () => {
    const lifecycle = createAccountLifecycle();
    const oldClear = vi.fn();
    const newClear = vi.fn();
    const disposeOld = lifecycle.register({ clear: oldClear, name: 'hot-resource' });

    lifecycle.register({ clear: newClear, name: 'hot-resource' });
    disposeOld();
    lifecycle.invalidate();

    expect(oldClear).not.toHaveBeenCalled();
    expect(newClear).toHaveBeenCalledOnce();
  });

  it('rejects empty active identities', () => {
    const lifecycle = createAccountLifecycle();

    expect(() => lifecycle.activate('')).toThrow('non-empty account id');
  });
});
