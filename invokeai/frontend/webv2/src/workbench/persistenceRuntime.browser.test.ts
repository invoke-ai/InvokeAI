import { describe, expect, it, vi } from 'vitest';

import { browserPageLifecycle } from './persistenceRuntime';

const unload = (): boolean => !window.dispatchEvent(new Event('beforeunload', { cancelable: true }));

describe('browser page lifecycle', () => {
  it('checkpoints before pagehide without prompting, and removes every lifecycle listener on disposal', () => {
    const checkpoints: string[] = [];
    let signal = 'beforeunload';
    let visibility: DocumentVisibilityState = 'visible';
    const visibilityState = vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility);
    const unsubscribe = browserPageLifecycle.subscribeHidden(() => checkpoints.push(signal));
    const unsubscribeUnload = browserPageLifecycle.subscribeBeforeUnload(() => {
      checkpoints.push(signal);
      return false;
    });
    try {
      expect(unload()).toBe(false);
      expect(checkpoints).toEqual(['beforeunload']);
      signal = 'pagehide';
      window.dispatchEvent(new Event('pagehide'));
      expect(checkpoints).toEqual(['beforeunload', 'pagehide']);
      signal = 'hidden';
      document.dispatchEvent(new Event('visibilitychange'));
      expect(checkpoints).toEqual(['beforeunload', 'pagehide']);
      visibility = 'hidden';
      document.dispatchEvent(new Event('visibilitychange'));
      expect(checkpoints).toEqual(['beforeunload', 'pagehide', 'hidden']);
      unsubscribe();
      unsubscribeUnload();
      signal = 'after disposal';
      expect(unload()).toBe(false);
      window.dispatchEvent(new Event('pagehide'));
      document.dispatchEvent(new Event('visibilitychange'));
      expect(checkpoints).toEqual(['beforeunload', 'pagehide', 'hidden']);
    } finally {
      unsubscribe();
      unsubscribeUnload();
      visibilityState.mockRestore();
    }
  });

  it('cancels `beforeunload`, so the browser asks, only while the runtime says leaving would lose changes', () => {
    let wouldLose = false;
    const unsubscribe = browserPageLifecycle.subscribeBeforeUnload(() => wouldLose);

    expect(unload()).toBe(false);
    wouldLose = true;
    expect(unload()).toBe(true);

    unsubscribe();
    expect(unload()).toBe(false);
  });
});
