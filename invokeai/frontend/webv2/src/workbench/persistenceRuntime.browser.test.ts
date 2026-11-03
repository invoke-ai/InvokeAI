import { describe, expect, it } from 'vitest';

import { browserPageLifecycle } from './persistenceRuntime';

const unload = (): boolean => !window.dispatchEvent(new Event('beforeunload', { cancelable: true }));

describe('browser page lifecycle', () => {
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
