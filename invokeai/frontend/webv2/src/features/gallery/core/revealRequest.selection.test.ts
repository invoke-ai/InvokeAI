import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getGalleryRevealRequest, requestGalleryItemReveal, subscribeGalleryRevealRequests } from './selection';

describe('gallery reveal requests', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-reveal-request-test');
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('notifies subscribers with a fresh token per request, even for the same item', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeGalleryRevealRequests(listener);

    requestGalleryItemReveal('image:a.png', accountLifecycle.capture().signal);
    const first = getGalleryRevealRequest();

    requestGalleryItemReveal('image:a.png', accountLifecycle.capture().signal);
    const second = getGalleryRevealRequest();

    expect(listener).toHaveBeenCalledTimes(2);
    expect(first?.itemKey).toBe('image:a.png');
    expect(second?.itemKey).toBe('image:a.png');
    // The token is what lets a repeated gesture on an unchanged selection
    // still read as a new reveal.
    expect(second?.token).not.toBe(first?.token);

    unsubscribe();
    requestGalleryItemReveal('image:b.png', accountLifecycle.capture().signal);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('preserves an optional absolute index for a verified deep reveal', () => {
    requestGalleryItemReveal('image:deep.png', accountLifecycle.capture().signal, 6073);

    expect(getGalleryRevealRequest()).toMatchObject({ absoluteIndex: 6073, itemKey: 'image:deep.png' });
  });

  it('hides an account-owned request when Gallery reads it after an account rotation', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeGalleryRevealRequests(listener);

    const accountA = accountLifecycle.activate('gallery-reveal-request-owner');
    requestGalleryItemReveal('image:account-a.png', accountA.signal);
    const request = getGalleryRevealRequest();

    accountLifecycle.activate('gallery-reveal-request-next-owner');

    expect(request).toMatchObject({ accountSignal: accountA.signal, itemKey: 'image:account-a.png' });
    expect(getGalleryRevealRequest()).toBeNull();

    requestGalleryItemReveal('image:account-b.png', accountLifecycle.capture().signal);

    expect(getGalleryRevealRequest()).toMatchObject({
      accountSignal: accountLifecycle.capture().signal,
      itemKey: 'image:account-b.png',
    });
    expect(getGalleryRevealRequest()?.token).toBeGreaterThan(request?.token ?? 0);
    expect(listener).toHaveBeenCalledTimes(2);

    unsubscribe();
  });
});
