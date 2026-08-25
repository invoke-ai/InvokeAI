import { describe, expect, it, vi } from 'vitest';

import { getGalleryRevealRequest, requestGalleryItemReveal, subscribeGalleryRevealRequests } from './selection';

describe('gallery reveal requests', () => {
  it('notifies subscribers with a fresh token per request, even for the same item', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeGalleryRevealRequests(listener);

    requestGalleryItemReveal('image:a.png');
    const first = getGalleryRevealRequest();

    requestGalleryItemReveal('image:a.png');
    const second = getGalleryRevealRequest();

    expect(listener).toHaveBeenCalledTimes(2);
    expect(first?.itemKey).toBe('image:a.png');
    expect(second?.itemKey).toBe('image:a.png');
    // The token is what lets a repeated gesture on an unchanged selection
    // still read as a new reveal.
    expect(second?.token).not.toBe(first?.token);

    unsubscribe();
    requestGalleryItemReveal('image:b.png');
    expect(listener).toHaveBeenCalledTimes(2);
  });
});
