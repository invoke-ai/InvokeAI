import type { GalleryImage } from '@features/gallery';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { requestCanvasStartImage, subscribeCanvasStartImage, takeCanvasStartImage } from './canvasStartImage';

const image: GalleryImage = {
  boardId: 'none',
  height: 750,
  imageCategory: 'user',
  imageName: 'pasted.png',
  imageUrl: '/images/pasted.png',
  queuedAt: '',
  sourceQueueItemId: '',
  starred: false,
  thumbnailUrl: '/thumbs/pasted.png',
  width: 1000,
};

beforeEach(() => {
  accountLifecycle.activate('user-a', ':user:user-a');
});

afterEach(() => {
  accountLifecycle.invalidate();
});

describe('canvasStartImage', () => {
  it('hands the image to its project once and to no other project', () => {
    requestCanvasStartImage({ image, projectId: 'draft' });

    expect(takeCanvasStartImage('other')).toBeNull();
    expect(takeCanvasStartImage('draft')).toBe(image);
    expect(takeCanvasStartImage('draft')).toBeNull();
  });

  it('notifies subscribers so a widget already mounted can place the image', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeCanvasStartImage(listener);

    requestCanvasStartImage({ image, projectId: 'draft' });

    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it('drops a staged image when the account changes', () => {
    requestCanvasStartImage({ image, projectId: 'draft' });

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b', ':user:user-b');

    expect(takeCanvasStartImage('draft')).toBeNull();
  });
});
