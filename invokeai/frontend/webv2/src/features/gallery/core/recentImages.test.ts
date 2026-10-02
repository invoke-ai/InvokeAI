import { describe, expect, it } from 'vitest';

import type { GeneratedImageContract } from './types';

import { getBoundedRecentImages } from './recentImages';

describe('getBoundedRecentImages', () => {
  it('retains the original generated-image object and its gallery context by reference', () => {
    const image: GeneratedImageContract & { boardId: string; imageCategory: 'general'; starred: boolean } = {
      boardId: 'board-1',
      height: 512,
      imageCategory: 'general',
      imageName: 'recent.png',
      imageUrl: '/images/recent.png/full',
      queuedAt: '2026-07-30T12:00:00.000Z',
      sourceQueueItemId: 'queue-123',
      starred: true,
      thumbnailUrl: '/images/recent.png/thumbnail',
      width: 768,
    };

    const [recent] = getBoundedRecentImages([image]);

    expect(recent).toBe(image);
  });
});
