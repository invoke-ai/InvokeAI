import type { GalleryVideoItem } from '@features/gallery/contracts';

import { getGenerationSelectedGalleryImage } from '@app/GenerationUiAdapter';
import { getWorkflowSelectedGalleryImage } from '@features/workflow/ui/fields/WorkflowFieldInput';
import { describe, expect, it } from 'vitest';

const selectedVideo: GalleryVideoItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  durationSeconds: 3,
  fullUrl: '/videos/selected',
  height: 720,
  isIntermediate: false,
  kind: 'video',
  name: 'selected',
  starred: false,
  thumbnailUrl: '/videos/selected/thumbnail',
  width: 1280,
};

describe('direct image-only Gallery consumers', () => {
  it('keeps a selected video out of GenerationUiAdapter', () => {
    expect(getGenerationSelectedGalleryImage({ selectedImage: selectedVideo })).toBeNull();
  });

  it('keeps a selected video out of WorkflowFieldInput', () => {
    expect(getWorkflowSelectedGalleryImage({ selectedImage: selectedVideo })).toBeNull();
  });
});
