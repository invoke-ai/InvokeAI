import { describe, expect, it } from 'vitest';

import { imageMakeDurableChanges, imageSaveToGalleryChanges } from './api';

describe('imageSaveToGalleryChanges', () => {
  it('promotes an intermediate candidate to a durable, gallery-visible image', () => {
    // The backend `ImageRecordChanges` body: clear `is_intermediate` (stop GC)
    // and set `image_category: 'general'` (surface in the gallery images view).
    expect(imageSaveToGalleryChanges()).toEqual({ image_category: 'general', is_intermediate: false });
  });
});

describe('imageMakeDurableChanges', () => {
  it('clears is_intermediate WITHOUT touching image_category (durable but not in gallery)', () => {
    // Applying a control-filter preview makes the chosen image the layer's new
    // source: it must survive GC (is_intermediate: false) but stay out of the
    // gallery images view (no image_category change).
    const body = imageMakeDurableChanges();
    expect(body).toEqual({ is_intermediate: false });
    expect('image_category' in body).toBe(false);
  });
});
