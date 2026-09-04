import type { FalCatalogModel } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { isFalNativeCanvasModel } from './falCatalog';

const model = (kind: FalCatalogModel['kind']): FalCatalogModel => ({
  endpoint_id: 'fal-ai/test',
  display_name: 'Test model',
  description: '',
  category: kind,
  kind,
  model_url: null,
  thumbnail_url: null,
  tags: [],
  installed: false,
});

describe('fal catalog model capabilities', () => {
  it('marks image endpoints as native Canvas-compatible', () => {
    expect(isFalNativeCanvasModel(model('text-to-image'))).toBe(true);
    expect(isFalNativeCanvasModel(model('image-to-image'))).toBe(true);
    expect(isFalNativeCanvasModel(model('inpaint'))).toBe(true);
    expect(isFalNativeCanvasModel(model('upscale'))).toBe(true);
  });

  it('keeps video and generic endpoints out of image Canvas picker', () => {
    expect(isFalNativeCanvasModel(model('text-to-video'))).toBe(false);
    expect(isFalNativeCanvasModel(model('image-to-video'))).toBe(false);
    expect(isFalNativeCanvasModel(model('generic'))).toBe(false);
  });
});
