import { describe, expect, it } from 'vitest';

import { resolveDefaultFilterForModel } from './controlRecommendations';

describe('resolveDefaultFilterForModel', () => {
  it.each([
    ['canny_edge_detection', 'canny_edge_detection'],
    ['canny_image_processor', 'canny_edge_detection'],
    ['normalbae_image_processor', 'normal_map'],
    ['zoe_depth_image_processor', 'depth_anything_depth_estimation'],
    ['lineart_anime_image_processor', 'lineart_anime_edge_detection'],
  ])('resolves current ids and legacy alias %s', (preprocessor, expected) => {
    expect(resolveDefaultFilterForModel({ default_settings: { preprocessor } })).toBe(expected);
  });

  it('returns null for missing or unknown metadata', () => {
    expect(resolveDefaultFilterForModel(null)).toBeNull();
    expect(resolveDefaultFilterForModel({ default_settings: null })).toBeNull();
    expect(resolveDefaultFilterForModel({ default_settings: { preprocessor: 'future_processor' } })).toBeNull();
  });
});
