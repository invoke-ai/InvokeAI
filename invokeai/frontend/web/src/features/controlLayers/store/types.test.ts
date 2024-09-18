import type {
  CannyEdgeDetectionFilterConfig,
  ColorMapFilterConfig,
  ContentShuffleFilterConfig,
  DepthAnythingFilterConfig,
  DepthAnythingModelSize,
  DWOpenposeDetectionFilterConfig,
  FilterConfig,
  FilterType,
  HEDEdgeDetectionFilterConfig,
  LineartAnimeEdgeDetectionFilterConfig,
  LineartEdgeDetectionFilterConfig,
  MediaPipeFaceDetectionFilterConfig,
  MLSDDetectionFilterConfig,
  NormalMapFilterConfig,
  PiDiNetEdgeDetectionFilterConfig,
} from 'features/controlLayers/store/filters';
import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';
import type { z } from 'zod';

import type {
  CanvasEntityIdentifier,
  CLIPVisionModelV2,
  ControlModeV2,
  IPMethodV2,
  zCanvasEntityIdentifer,
} from './types';

describe('Control Adapter Types', () => {
  test('FilterType', () => {
    // FilterType is a union of all filter types. FilterConfig is inferred from a zod union. zod does not support
    // extracting a specific field from a union, so FilterType is defined separately. To ensure that FilterType is
    // consistent with FilterConfig['type'], we compare the two types.
    assert<Equals<FilterConfig['type'], FilterType>>();
  });
  test('IP Adapter Method', () => {
    // This ensures the manually defined IPMethodV2 type is consistent with the type we get from the API.
    assert<Equals<NonNullable<Invocation<'ip_adapter'>['method']>, IPMethodV2>>();
  });
  test('CLIP Vision Model', () => {
    // This ensures the manually defined CLIPVisionModelV2 type is consistent with the type we get from the API.
    assert<Equals<NonNullable<Invocation<'ip_adapter'>['clip_vision_model']>, CLIPVisionModelV2>>();
  });
  test('Control Mode', () => {
    // This ensures the manually defined ControlModeV2 type is consistent with the type we get from the API.
    assert<Equals<NonNullable<Invocation<'controlnet'>['control_mode']>, ControlModeV2>>();
  });
  test('DepthAnything Model Size', () => {
    // This ensures the manually defined DepthAnythingModelSize type is consistent with the type we get from the API.
    assert<Equals<NonNullable<Invocation<'depth_anything_depth_estimation'>['model_size']>, DepthAnythingModelSize>>();
  });
  test('Processor Configs', () => {
    // Types derived from OpenAPI
    type _CannyEdgeDetectionFilterConfig = Required<
      Pick<Invocation<'canny_edge_detection'>, 'type' | 'low_threshold' | 'high_threshold'>
    >;
    type _ColorMapFilterConfig = Required<Pick<Invocation<'color_map'>, 'type' | 'tile_size'>>;
    type _ContentShuffleFilterConfig = Required<Pick<Invocation<'content_shuffle'>, 'type' | 'scale_factor'>>;
    type _DepthAnythingFilterConfig = Required<
      Pick<Invocation<'depth_anything_depth_estimation'>, 'type' | 'model_size'>
    >;
    type _HEDEdgeDetectionFilterConfig = Required<Pick<Invocation<'hed_edge_detection'>, 'type' | 'scribble'>>;
    type _LineartAnimeEdgeDetectionFilterConfig = Required<Pick<Invocation<'lineart_anime_edge_detection'>, 'type'>>;
    type _LineartEdgeDetectionFilterConfig = Required<Pick<Invocation<'lineart_edge_detection'>, 'type' | 'coarse'>>;
    type _MediaPipeFaceDetectionFilterConfig = Required<
      Pick<Invocation<'mediapipe_face_detection'>, 'type' | 'max_faces' | 'min_confidence'>
    >;
    type _MLSDDetectionFilterConfig = Required<
      Pick<Invocation<'mlsd_detection'>, 'type' | 'score_threshold' | 'distance_threshold'>
    >;
    type _NormalMapFilterConfig = Required<Pick<Invocation<'normal_map'>, 'type'>>;
    type _DWOpenposeDetectionFilterConfig = Required<
      Pick<Invocation<'dw_openpose_detection'>, 'type' | 'draw_body' | 'draw_face' | 'draw_hands'>
    >;
    type _PiDiNetEdgeDetectionFilterConfig = Required<
      Pick<Invocation<'pidi_edge_detection'>, 'type' | 'quantize_edges' | 'scribble'>
    >;

    // The processor configs are manually modeled zod schemas. This test ensures that the inferred types are correct.
    // The types prefixed with `_` are types generated from OpenAPI, while the types without the prefix are manually modeled.
    assert<Equals<_CannyEdgeDetectionFilterConfig, CannyEdgeDetectionFilterConfig>>();
    assert<Equals<_ColorMapFilterConfig, ColorMapFilterConfig>>();
    assert<Equals<_ContentShuffleFilterConfig, ContentShuffleFilterConfig>>();
    assert<Equals<_DepthAnythingFilterConfig, DepthAnythingFilterConfig>>();
    assert<Equals<_HEDEdgeDetectionFilterConfig, HEDEdgeDetectionFilterConfig>>();
    assert<Equals<_LineartAnimeEdgeDetectionFilterConfig, LineartAnimeEdgeDetectionFilterConfig>>();
    assert<Equals<_LineartEdgeDetectionFilterConfig, LineartEdgeDetectionFilterConfig>>();
    assert<Equals<_MediaPipeFaceDetectionFilterConfig, MediaPipeFaceDetectionFilterConfig>>();
    assert<Equals<_MLSDDetectionFilterConfig, MLSDDetectionFilterConfig>>();
    assert<Equals<_NormalMapFilterConfig, NormalMapFilterConfig>>();
    assert<Equals<_DWOpenposeDetectionFilterConfig, DWOpenposeDetectionFilterConfig>>();
    assert<Equals<_PiDiNetEdgeDetectionFilterConfig, PiDiNetEdgeDetectionFilterConfig>>();
  });
  test('CanvasEntityIdentifier', () => {
    // The generic type `CanvasEntityIdentifier` is defined manually, but it must be equal to the inferred type from
    // the zod schema.
    assert<Equals<CanvasEntityIdentifier, z.infer<typeof zCanvasEntityIdentifer>>>();
  });
});
