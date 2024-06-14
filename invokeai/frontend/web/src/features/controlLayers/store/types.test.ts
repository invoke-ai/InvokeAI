import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

import type {
  CannyProcessorConfig,
  CLIPVisionModelV2,
  ColorMapProcessorConfig,
  ContentShuffleProcessorConfig,
  ControlModeV2,
  DepthAnythingModelSize,
  DepthAnythingProcessorConfig,
  DWOpenposeProcessorConfig,
  HedProcessorConfig,
  IPMethodV2,
  LineartAnimeProcessorConfig,
  LineartProcessorConfig,
  MediapipeFaceProcessorConfig,
  MidasDepthProcessorConfig,
  MlsdProcessorConfig,
  NormalbaeProcessorConfig,
  PidiProcessorConfig,
  ProcessorConfig,
  ProcessorTypeV2,
  ZoeDepthProcessorConfig,
} from './types';

describe('Control Adapter Types', () => {
  test('ProcessorType', () => {
    assert<Equals<ProcessorConfig['type'], ProcessorTypeV2>>();
  });
  test('IP Adapter Method', () => {
    assert<Equals<NonNullable<Invocation<'ip_adapter'>['method']>, IPMethodV2>>();
  });
  test('CLIP Vision Model', () => {
    assert<Equals<NonNullable<Invocation<'ip_adapter'>['clip_vision_model']>, CLIPVisionModelV2>>();
  });
  test('Control Mode', () => {
    assert<Equals<NonNullable<Invocation<'controlnet'>['control_mode']>, ControlModeV2>>();
  });
  test('DepthAnything Model Size', () => {
    assert<Equals<NonNullable<Invocation<'depth_anything_image_processor'>['model_size']>, DepthAnythingModelSize>>();
  });
  test('Processor Configs', () => {
    // The processor configs are manually modeled zod schemas. This test ensures that the inferred types are correct.
    // The types prefixed with `_` are types generated from OpenAPI, while the types without the prefix are manually modeled.
    assert<Equals<_CannyProcessorConfig, CannyProcessorConfig>>();
    assert<Equals<_ColorMapProcessorConfig, ColorMapProcessorConfig>>();
    assert<Equals<_ContentShuffleProcessorConfig, ContentShuffleProcessorConfig>>();
    assert<Equals<_DepthAnythingProcessorConfig, DepthAnythingProcessorConfig>>();
    assert<Equals<_HedProcessorConfig, HedProcessorConfig>>();
    assert<Equals<_LineartAnimeProcessorConfig, LineartAnimeProcessorConfig>>();
    assert<Equals<_LineartProcessorConfig, LineartProcessorConfig>>();
    assert<Equals<_MediapipeFaceProcessorConfig, MediapipeFaceProcessorConfig>>();
    assert<Equals<_MidasDepthProcessorConfig, MidasDepthProcessorConfig>>();
    assert<Equals<_MlsdProcessorConfig, MlsdProcessorConfig>>();
    assert<Equals<_NormalbaeProcessorConfig, NormalbaeProcessorConfig>>();
    assert<Equals<_DWOpenposeProcessorConfig, DWOpenposeProcessorConfig>>();
    assert<Equals<_PidiProcessorConfig, PidiProcessorConfig>>();
    assert<Equals<_ZoeDepthProcessorConfig, ZoeDepthProcessorConfig>>();
  });
});

// Types derived from OpenAPI
type _CannyProcessorConfig = Required<
  Pick<Invocation<'canny_image_processor'>, 'id' | 'type' | 'low_threshold' | 'high_threshold'>
>;
type _ColorMapProcessorConfig = Required<
  Pick<Invocation<'color_map_image_processor'>, 'id' | 'type' | 'color_map_tile_size'>
>;
type _ContentShuffleProcessorConfig = Required<
  Pick<Invocation<'content_shuffle_image_processor'>, 'id' | 'type' | 'w' | 'h' | 'f'>
>;
type _DepthAnythingProcessorConfig = Required<
  Pick<Invocation<'depth_anything_image_processor'>, 'id' | 'type' | 'model_size'>
>;
type _HedProcessorConfig = Required<Pick<Invocation<'hed_image_processor'>, 'id' | 'type' | 'scribble'>>;
type _LineartAnimeProcessorConfig = Required<Pick<Invocation<'lineart_anime_image_processor'>, 'id' | 'type'>>;
type _LineartProcessorConfig = Required<Pick<Invocation<'lineart_image_processor'>, 'id' | 'type' | 'coarse'>>;
type _MediapipeFaceProcessorConfig = Required<
  Pick<Invocation<'mediapipe_face_processor'>, 'id' | 'type' | 'max_faces' | 'min_confidence'>
>;
type _MidasDepthProcessorConfig = Required<
  Pick<Invocation<'midas_depth_image_processor'>, 'id' | 'type' | 'a_mult' | 'bg_th'>
>;
type _MlsdProcessorConfig = Required<Pick<Invocation<'mlsd_image_processor'>, 'id' | 'type' | 'thr_v' | 'thr_d'>>;
type _NormalbaeProcessorConfig = Required<Pick<Invocation<'normalbae_image_processor'>, 'id' | 'type'>>;
type _DWOpenposeProcessorConfig = Required<
  Pick<Invocation<'dw_openpose_image_processor'>, 'id' | 'type' | 'draw_body' | 'draw_face' | 'draw_hands'>
>;
type _PidiProcessorConfig = Required<Pick<Invocation<'pidi_image_processor'>, 'id' | 'type' | 'safe' | 'scribble'>>;
type _ZoeDepthProcessorConfig = Required<Pick<Invocation<'zoe_depth_image_processor'>, 'id' | 'type'>>;
