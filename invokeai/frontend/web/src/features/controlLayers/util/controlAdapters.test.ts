import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

import type {
  _CannyProcessorConfig,
  _ColorMapProcessorConfig,
  _ContentShuffleProcessorConfig,
  _DepthAnythingProcessorConfig,
  _DWOpenposeProcessorConfig,
  _HedProcessorConfig,
  _LineartAnimeProcessorConfig,
  _LineartProcessorConfig,
  _MediapipeFaceProcessorConfig,
  _MidasDepthProcessorConfig,
  _MlsdProcessorConfig,
  _NormalbaeProcessorConfig,
  _PidiProcessorConfig,
  _ZoeDepthProcessorConfig,
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
} from './controlAdapters';

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
