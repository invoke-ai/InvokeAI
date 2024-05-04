import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

import type {
  CLIPVisionModelV2,
  ControlModeV2,
  DepthAnythingModelSize,
  IPMethodV2,
  ProcessorConfig,
  ProcessorTypeV2,
} from './controlAdapters';

describe('Control Adapter Types', () => {
  test('ProcessorType', () => assert<Equals<ProcessorConfig['type'], ProcessorTypeV2>>());
  test('IP Adapter Method', () => assert<Equals<NonNullable<S['IPAdapterInvocation']['method']>, IPMethodV2>>());
  test('CLIP Vision Model', () =>
    assert<Equals<NonNullable<S['IPAdapterInvocation']['clip_vision_model']>, CLIPVisionModelV2>>());
  test('Control Mode', () => assert<Equals<NonNullable<S['ControlNetInvocation']['control_mode']>, ControlModeV2>>());
  test('DepthAnything Model Size', () =>
    assert<Equals<NonNullable<S['DepthAnythingImageProcessorInvocation']['model_size']>, DepthAnythingModelSize>>());
});
