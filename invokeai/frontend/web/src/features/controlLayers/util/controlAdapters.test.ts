import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

import type { CLIPVisionModel, ControlMode, IPMethod, ProcessorConfig, ProcessorType } from './controlAdapters';

describe('Control Adapter Types', () => {
  test('ProcessorType', () => assert<Equals<ProcessorConfig['type'], ProcessorType>>());
  test('IP Adapter Method', () => assert<Equals<NonNullable<S['IPAdapterInvocation']['method']>, IPMethod>>());
  test('CLIP Vision Model', () =>
    assert<Equals<NonNullable<S['IPAdapterInvocation']['clip_vision_model']>, CLIPVisionModel>>());
  test('Control Mode', () => assert<Equals<NonNullable<S['ControlNetInvocation']['control_mode']>, ControlMode>>());
});
