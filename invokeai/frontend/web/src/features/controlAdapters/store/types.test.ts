import type { ControlAdapterProcessorType, zControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';
import type { z } from 'zod';

describe('Control Adapter Types', () => {
  test('ControlAdapterProcessorType', () =>
    assert<Equals<ControlAdapterProcessorType, z.infer<typeof zControlAdapterProcessorType>>>());
});
