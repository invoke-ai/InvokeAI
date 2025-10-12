import { createAction } from '@reduxjs/toolkit';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';

import type { ParamsEnrichedPayload } from './types';

// Needed to split this from canvasSlice.ts to avoid circular dependencies
export const canvasReset = createAction('canvas/canvasReset');

// Needed to split this from paramsSlice.ts to avoid circular dependencies
export const modelChanged =
  createAction<ParamsEnrichedPayload<{ model: ParameterModel | null; previousModel?: ParameterModel | null }>>(
    'params/modelChanged'
  );
