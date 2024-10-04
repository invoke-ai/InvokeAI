import { z } from 'zod';

import { zFieldInputInstance, zFieldOutputInstance } from './field';
import { zSemVer } from './semver';

// #region NodeData
export const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.string().trim().min(1),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
  isIntermediate: z.boolean(),
  useCache: z.boolean(),
  version: zSemVer,
  nodePack: z.string().min(1).nullish(),
  inputs: z.record(zFieldInputInstance),
  outputs: z.record(zFieldOutputInstance),
});

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});
// #endregion
