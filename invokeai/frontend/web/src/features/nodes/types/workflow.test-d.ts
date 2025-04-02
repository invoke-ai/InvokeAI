import type { XYPosition as ReactFlowXYPosition } from '@xyflow/react';
import type { WorkflowCategory, WorkflowV3, XYPosition } from 'features/nodes/types/workflow';
import type { S } from 'services/api/types';
import type { Equals, Extends } from 'tsafe';
import { assert } from 'tsafe';
import type { SetRequired } from 'type-fest';
import { describe, test } from 'vitest';

/**
 * These types originate from the server and are recreated as zod schemas manually, for use at runtime.
 * The tests ensure that the types are correctly recreated.
 */

describe('Workflow types', () => {
  test('XYPosition', () => assert<Equals<XYPosition, ReactFlowXYPosition>>());
  test('WorkflowCategory', () => assert<Equals<WorkflowCategory, S['WorkflowCategory']>>());
  test('WorkflowV3', () => assert<Extends<SetRequired<WorkflowV3, 'id'>, S['Workflow']>>());
});
