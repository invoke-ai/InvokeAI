import type { WorkflowCategory, WorkflowV3, XYPosition } from 'features/nodes/types/workflow';
import type * as ReactFlow from 'reactflow';
import type { S } from 'services/api/types';
import type { Equals, Extends } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

/**
 * These types originate from the server and are recreated as zod schemas manually, for use at runtime.
 * The tests ensure that the types are correctly recreated.
 */

describe('Workflow types', () => {
  test('XYPosition', () => assert<Equals<XYPosition, ReactFlow.XYPosition>>());
  test('WorkflowCategory', () => assert<Equals<WorkflowCategory, S['WorkflowCategory']>>());
  // @ts-expect-error TODO(psyche): Need to revise server types!
  test('WorkflowV3', () => assert<Extends<WorkflowV3, S['Workflow']>>());
});
