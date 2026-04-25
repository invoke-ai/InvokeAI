import { describe, expect, it } from 'vitest';

import { connectionToEdge } from './reactFlowUtil';

describe('connectionToEdge', () => {
  it('creates a default edge with the expected id and endpoints', () => {
    expect(
      connectionToEdge({
        source: 'source-node',
        sourceHandle: 'value',
        target: 'target-node',
        targetHandle: 'a',
      })
    ).toEqual({
      type: 'default',
      source: 'source-node',
      sourceHandle: 'value',
      target: 'target-node',
      targetHandle: 'a',
      id: 'reactflow__edge-source-nodevalue-target-nodea',
    });
  });
});
