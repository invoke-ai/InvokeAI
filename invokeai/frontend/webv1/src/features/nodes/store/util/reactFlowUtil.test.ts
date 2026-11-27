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

  // `connectionToEdge` is the only edge constructor this editor has, and it is
  // used by reconnect and by fresh connection drags. Now that loop_linkage edges
  // load here, dragging one back onto the same handle must not silently turn it
  // into a data edge: webv2 infers the type from the handle names, and the
  // backend rejects a loop_linkage field wired by a default edge.
  it('creates a loop_linkage edge when both endpoints are loop linkage handles', () => {
    expect(
      connectionToEdge({
        source: 'for-node',
        sourceHandle: 'loop_linkage',
        target: 'for-return-node',
        targetHandle: 'loop_linkage',
      })
    ).toMatchObject({ type: 'loop_linkage' });
  });
});
