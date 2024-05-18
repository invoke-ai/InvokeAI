import { getHasCycles } from 'features/nodes/store/util/getHasCycles';
import { add, buildEdge, position } from 'features/nodes/store/util/testUtils';
import { buildInvocationNode } from 'features/nodes/util/node/buildInvocationNode';
import { describe, expect, it } from 'vitest';

describe(getHasCycles.name, () => {
  const n1 = buildInvocationNode(position, add);
  const n2 = buildInvocationNode(position, add);
  const n3 = buildInvocationNode(position, add);
  const nodes = [n1, n2, n3];

  it('should return true if the graph WOULD have cycles after adding the edge', () => {
    const edges = [buildEdge(n1.id, 'value', n2.id, 'a'), buildEdge(n2.id, 'value', n3.id, 'a')];
    const result = getHasCycles(n3.id, n1.id, nodes, edges);
    expect(result).toBe(true);
  });

  it('should return false if the graph WOULD NOT have cycles after adding the edge', () => {
    const edges = [buildEdge(n1.id, 'value', n2.id, 'a')];
    const result = getHasCycles(n2.id, n3.id, nodes, edges);
    expect(result).toBe(false);
  });
});
