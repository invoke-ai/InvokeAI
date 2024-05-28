import { deepClone } from 'common/util/deepClone';
import { getCollectItemType } from 'features/nodes/store/util/getCollectItemType';
import { add, buildEdge, buildNode, collect, templates } from 'features/nodes/store/util/testUtils';
import type { FieldType } from 'features/nodes/types/field';
import { unset } from 'lodash-es';
import { describe, expect, it } from 'vitest';

describe(getCollectItemType.name, () => {
  it('should return the type of the items the collect node collects', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(collect);
    const e1 = buildEdge(n1.id, 'value', n2.id, 'item');
    const result = getCollectItemType(templates, [n1, n2], [e1], n2.id);
    expect(result).toEqual<FieldType>({ name: 'IntegerField', cardinality: 'SINGLE' });
  });
  it('should return null if the collect node does not have any connections', () => {
    const n1 = buildNode(collect);
    const result = getCollectItemType(templates, [n1], [], n1.id);
    expect(result).toBeNull();
  });
  it("should return null if the first edge to collect's node doesn't exist", () => {
    const n1 = buildNode(collect);
    const n2 = buildNode(add);
    const e1 = buildEdge(n2.id, 'value', n1.id, 'item');
    const result = getCollectItemType(templates, [n1], [e1], n1.id);
    expect(result).toBeNull();
  });
  it("should return null if the first edge to collect's node template doesn't exist", () => {
    const n1 = buildNode(collect);
    const n2 = buildNode(add);
    const e1 = buildEdge(n2.id, 'value', n1.id, 'item');
    const result = getCollectItemType({ collect }, [n1, n2], [e1], n1.id);
    expect(result).toBeNull();
  });
  it("should return null if the first edge to the collect's field template doesn't exist", () => {
    const n1 = buildNode(collect);
    const n2 = buildNode(add);
    const addWithoutOutputValue = deepClone(add);
    unset(addWithoutOutputValue, 'outputs.value');
    const e1 = buildEdge(n2.id, 'value', n1.id, 'item');
    const result = getCollectItemType({ add: addWithoutOutputValue, collect }, [n2, n1], [e1], n1.id);
    expect(result).toBeNull();
  });
});
