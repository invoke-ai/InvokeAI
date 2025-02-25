import type { AppStore } from 'app/store/store';
import {
  isFloatFieldCollectionInputInstance,
  isFloatGeneratorFieldInputInstance,
  isImageFieldCollectionInputInstance,
  isIntegerFieldCollectionInputInstance,
  isIntegerGeneratorFieldInputInstance,
  isStringFieldCollectionInputInstance,
  isStringGeneratorFieldInputInstance,
  resolveFloatGeneratorField,
  resolveIntegerGeneratorField,
  resolveStringGeneratorField,
} from 'features/nodes/types/field';
import type { AnyEdge, InvocationNode } from 'features/nodes/types/invocation';
import { assert } from 'tsafe';

export const resolveBatchValue = async (
  batchNode: InvocationNode,
  nodes: InvocationNode[],
  edges: AnyEdge[],
  store: AppStore
) => {
  if (batchNode.data.type === 'image_batch') {
    assert(isImageFieldCollectionInputInstance(batchNode.data.inputs.images));
    const ownValue = batchNode.data.inputs.images.value ?? [];
    // no generators for images yet
    return ownValue;
  } else if (batchNode.data.type === 'string_batch') {
    assert(isStringFieldCollectionInputInstance(batchNode.data.inputs.strings));
    const ownValue = batchNode.data.inputs.strings.value;
    const edgeToStrings = edges.find((edge) => edge.target === batchNode.id && edge.targetHandle === 'strings');

    if (!edgeToStrings) {
      return ownValue ?? [];
    }

    const generatorNode = nodes.find((node) => node.id === edgeToStrings.source);
    assert(generatorNode, 'Missing edge from string generator to string batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isStringGeneratorFieldInputInstance(generatorField), 'Invalid string generator');

    const generatorValue = await resolveStringGeneratorField(generatorField, store);
    return generatorValue;
  } else if (batchNode.data.type === 'float_batch') {
    assert(isFloatFieldCollectionInputInstance(batchNode.data.inputs.floats));
    const ownValue = batchNode.data.inputs.floats.value;
    const edgeToFloats = edges.find((edge) => edge.target === batchNode.id && edge.targetHandle === 'floats');

    if (!edgeToFloats) {
      return ownValue ?? [];
    }

    const generatorNode = nodes.find((node) => node.id === edgeToFloats.source);
    assert(generatorNode, 'Missing edge from float generator to float batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isFloatGeneratorFieldInputInstance(generatorField), 'Invalid float generator');

    const generatorValue = resolveFloatGeneratorField(generatorField);
    return generatorValue;
  } else if (batchNode.data.type === 'integer_batch') {
    assert(isIntegerFieldCollectionInputInstance(batchNode.data.inputs.integers));
    const ownValue = batchNode.data.inputs.integers.value;
    const incomers = edges.find((edge) => edge.target === batchNode.id && edge.targetHandle === 'integers');

    if (!incomers) {
      return ownValue ?? [];
    }

    const generatorNode = nodes.find((node) => node.id === incomers.source);
    assert(generatorNode, 'Missing edge from integer generator to integer batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isIntegerGeneratorFieldInputInstance(generatorField), 'Invalid integer generator field');

    const generatorValue = resolveIntegerGeneratorField(generatorField);
    return generatorValue;
  }
  assert(false, 'Invalid batch node type');
};
