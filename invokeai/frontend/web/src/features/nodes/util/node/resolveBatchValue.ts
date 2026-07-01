import type { AppDispatch } from 'app/store/store';
import type { NodesState } from 'features/nodes/store/types';
import type { ImageField } from 'features/nodes/types/common';
import {
  isFloatFieldCollectionInputInstance,
  isFloatGeneratorFieldInputInstance,
  isImageFieldCollectionInputInstance,
  isImageGeneratorFieldInputInstance,
  isIntegerFieldCollectionInputInstance,
  isIntegerGeneratorFieldInputInstance,
  isStringFieldCollectionInputInstance,
  isStringGeneratorFieldInputInstance,
  resolveFloatGeneratorField,
  resolveImageGeneratorField,
  resolveIntegerGeneratorField,
  resolveStringGeneratorField,
} from 'features/nodes/types/field';
import type { InvocationNode } from 'features/nodes/types/invocation';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { assert } from 'tsafe';

export const resolveBatchValue = async (arg: {
  dispatch: AppDispatch;
  nodesState: NodesState;
  node: InvocationNode;
}): Promise<number[] | string[] | ImageField[]> => {
  const { node, dispatch, nodesState } = arg;
  const { nodes, edges } = nodesState;
  const invocationNodes = nodes.filter(isInvocationNode);

  if (node.data.type === 'image_batch') {
    assert(isImageFieldCollectionInputInstance(node.data.inputs.images));
    const ownValue = node.data.inputs.images.value ?? [];
    const incomers = edges.find((edge) => edge.target === node.id && edge.targetHandle === 'images');

    if (!incomers) {
      return ownValue ?? [];
    }

    const generatorNode = invocationNodes.find((node) => node.id === incomers.source);
    assert(generatorNode, 'Missing edge from image generator to image batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isImageGeneratorFieldInputInstance(generatorField), 'Invalid image generator field');

    const generatorValue = await resolveImageGeneratorField(generatorField, dispatch);
    return generatorValue;
  } else if (node.data.type === 'string_batch') {
    assert(isStringFieldCollectionInputInstance(node.data.inputs.strings));
    const ownValue = node.data.inputs.strings.value;
    const edgeToStrings = edges.find((edge) => edge.target === node.id && edge.targetHandle === 'strings');

    if (!edgeToStrings) {
      return ownValue ?? [];
    }

    const generatorNode = invocationNodes.find((node) => node.id === edgeToStrings.source);
    assert(generatorNode, 'Missing edge from string generator to string batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isStringGeneratorFieldInputInstance(generatorField), 'Invalid string generator');

    const generatorValue = await resolveStringGeneratorField(generatorField, dispatch);
    return generatorValue;
  } else if (node.data.type === 'float_batch') {
    assert(isFloatFieldCollectionInputInstance(node.data.inputs.floats));
    const ownValue = node.data.inputs.floats.value;
    const edgeToFloats = edges.find((edge) => edge.target === node.id && edge.targetHandle === 'floats');

    if (!edgeToFloats) {
      return ownValue ?? [];
    }

    const generatorNode = invocationNodes.find((node) => node.id === edgeToFloats.source);
    assert(generatorNode, 'Missing edge from float generator to float batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isFloatGeneratorFieldInputInstance(generatorField), 'Invalid float generator');

    const generatorValue = resolveFloatGeneratorField(generatorField);
    return generatorValue;
  } else if (node.data.type === 'integer_batch') {
    assert(isIntegerFieldCollectionInputInstance(node.data.inputs.integers));
    const ownValue = node.data.inputs.integers.value;
    const incomers = edges.find((edge) => edge.target === node.id && edge.targetHandle === 'integers');

    if (!incomers) {
      return ownValue ?? [];
    }

    const generatorNode = invocationNodes.find((node) => node.id === incomers.source);
    assert(generatorNode, 'Missing edge from integer generator to integer batch');

    const generatorField = generatorNode.data.inputs['generator'];
    assert(isIntegerGeneratorFieldInputInstance(generatorField), 'Invalid integer generator field');

    const generatorValue = resolveIntegerGeneratorField(generatorField);
    return generatorValue;
  }
  assert(false, 'Invalid batch node type');
};

export type BatchSizeResult = number | 'EMPTY_BATCHES' | 'NO_BATCHES' | 'MISMATCHED_BATCH_GROUP';

export const getBatchSize = async (nodesState: NodesState, dispatch: AppDispatch): Promise<BatchSizeResult> => {
  const { nodes } = nodesState;
  const batchNodes = nodes.filter(isInvocationNode).filter(isBatchNode);
  const ungroupedBatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'None');
  const group1BatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'Group 1');
  const group2BatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'Group 2');
  const group3BatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'Group 3');
  const group4BatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'Group 4');
  const group5BatchNodes = batchNodes.filter((node) => node.data.inputs['batch_group_id']?.value === 'Group 5');

  const ungroupedBatchSizes = await Promise.all(
    ungroupedBatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );
  const group1BatchSizes = await Promise.all(
    group1BatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );
  const group2BatchSizes = await Promise.all(
    group2BatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );
  const group3BatchSizes = await Promise.all(
    group3BatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );
  const group4BatchSizes = await Promise.all(
    group4BatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );
  const group5BatchSizes = await Promise.all(
    group5BatchNodes.map(async (node) => (await resolveBatchValue({ nodesState, dispatch, node })).length)
  );

  // All batch nodes _must_ have a populated collection

  const allBatchSizes = [
    ...ungroupedBatchSizes,
    ...group1BatchSizes,
    ...group2BatchSizes,
    ...group3BatchSizes,
    ...group4BatchSizes,
    ...group5BatchSizes,
  ];

  // There are no batch nodes
  if (allBatchSizes.length === 0) {
    return 'NO_BATCHES';
  }

  // All batch nodes must have a populated collection
  if (allBatchSizes.some((size) => size === 0)) {
    return 'EMPTY_BATCHES';
  }

  for (const group of [group1BatchSizes, group2BatchSizes, group3BatchSizes, group4BatchSizes, group5BatchSizes]) {
    // Ignore groups with no batch nodes
    if (group.length === 0) {
      continue;
    }
    // Grouped batch nodes must have the same collection size
    if (group.some((size) => size !== group[0])) {
      return 'MISMATCHED_BATCH_GROUP';
    }
  }

  // Total batch size = product of all ungrouped batches and each grouped batch
  const totalBatchSize = [
    ...ungroupedBatchSizes,
    // In case of no batch nodes in a group, fall back to 1 for the product calculation
    group1BatchSizes[0] ?? 1,
    group2BatchSizes[0] ?? 1,
    group3BatchSizes[0] ?? 1,
    group4BatchSizes[0] ?? 1,
    group5BatchSizes[0] ?? 1,
  ].reduce((acc, size) => acc * size, 1);

  return totalBatchSize;
};
