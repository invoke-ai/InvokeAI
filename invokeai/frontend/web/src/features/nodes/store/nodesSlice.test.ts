import { deepClone } from 'common/util/deepClone';
import type { IntegerFieldInputTemplate, StringFieldInputTemplate } from 'features/nodes/types/field';
import { buildConnectorNode } from 'features/nodes/util/node/buildConnectorNode';
import { describe, expect, it } from 'vitest';

import {
  callSavedWorkflowDynamicFieldsChanged,
  connectorInserted,
  fieldIntegerValueChanged,
  nodesChanged,
  nodesSliceConfig,
} from './nodesSlice';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from './util/connectorTopology';
import { add, buildEdge, buildNode, sub, templates } from './util/testUtils';

const callSavedWorkflowTemplate = templates.call_saved_workflow;
const addTemplate = templates.add;

if (!callSavedWorkflowTemplate || !addTemplate || !addTemplate.inputs.a) {
  throw new Error('Expected saved workflow and add templates');
}
const addIntegerInputTemplate = addTemplate.inputs.a as IntegerFieldInputTemplate;

const buildDynamicIntegerTemplate = (fieldName: string): IntegerFieldInputTemplate => ({
  ...addIntegerInputTemplate,
  name: fieldName,
  title: 'Left Addend',
  input: 'any' as const,
});

const buildDynamicStringTemplate = (fieldName: string): StringFieldInputTemplate => ({
  name: fieldName,
  title: 'Prompt',
  required: false,
  description: 'Prompt text',
  fieldKind: 'input',
  input: 'any',
  ui_hidden: false,
  default: 'new default',
  type: {
    name: 'StringField',
    cardinality: 'SINGLE',
    batch: false,
  },
});

const buildFixedConnectorNode = (id: string) => {
  const connectorNode = buildConnectorNode({ x: 0, y: 0 });
  return {
    ...connectorNode,
    id,
    data: {
      ...connectorNode.data,
      id,
    },
  };
};

describe('callSavedWorkflowDynamicFieldsChanged', () => {
  it('seeds new dynamic fields with the source workflow values', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName: 'saved_workflow_input::node-1::a',
            fieldTemplate: buildDynamicIntegerTemplate('saved_workflow_input::node-1::a'),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    const dynamicField = nextState.nodes[0];
    if (!dynamicField || dynamicField.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(dynamicField.data.inputs['saved_workflow_input::node-1::a']?.value).toBe(23);
    expect(dynamicField.data.inputs['saved_workflow_input::node-1::a']?.label).toBe('Left Addend');
    expect(dynamicField.data.dynamicInputTemplates['saved_workflow_input::node-1::a']?.title).toBe('Left Addend');
  });

  it('preserves existing dynamic field values on resync', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const fieldName = 'saved_workflow_input::node-1::a';

    let nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      fieldIntegerValueChanged({
        nodeId: node.id,
        fieldName,
        value: 99,
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    const resyncedNode = nextState.nodes[0];
    if (!resyncedNode || resyncedNode.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(resyncedNode.data.inputs[fieldName]?.value).toBe(99);
    expect(resyncedNode.data.dynamicInputTemplates[fieldName]?.name).toBe(fieldName);
  });

  it('resets an existing dynamic field value when the exposed field type changes', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const fieldName = 'saved_workflow_input::node-1::a';

    let nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      fieldIntegerValueChanged({
        nodeId: node.id,
        fieldName,
        value: 99,
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicStringTemplate(fieldName),
            label: 'Prompt',
            description: 'Prompt text',
            initialValue: 'new default',
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    const resyncedNode = nextState.nodes[0];
    if (!resyncedNode || resyncedNode.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(resyncedNode.data.inputs[fieldName]?.value).toBe('new default');
    expect(resyncedNode.data.dynamicInputTemplates[fieldName]?.type.name).toBe('StringField');
  });

  it('removes stale dynamic field templates when the selected workflow fields change', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const fieldName = 'saved_workflow_input::node-1::a';

    let nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [],
        edgeIdsToRemove: [],
      })
    );

    const resyncedNode = nextState.nodes[0];
    if (!resyncedNode || resyncedNode.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(resyncedNode.data.inputs[fieldName]).toBeUndefined();
    expect(resyncedNode.data.dynamicInputTemplates[fieldName]).toBeUndefined();
  });

  it('removes specified inbound edges during dynamic field resync', () => {
    const state = nodesSliceConfig.getInitialState();
    const sourceNode = buildNode(addTemplate);
    const targetNode = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(sourceNode, targetNode);
    state.edges.push({
      id: 'edge-1',
      type: 'default',
      source: sourceNode.id,
      sourceHandle: 'value',
      target: targetNode.id,
      targetHandle: 'saved_workflow_input::node-1::a',
    });

    const nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: targetNode.id,
        fields: [],
        edgeIdsToRemove: ['edge-1'],
      })
    );

    expect(nextState.edges).toHaveLength(0);
  });
});

describe('nodesSlice connector actions', () => {
  it('removes an unconnected connector', () => {
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [connector];
    initialState.edges = [];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes).toEqual([]);
    expect(nextState.edges).toEqual([]);
  });

  it('splits a direct edge into source -> connector -> target edges when inserting a connector', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');
    const directEdge = buildEdge(source.id, 'value', target.id, 'a');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, target];
    initialState.edges = [directEdge];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      connectorInserted({
        edgeId: directEdge.id,
        connector,
      })
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id, connector.id]);
    expect(nextState.edges).toEqual([
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ]);
  });

  it('splices connector outputs back to the resolved upstream source when removed', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id]);
    expect(nextState.edges).toEqual([buildEdge(source.id, 'value', target.id, 'a')]);
  });

  it('splices one connector source back to multiple downstream targets when removed', () => {
    const source = buildNode(add);
    const targetA = buildNode(sub);
    const targetB = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, targetA, targetB];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetA.id, 'a'),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetB.id, 'b'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, targetA.id, targetB.id]);
    expect(nextState.edges).toEqual([
      buildEdge(source.id, 'value', targetA.id, 'a'),
      buildEdge(source.id, 'value', targetB.id, 'b'),
    ]);
  });

  it('does not create any edges when removing a connector with no downstream targets', () => {
    const source = buildNode(add);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector];
    initialState.edges = [buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id]);
    expect(nextState.edges).toEqual([]);
  });

  it('removes a connector while preserving downstream connector edges in a chained splice case', () => {
    const source = buildNode(add);
    const connectorA = buildFixedConnectorNode('connector-a');
    const connectorB = buildFixedConnectorNode('connector-b');
    const target = buildNode(sub);

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connectorA, connectorB, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connectorA.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, connectorB.id, target.id]);
    expect(nextState.edges).toHaveLength(2);
    expect(nextState.edges).toEqual(
      expect.arrayContaining([
        buildEdge(source.id, 'value', connectorB.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ])
    );
  });

  it('splices connector edges when the connector is removed through generic node removal', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id]);
    expect(nextState.edges).toEqual([buildEdge(source.id, 'value', target.id, 'a')]);
  });
});
