import { deepClone } from 'common/util/deepClone';
import { callSavedWorkflowDynamicFieldsChanged, nodesSliceConfig } from 'features/nodes/store/nodesSlice';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import { add, buildEdge, buildNode, img_resize, sub, templates } from 'features/nodes/store/util/testUtils';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { buildNodesGraph } from './buildNodesGraph';

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
  input: 'any',
});

const buildConnectorNode = (id: string) => ({
  id,
  type: 'connector' as const,
  position: { x: 0, y: 0 },
  data: {
    id,
    type: 'connector' as const,
    label: 'Connector',
    isOpen: true,
  },
});

const buildState = (nodes: unknown[], edges: unknown[]) =>
  ({
    nodes: {
      present: {
        _version: 1,
        nodes,
        edges,
        formFieldInitialValues: {},
        id: undefined,
        name: '',
        author: '',
        description: '',
        version: '',
        contact: '',
        tags: '',
        notes: '',
        exposedFields: [],
        meta: { version: '4.0.0', category: 'user' },
        form: {
          rootElementId: 'root',
          elements: {
            root: {
              id: 'root',
              type: 'container',
              data: { layout: 'column', children: [] },
            },
          },
        },
      },
    },
    gallery: {
      autoAddBoardId: 'none',
      selection: [],
    },
  }) as unknown as Parameters<typeof buildNodesGraph>[0];

describe('buildNodesGraph', () => {
  it('serializes dynamic saved workflow inputs into workflow_inputs', () => {
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

    const rootState = {
      nodes: {
        past: [],
        future: [],
        present: nextState,
      },
      gallery: {
        autoAddBoardId: 'none',
      },
    } as never;

    const graph = buildNodesGraph(rootState, templates);

    expect(graph.nodes[node.id]).toMatchObject({
      workflow_id: '',
      workflow_inputs: {
        ['saved_workflow_input::node-1::a']: 23,
      },
    });
  });

  it('omits connected dynamic saved workflow literal values from workflow_inputs while preserving the edge', () => {
    const state = nodesSliceConfig.getInitialState();
    const sourceNode = buildNode(add);
    const callNode = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(sourceNode, callNode);

    const nextState = deepClone(
      nodesSliceConfig.slice.reducer(
        state,
        callSavedWorkflowDynamicFieldsChanged({
          nodeId: callNode.id,
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
      )
    );

    nextState.edges.push(buildEdge(sourceNode.id, 'value', callNode.id, 'saved_workflow_input::node-1::a'));

    const rootState = {
      nodes: {
        past: [],
        future: [],
        present: nextState,
      },
      gallery: {
        autoAddBoardId: 'none',
      },
    } as never;

    const graph = buildNodesGraph(rootState, templates);

    expect(graph.nodes[callNode.id]).toMatchObject({
      workflow_id: '',
      workflow_inputs: {},
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: sourceNode.id, field: 'value' },
      destination: { node_id: callNode.id, field: 'saved_workflow_input::node-1::a' },
    });
  });

  it('does not serialize stale hidden saved workflow input values without matching dynamic fields', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    node.data.inputs.workflow_inputs = {
      name: 'workflow_inputs',
      type: 'workflow_inputs',
      value: {
        ['saved_workflow_input::old-node::a']: 23,
      },
    } as never;
    state.nodes.push(node);
    const templatesWithWorkflowInputs = {
      ...templates,
      call_saved_workflow: {
        ...callSavedWorkflowTemplate,
        inputs: {
          ...callSavedWorkflowTemplate.inputs,
          workflow_inputs: buildDynamicIntegerTemplate('workflow_inputs'),
        },
      },
    };

    const rootState = {
      nodes: {
        past: [],
        future: [],
        present: state,
      },
      gallery: {
        autoAddBoardId: 'none',
      },
    } as never;

    const graph = buildNodesGraph(rootState, templatesWithWorkflowInputs);
    const graphNode = graph.nodes[node.id] as { workflow_id: string; workflow_inputs: Record<string, unknown> };

    expect(graphNode.workflow_id).toBe('');
    expect(graphNode.workflow_inputs).toEqual({});
  });

  it('flattens a single connector to one direct execution edge', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.nodes).not.toHaveProperty(connector.id);
    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('flattens chained connectors transitively', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connectorA = buildConnectorNode('connector-a');
    const connectorB = buildConnectorNode('connector-b');
    const state = buildState(
      [source, target, connectorA, connectorB],
      [
        buildEdge(source.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('fans out through a connector into multiple execution edges', () => {
    const source = buildNode(add);
    const targetA = buildNode(sub);
    const targetB = buildNode(img_resize);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, targetA, targetB, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetA.id, 'a'),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetB.id, 'width'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: targetA.id, field: 'a' },
      },
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: targetB.id, field: 'width' },
      },
    ]);
  });

  it('drops unresolved connector paths from the execution graph', () => {
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState([target, connector], [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a')]);

    const graph = buildNodesGraph(state, templates);

    expect(graph.nodes).not.toHaveProperty(connector.id);
    expect(graph.edges).toEqual([]);
  });

  it('deduplicates effective execution edges created by flattening', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
        buildEdge(source.id, 'value', target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('still omits explicit destination input values when the flattened edge exists', () => {
    const source = buildNode(add);
    const target = deepClone(buildNode(sub));
    const connector = buildConnectorNode('connector-1');
    const inputA = target.data.inputs.a;
    expect(inputA).toBeDefined();
    if (!inputA) {
      throw new Error('Missing input a');
    }
    inputA.value = 'not-an-integer' as never;
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
    expect(graph.nodes[target.id]).not.toHaveProperty('a');
  });
});
