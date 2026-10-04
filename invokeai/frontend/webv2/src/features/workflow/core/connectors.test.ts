import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, InvocationTemplates, WorkflowInvocationNode } from './types';

import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  getConnectorDeletionSpliceConnections,
  resolveConnectorTarget,
} from './connectors';
import { createProjectGraph, projectGraphReducer } from './document';

const node = (id: string, type: string): WorkflowInvocationNode => ({
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

const dynamicInput = (name: string): FieldInputTemplate => ({
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name,
  type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
});

describe('connector deletion', () => {
  it('reconnects ordinary data through a removed connector', () => {
    const source = node('source', 'source');
    const target = node('target', 'target');
    const connector = {
      data: { label: '' },
      id: 'connector',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const document = {
      ...createProjectGraph('connector-delete'),
      nodes: [source, connector, target],
      edges: [
        {
          id: 'in',
          source: 'source',
          sourceHandle: 'out',
          target: 'connector',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'out',
          source: 'connector',
          sourceHandle: 'out',
          target: 'target',
          targetHandle: 'input',
          type: 'default' as const,
        },
      ],
    };

    expect(getConnectorDeletionSpliceConnections('connector', document.nodes, document.edges)).toEqual([
      {
        id: 'splice-source-out-target-input',
        source: 'source',
        sourceHandle: 'out',
        target: 'target',
        targetHandle: 'input',
        type: 'default',
      },
    ]);

    const next = projectGraphReducer(document, { nodeIds: ['connector'], type: 'removeNodes' });
    expect(next.nodes.map((candidate) => candidate.id)).toEqual(['source', 'target']);
    expect(next.edges).toEqual([
      {
        id: 'splice-source-out-target-input',
        source: 'source',
        sourceHandle: 'out',
        target: 'target',
        targetHandle: 'input',
        type: 'default',
      },
    ]);
  });

  it('reconnects a removed connector alias as a direct loop_linkage edge', () => {
    const forNode = node('for', 'for');
    const returnNode = node('return', 'for_return');
    const connector = {
      data: { label: '' },
      id: 'connector',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const document = {
      ...createProjectGraph('connector-loop-delete'),
      nodes: [forNode, connector, returnNode],
      edges: [
        {
          id: 'in',
          source: 'for',
          sourceHandle: 'loop_linkage',
          target: 'connector',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'out',
          source: 'connector',
          sourceHandle: 'out',
          target: 'return',
          targetHandle: 'loop_linkage',
          type: 'default' as const,
        },
      ],
    };

    const next = projectGraphReducer(document, { nodeIds: ['connector'], type: 'removeNodes' });
    expect(next.edges).toEqual([
      {
        id: 'splice-for-loop_linkage-return-loop_linkage',
        source: 'for',
        sourceHandle: 'loop_linkage',
        target: 'return',
        targetHandle: 'loop_linkage',
        type: 'loop_linkage',
      },
    ]);
  });

  it('keeps a surviving upstream connector when deleting a downstream connector', () => {
    const source = node('source', 'source');
    const target = node('target', 'target');
    const upstream = {
      data: { label: '' },
      id: 'upstream',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const downstream = {
      data: { label: '' },
      id: 'downstream',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const document = {
      ...createProjectGraph('connector-chain-delete'),
      nodes: [source, upstream, downstream, target],
      edges: [
        {
          id: 'source-upstream',
          source: 'source',
          sourceHandle: 'out',
          target: 'upstream',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'upstream-downstream',
          source: 'upstream',
          sourceHandle: 'out',
          target: 'downstream',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'downstream-target',
          source: 'downstream',
          sourceHandle: 'out',
          target: 'target',
          targetHandle: 'input',
          type: 'default' as const,
        },
      ],
    };

    const next = projectGraphReducer(document, { nodeIds: ['downstream'], type: 'removeNodes' });
    expect(next.nodes.map((candidate) => candidate.id)).toEqual(['source', 'upstream', 'target']);
    expect(next.edges).toEqual([
      {
        id: 'source-upstream',
        source: 'source',
        sourceHandle: 'out',
        target: 'upstream',
        targetHandle: 'in',
        type: 'default',
      },
      {
        id: 'splice-source-out-target-input',
        source: 'source',
        sourceHandle: 'out',
        target: 'target',
        targetHandle: 'input',
        type: 'default',
      },
    ]);
  });

  it('preserves the loop-linkage alias when deleting one connector from a chain', () => {
    const forNode = node('for', 'for');
    const returnNode = node('return', 'for_return');
    const upstream = {
      data: { label: '' },
      id: 'upstream',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const downstream = {
      data: { label: '' },
      id: 'downstream',
      position: { x: 0, y: 0 },
      type: 'connector' as const,
    };
    const document = {
      ...createProjectGraph('connector-loop-chain-delete'),
      nodes: [forNode, upstream, downstream, returnNode],
      edges: [
        {
          id: 'for-upstream',
          source: 'for',
          sourceHandle: 'loop_linkage',
          target: 'upstream',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'upstream-downstream',
          source: 'upstream',
          sourceHandle: 'out',
          target: 'downstream',
          targetHandle: 'in',
          type: 'default' as const,
        },
        {
          id: 'downstream-return',
          source: 'downstream',
          sourceHandle: 'out',
          target: 'return',
          targetHandle: 'loop_linkage',
          type: 'default' as const,
        },
      ],
    };

    expect(projectGraphReducer(document, { nodeIds: ['upstream'], type: 'removeNodes' }).edges).toEqual([
      {
        id: 'downstream-return',
        source: 'downstream',
        sourceHandle: 'out',
        target: 'return',
        targetHandle: 'loop_linkage',
        type: 'default',
      },
      {
        id: 'splice-for-loop_linkage-downstream-in',
        source: 'for',
        sourceHandle: 'loop_linkage',
        target: 'downstream',
        targetHandle: 'in',
        type: 'default',
      },
    ]);

    expect(projectGraphReducer(document, { nodeIds: ['downstream'], type: 'removeNodes' }).edges).toEqual([
      {
        id: 'for-upstream',
        source: 'for',
        sourceHandle: 'loop_linkage',
        target: 'upstream',
        targetHandle: 'in',
        type: 'default',
      },
      {
        id: 'splice-upstream-out-return-loop_linkage',
        source: 'upstream',
        sourceHandle: 'out',
        target: 'return',
        targetHandle: 'loop_linkage',
        type: 'default',
      },
    ]);
  });
});

describe('connector target resolution', () => {
  it('uses a persisted dynamic input template for a Call Saved Workflow target', () => {
    const source = node('source', 'source');
    const target = node('target', 'call_saved_workflow');
    target.data.dynamicInputTemplates = { dynamic_input: dynamicInput('dynamic_input') };
    const templates: InvocationTemplates = {};

    expect(
      resolveConnectorTarget(
        'connector',
        [
          source,
          {
            data: { label: '' },
            id: 'connector',
            position: { x: 0, y: 0 },
            type: 'connector',
          },
          target,
        ],
        [
          {
            id: 'source-connector',
            source: 'source',
            sourceHandle: 'output',
            target: 'connector',
            targetHandle: CONNECTOR_INPUT_HANDLE,
            type: 'default',
          },
          {
            id: 'connector-target',
            source: 'connector',
            sourceHandle: CONNECTOR_OUTPUT_HANDLE,
            target: 'target',
            targetHandle: 'dynamic_input',
            type: 'default',
          },
        ],
        templates
      )?.type
    ).toEqual({ batch: false, cardinality: 'SINGLE', name: 'IntegerField' });
  });
});
