import openapi from '@api/openapi.json';
import { parseOpenApiToTemplates } from '@features/workflow/data/templates';
import { expect, it } from 'vitest';

import { getProjectGraphReadiness, compileProjectGraph } from './buildGraph';
import { buildInvocationNode, projectGraphReducer, updateWorkflowNodes } from './document';
import { previewGraphToDocument } from './graphToDocument';
import { isInvocationNode } from './types';

const templates = parseOpenApiToTemplates(openapi);

const recalled = (inputs: Record<string, unknown>) =>
  previewGraphToDocument(
    {
      nodes: [{ id: 'denoise', type: 'denoise_latents', inputs }],
      edges: [],
    },
    templates
  ).document;

it.each([
  [{}, false, 0.4, 0],
  [{ hidiffusion_t1_ratio: 0.4, hidiffusion_t2_ratio: 0 }, false, 0.4, 0],
  [{ hidiffusion_t1_ratio: null, hidiffusion_t2_ratio: null }, true, 0, 0],
  [{ hidiffusion_t1_ratio: 0.4, hidiffusion_t2_ratio: null }, false, 0.4, 0],
  [{ hidiffusion_auto_ratios: false }, false, 0.4, 0],
  [{ hidiffusion_t2_ratio: 0.2 }, false, 0.4, 0.2],
] as const)('recalls legacy mode before template defaults hide it', (inputs, auto, t1, t2) => {
  const graph = compileProjectGraph(recalled(inputs), templates);
  expect(graph.backendGraph.nodes.denoise).toMatchObject({
    hidiffusion_auto_ratios: auto,
    hidiffusion_t1_ratio: t1,
    hidiffusion_t2_ratio: t2,
  });
});

it('upgrades saved nodes preserving manual zero and numeric values', () => {
  const doc = recalled({ hidiffusion_t1_ratio: 0.4, hidiffusion_t2_ratio: 0 });
  const node = doc.nodes[0]!;
  if (!isInvocationNode(node)) {
    throw new Error('expected invocation');
  }
  delete node.data.inputs.hidiffusion_auto_ratios;
  node.data.version = '1.7.0';
  const updated = updateWorkflowNodes(doc, templates);
  const graph = compileProjectGraph(updated.document, templates);
  expect(graph.backendGraph.nodes.denoise).toMatchObject({
    hidiffusion_auto_ratios: false,
    hidiffusion_t1_ratio: 0.4,
    hidiffusion_t2_ratio: 0,
  });
});

it('keeps numeric positions across toggles and turns cleared numeric inputs into zero', () => {
  let doc = recalled({ hidiffusion_auto_ratios: false, hidiffusion_t1_ratio: 0.7, hidiffusion_t2_ratio: 0.2 });
  for (const value of [true, false]) {
    doc = projectGraphReducer(doc, {
      type: 'setFieldValue',
      nodeId: 'denoise',
      fieldName: 'hidiffusion_auto_ratios',
      value,
    });
  }
  expect(compileProjectGraph(doc, templates).backendGraph.nodes.denoise).toMatchObject({
    hidiffusion_t1_ratio: 0.7,
    hidiffusion_t2_ratio: 0.2,
  });
  doc = projectGraphReducer(doc, {
    type: 'setFieldValue',
    nodeId: 'denoise',
    fieldName: 'hidiffusion_t2_ratio',
    value: null,
  });
  expect(compileProjectGraph(doc, templates).backendGraph.nodes.denoise).toMatchObject({ hidiffusion_t2_ratio: 0 });
});

it('rejects reversed direct ratios before enqueue but allows equality, auto and connected values', () => {
  const doc = recalled({
    hidiffusion: true,
    hidiffusion_auto_ratios: false,
    hidiffusion_t1_ratio: 0,
    hidiffusion_t2_ratio: 0.2,
  });
  const snapshot = { status: 'loaded' as const, error: null, templates };
  const orderErrors = (document: typeof doc) =>
    getProjectGraphReadiness(document, snapshot).reasons.filter(
      (reason) => 'workflowIssue' in reason && reason.workflowIssue.message.includes('T2 ratio')
    );
  expect(orderErrors(doc)).toHaveLength(1);
  const equal = projectGraphReducer(doc, {
    type: 'setFieldValue',
    nodeId: 'denoise',
    fieldName: 'hidiffusion_t2_ratio',
    value: 0,
  });
  expect(orderErrors(equal)).toHaveLength(0);
  const auto = projectGraphReducer(doc, {
    type: 'setFieldValue',
    nodeId: 'denoise',
    fieldName: 'hidiffusion_auto_ratios',
    value: true,
  });
  expect(orderErrors(auto)).toHaveLength(0);
  const source = buildInvocationNode(templates.float!, { x: 0, y: 0 });
  const connected = {
    ...doc,
    nodes: [...doc.nodes, source],
    edges: [
      {
        id: 'ratio',
        source: source.id,
        sourceHandle: 'value',
        target: 'denoise',
        targetHandle: 'hidiffusion_t1_ratio',
        type: 'default' as const,
      },
    ],
  };
  expect(orderErrors(connected)).toHaveLength(0);
});

it('keeps legacy incoming ratio connections manual unless the mode is explicitly automatic', () => {
  for (const explicitAuto of [undefined, true]) {
    const inputs = {
      hidiffusion_t1_ratio: null,
      hidiffusion_t2_ratio: null,
      ...(explicitAuto === undefined ? {} : { hidiffusion_auto_ratios: explicitAuto }),
    };
    const source = buildInvocationNode(templates.float!, { x: 0, y: 0 });
    const graph = {
      nodes: [
        { id: source.id, type: 'float', inputs: { value: 0.7 } },
        { id: 'denoise', type: 'denoise_latents', inputs },
      ],
      edges: ['hidiffusion_t1_ratio', 'hidiffusion_t2_ratio'].map((targetField, index) => ({
        id: String(index),
        sourceNodeId: source.id,
        sourceField: 'value',
        targetNodeId: 'denoise',
        targetField,
      })),
    };
    const doc = previewGraphToDocument(graph, templates).document;
    expect(compileProjectGraph(doc, templates).backendGraph.nodes.denoise).toMatchObject({
      hidiffusion_auto_ratios: explicitAuto === true,
    });
    const node = doc.nodes.find((candidate) => candidate.id === 'denoise')!;
    if (!isInvocationNode(node)) {
      throw new Error('expected invocation');
    }
    if (explicitAuto === undefined) {
      delete node.data.inputs.hidiffusion_auto_ratios;
    }
    node.data.version = '1.7.0';
    expect(
      compileProjectGraph(updateWorkflowNodes(doc, templates).document, templates).backendGraph.nodes.denoise
    ).toMatchObject({ hidiffusion_auto_ratios: explicitAuto === true });
  }
});
