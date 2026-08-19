import type { FieldInputTemplate, InvocationTemplate, ProjectGraphState } from '@features/workflow/contracts';

import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { buildLibraryGraphPreviewSource } from './libraryPreviewSource';

const input = (name: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
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
  type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const template = (type: string, inputs: Record<string, FieldInputTemplate>): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs: {
    out: {
      description: '',
      name: 'out',
      title: 'Out',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
    },
  },
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const templates = {
  sink: template('sink', { text: input('text', { required: true }) }),
  source: template('source', { value: input('value', { default: 'hello' }) }),
};

const buildDocument = (): { doc: ProjectGraphState; sinkId: string; sourceId: string } => {
  const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
  const sinkNode = buildInvocationNode(templates.sink, { x: 240, y: 80 });
  let doc = createProjectGraph('library-preview-fixture');

  doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
  doc = projectGraphReducer(doc, {
    edge: {
      id: 'e1',
      source: sourceNode.id,
      sourceHandle: 'out',
      target: sinkNode.id,
      targetHandle: 'text',
      type: 'default',
    },
    type: 'addEdge',
  });

  return { doc, sinkId: sinkNode.id, sourceId: sourceNode.id };
};

describe('buildLibraryGraphPreviewSource', () => {
  it('compiles the document into a graph with its nodes and edges', () => {
    const { doc, sinkId, sourceId } = buildDocument();

    const result = buildLibraryGraphPreviewSource(doc, templates);

    expect(result.invalidReasons).toEqual([]);
    expect(result.graph?.nodes.map((node) => node.id).sort()).toEqual([sinkId, sourceId].sort());
    expect(result.graph?.edges).toHaveLength(1);
    expect(result.graph?.edges[0]).toMatchObject({ sourceNodeId: sourceId, targetNodeId: sinkId });
  });

  it('is never live and carries no destination or notices', () => {
    const { doc } = buildDocument();

    const result = buildLibraryGraphPreviewSource(doc, templates);

    expect(result.isLive).toBe(false);
    expect(result.destinationLabel).toBeNull();
    expect(result.notices).toEqual([]);
  });

  it('adds no summary rows of its own — the side panel already counts the nodes', () => {
    const { doc } = buildDocument();

    const result = buildLibraryGraphPreviewSource(doc, templates);

    // A 'nodes' row here rendered a second "Nodes" line under the panel's own.
    expect(result.summaryRows).toEqual([]);
  });

  it('maps position hints from the document node positions', () => {
    const { doc, sinkId, sourceId } = buildDocument();

    const result = buildLibraryGraphPreviewSource(doc, templates);

    expect(result.positionHints?.[sourceId]).toEqual({ x: 0, y: 0 });
    expect(result.positionHints?.[sinkId]).toEqual({ x: 240, y: 80 });
  });

  it('surfaces a compile failure as an invalid reason instead of throwing', () => {
    const { doc } = buildDocument();
    // A malformed cached document (the shape a corrupted library record could
    // produce) — `nodes` is not an array, so compilation throws instead of
    // silently producing an empty graph.
    const corrupt = { ...doc, nodes: null } as unknown as ProjectGraphState;

    const result = buildLibraryGraphPreviewSource(corrupt, templates);

    expect(result.graph).toBeNull();
    expect(result.invalidReasons).toHaveLength(1);
    expect(result.invalidReasons[0]).toEqual(expect.any(String));
    expect(result.summaryRows).toEqual([]);
  });
});
