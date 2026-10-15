import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it } from 'vitest';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

import { CONTENT_VISIBILITY_ZOOM } from './InvocationFlowNode';
import { WORKFLOW_INITIAL_RENDER_NODE_COUNT } from './performanceConstants';
import {
  getInitialRenderFlowModel,
  getRenderedFlowModel,
  getZoomedOutMountViewport,
  WorkflowEditorPreparingState,
} from './WorkflowEditorView';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const testI18n = createInstance();
await testI18n.init({
  initAsync: false,
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});

const createNode = (id: string, x: number): WorkflowFlowNode => ({
  data: { documentNode: { data: { label: '', notes: '' }, id, position: { x, y: 0 }, type: 'notes' } },
  id,
  position: { x, y: 0 },
  type: 'notes',
});

const createEdge = (id: string, source: string, target: string): WorkflowFlowEdge => ({
  data: {
    fieldTypeLabel: null,
    pathType: 'default',
    stroke: 'var(--xy-edge-stroke)',
    strokeWidth: 2,
  },
  id,
  source,
  sourceHandle: 'value',
  target,
  targetHandle: 'value',
  type: 'default',
});

describe('getInitialRenderFlowModel', () => {
  it('keeps the nearest initial nodes and filters edges to that window', () => {
    const nodes = Array.from({ length: WORKFLOW_INITIAL_RENDER_NODE_COUNT + 2 }, (_, index) =>
      createNode(`node-${index}`, index)
    );
    const model = {
      nodes,
      edges: [
        createEdge('inside', 'node-0', 'node-1'),
        createEdge('outside', 'node-0', `node-${WORKFLOW_INITIAL_RENDER_NODE_COUNT + 1}`),
      ],
    };

    const windowed = getInitialRenderFlowModel(model, { x: 0, y: 0, zoom: 1 });

    expect(windowed.nodes).toHaveLength(WORKFLOW_INITIAL_RENDER_NODE_COUNT);
    expect(windowed.nodes.map((node) => node.id)).toContain('node-0');
    expect(windowed.nodes.map((node) => node.id)).not.toContain(`node-${WORKFLOW_INITIAL_RENDER_NODE_COUNT + 1}`);
    expect(windowed.edges.map((edge) => edge.id)).toEqual(['inside']);
  });

  it('keeps the visible large graph windowed until its initial mount completes', () => {
    const nodes = Array.from({ length: WORKFLOW_INITIAL_RENDER_NODE_COUNT + 2 }, (_, index) =>
      createNode(`node-${index}`, index)
    );
    const model = {
      nodes,
      edges: [
        createEdge('inside', 'node-0', 'node-1'),
        createEdge('outside', 'node-0', `node-${WORKFLOW_INITIAL_RENDER_NODE_COUNT + 1}`),
      ],
    };

    const rendered = getRenderedFlowModel(
      model,
      { x: 0, y: 0, zoom: 1 },
      {
        isFullGraphMounted: false,
        isLargeGraph: true,
      }
    );

    expect(rendered?.nodes).toHaveLength(WORKFLOW_INITIAL_RENDER_NODE_COUNT);
    expect(rendered?.edges.map((edge) => edge.id)).toEqual(['inside']);
  });
});

describe('getZoomedOutMountViewport', () => {
  const target = (id: string, x: number, y: number) => ({ id, position: { x, y } });
  const container = { height: 900, width: 450 };

  it('keeps the default viewport while the fit could still land at a readable zoom', () => {
    expect(getZoomedOutMountViewport([target('a', 0, 0), target('b', 1000, 0)], container)).toBeNull();
    expect(getZoomedOutMountViewport([target('a', 0, 0), target('b', 0, 2000)], container)).toBeNull();
  });

  it('opens below the content zoom when the positions alone are wider or taller than that zoom shows', () => {
    const wide = getZoomedOutMountViewport([target('a', 0, 0), target('b', 1200, 0)], container);
    const tall = getZoomedOutMountViewport([target('a', 0, 0), target('b', 0, 2400)], container);

    expect(wide?.zoom).toBeLessThan(CONTENT_VISIBILITY_ZOOM);
    expect(tall?.zoom).toBeLessThan(CONTENT_VISIBILITY_ZOOM);
  });

  it('counts the fit’s padding, which lowers the zoom it can land at', () => {
    // 450 px less XYFlow's 10% padding leaves 410 px, which shows 1025 px of positions at the content zoom.
    expect(getZoomedOutMountViewport([target('a', 0, 0), target('b', 1030, 0)], container)?.zoom).toBeLessThan(
      CONTENT_VISIBILITY_ZOOM
    );
    expect(getZoomedOutMountViewport([target('a', 0, 0), target('b', 1020, 0)], container)).toBeNull();
  });

  it('frames the positions in the container', () => {
    const viewport = getZoomedOutMountViewport([target('a', -500, 100), target('b', 2500, 700)], container);

    expect(viewport).not.toBeNull();
    const { x, y, zoom } = viewport!;

    for (const position of [
      { x: -500, y: 100 },
      { x: 2500, y: 700 },
    ]) {
      expect(position.x * zoom + x).toBeGreaterThanOrEqual(0);
      expect(position.x * zoom + x).toBeLessThanOrEqual(container.width);
      expect(position.y * zoom + y).toBeGreaterThanOrEqual(0);
      expect(position.y * zoom + y).toBeLessThanOrEqual(container.height);
    }
  });

  it('keeps the default viewport for a single position', () => {
    expect(getZoomedOutMountViewport([target('a', 4000, 4000)], container)).toBeNull();
    expect(getZoomedOutMountViewport([], container)).toBeNull();
  });
});

describe('WorkflowEditorPreparingState', () => {
  const sentence = (nodeCount: number, edgeCount: number) =>
    renderToStaticMarkup(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={testI18n}>
          <WorkflowEditorPreparingState edgeCount={edgeCount} nodeCount={nodeCount} />
        </I18nextProvider>
      </ChakraProvider>
    ).match(/Loading [^<]*/)?.[0];

  it('counts nodes and edges with their own plural forms and grouped digits', () => {
    expect(sentence(1, 1)).toBe('Loading 1 node and 1 edge.');
    expect(sentence(1234, 2)).toBe('Loading 1,234 nodes and 2 edges.');
  });
});
