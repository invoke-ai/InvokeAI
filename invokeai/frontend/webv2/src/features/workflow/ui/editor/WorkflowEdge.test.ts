import { Position } from '@xyflow/react';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { WorkflowEdgeData } from './flowAdapters';

import { WorkflowEdge } from './WorkflowEdge';

const data: WorkflowEdgeData = {
  fieldTypeLabel: 'Image',
  pathType: 'default',
  stroke: '#c4b5fd',
  strokeWidth: 2,
  tooltip: 'Image',
};

const props = {
  data,
  id: 'edge-1',
  markerEnd: undefined,
  selected: false,
  source: 'a',
  sourcePosition: Position.Right,
  sourceX: 0,
  sourceY: 0,
  target: 'b',
  targetPosition: Position.Left,
  targetX: 100,
  targetY: 0,
};

describe('WorkflowEdge', () => {
  it('attaches the field type tooltip to visible and interactive edge paths', () => {
    const markup = renderToStaticMarkup(createElement(WorkflowEdge, props));

    expect(markup).toContain('react-flow__edge-path');
    expect(markup).toContain('react-flow__edge-interaction');
    expect(markup.match(/<title>Image<\/title>/g)).toHaveLength(2);
  });
});
