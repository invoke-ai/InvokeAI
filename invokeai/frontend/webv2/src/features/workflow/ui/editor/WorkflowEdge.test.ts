import { Position } from '@xyflow/react';
import { createInstance } from 'i18next';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it } from 'vitest';

import type { WorkflowEdgeData } from './flowAdapters';

import { WorkflowEdge } from './WorkflowEdge';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const testI18n = createInstance();
await testI18n.init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});

const imageEdge: WorkflowEdgeData = {
  fieldTypeLabel: 'Image',
  pathType: 'default',
  stroke: '#c4b5fd',
  strokeWidth: 2,
};

const props = {
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

const titlesOf = (data: WorkflowEdgeData | undefined) =>
  [
    ...renderToStaticMarkup(
      createElement(I18nextProvider, { i18n: testI18n }, createElement(WorkflowEdge, { ...props, data }))
    ).matchAll(/<title>([^<]*)<\/title>/g),
  ].map((match) => match[1]);

describe('WorkflowEdge', () => {
  it('attaches the field type tooltip to visible and interactive edge paths', () => {
    expect(titlesOf(imageEdge)).toEqual(['Image', 'Image']);
  });

  it('names batch, loop linkage and untyped edges in the UI language', () => {
    expect(titlesOf({ ...imageEdge, isBatch: true })).toEqual(['Image batch', 'Image batch']);
    expect(titlesOf({ ...imageEdge, fieldTypeLabel: null, isLoopLinkage: true })).toEqual([
      'Loop linkage',
      'Loop linkage',
    ]);
    expect(titlesOf({ ...imageEdge, fieldTypeLabel: null })).toEqual(['Unknown field type', 'Unknown field type']);
    expect(titlesOf(undefined)).toEqual(['Unknown field type', 'Unknown field type']);
  });
});
