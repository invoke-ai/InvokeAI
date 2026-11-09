import type { WorkflowNode } from '@features/workflow/core/types';

import { createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { getWorkflowSilhouetteImage, layoutWorkflowSilhouette, MAX_SILHOUETTE_RECTS } from './workflowSilhouette';

const notes = (id: string, x: number, y: number): WorkflowNode => ({
  data: { label: '', notes: '' },
  id,
  position: { x, y },
  type: 'notes',
});

describe('layoutWorkflowSilhouette', () => {
  it('draws nothing for a document without positioned nodes', () => {
    expect(layoutWorkflowSilhouette([])).toEqual([]);
    expect(layoutWorkflowSilhouette([notes('a', Number.NaN, 0)])).toEqual([]);
  });

  it('fits the graph inside the padded 3:2 well, centred, keeping node proportions', () => {
    // Two 320×160 notes 640 apart: a 960×160 graph whose width binds, at 0.76 of the well's 1.5 units.
    const [left, right] = layoutWorkflowSilhouette([notes('a', 0, 0), notes('b', 640, 0)]);

    expect(left!.x).toBeCloseTo(0.12);
    expect(left!.width).toBeCloseTo(0.25333);
    expect(left!.y).toBeCloseTo(0.405);
    expect(left!.height).toBeCloseTo(0.19);
    expect(right!.x + right!.width).toBeCloseTo(0.88);
    expect(right!.y).toBeCloseTo(left!.y);
  });

  it('keeps a lone node from filling the well', () => {
    const [rect] = layoutWorkflowSilhouette([notes('a', 5000, -300)]);

    expect(rect!.width).toBeCloseTo(0.304);
    expect(rect!.x + rect!.width / 2).toBeCloseTo(0.5);
    expect(rect!.y + rect!.height / 2).toBeCloseTo(0.5);
  });

  it('draws a bounded, order-independent subset of a large graph inside the well', () => {
    const nodes = Array.from({ length: 500 }, (_, index) =>
      notes(`n${index}`, (index % 25) * 400, Math.floor(index / 25) * 200)
    );
    const rects = layoutWorkflowSilhouette(nodes);

    expect(rects).toHaveLength(MAX_SILHOUETTE_RECTS);
    expect(layoutWorkflowSilhouette([...nodes].reverse())).toEqual(rects);
    for (const rect of rects) {
      expect(rect.x).toBeGreaterThanOrEqual(0);
      expect(rect.y).toBeGreaterThanOrEqual(0);
      expect(rect.x + rect.width).toBeLessThanOrEqual(1);
      expect(rect.y + rect.height).toBeLessThanOrEqual(1);
    }
    // The subset spans the graph rather than clustering at one edge.
    expect(Math.min(...rects.map((rect) => rect.x))).toBeCloseTo(0.12);
    expect(Math.max(...rects.map((rect) => rect.x))).toBeGreaterThan(0.7);
  });

  it('draws a document once, as one SVG data URI holding a rect per drawn node', () => {
    const document = { ...createProjectGraph('silhouette'), nodes: [notes('a', 0, 0), notes('b', 640, 0)] };
    const image = getWorkflowSilhouetteImage(document);

    expect(image?.rectCount).toBe(2);
    expect(decodeURIComponent(image!.url).match(/<rect /g)).toHaveLength(2);
    expect(getWorkflowSilhouetteImage(document)).toBe(image);
    expect(getWorkflowSilhouetteImage(createProjectGraph('empty'))).toBeNull();
  });
});
