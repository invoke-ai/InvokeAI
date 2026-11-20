import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { VectorEditSession, VectorPointRef } from '@workbench/canvas-engine/vectorEditState';

import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { captureInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import { createVectorLayer } from '@workbench/canvas-engine/document/vectorLayerFactories';
import { anchorsToBezierPoints } from '@workbench/canvas-engine/math/bezierPath';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { normalizeCanvasDocumentContract } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { VectorEditingController } from './vectorEditingController';

const pointer = (x: number, y: number): PointerInput => ({
  documentPoint: { x, y },
  screenPoint: { x, y },
  buttons: 1,
  pressure: 0.5,
  pointerType: 'mouse',
  timeStamp: 0,
  modifiers: { alt: false, shift: false, ctrl: false, meta: false },
});
const harness = () => {
  const layer = createVectorLayer({ id: 'v', name: 'Vector' });
  Object.assign(layer, {
    guideColor: '#123456',
    fillColor: '#abcdef',
    fillEnabled: true,
    strokeEnabled: false,
    traceTaperEnds: true,
    traceTaper: 250,
  });
  layer.paths = [
    {
      id: 'p',
      name: null,
      isClosed: false,
      points: anchorsToBezierPoints([
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
      ]),
    },
  ];
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    type: 'replaceCanvasDocument',
    document: documentFrom([layer], layer.id),
  });
  let session: VectorEditSession | null = null,
    refuse = false,
    nextId = 0;
  const edits: { forward: CanvasProjectMutation; inverse: CanvasProjectMutation }[] = [],
    refusals: CanvasEditRefusal[] = [];
  const viewport = createViewport();
  const controller = new VectorEditingController({
    session: {
      get: () => session,
      set: (value) => {
        session = value;
      },
    },
    getDocument: () => project.canvas.document,
    canEdit: () => true,
    isGestureActive: () => false,
    getToolId: () => 'spline',
    createId: () => `new-${++nextId}`,
    invalidate: vi.fn(),
    viewport,
    reportRefusal: (reason) => refusals.push(reason),
    captureInsertionAnchor: (id) =>
      captureInsertionAnchor(project.canvas.document.stacks, {
        projectId: project.id,
        stack: 'vector',
        editRevision: 0,
        aboveId: id,
      }),
    commit: (_label, forward, inverse) => {
      if (refuse) {
        return { status: 'over-budget' };
      }
      edits.push({ forward, inverse });
      project = applyCanvasProjectMutation(project, forward);
      controller.onDocumentChanged();
      return { status: 'committed' };
    },
  });
  controller.onActivate();
  const apply = (mutation: CanvasProjectMutation) => {
    project = applyCanvasProjectMutation(project, mutation);
    controller.onDocumentChanged();
  };
  return {
    controller,
    edits,
    refusals,
    state: () => session!,
    document: () => project.canvas.document,
    refuse: () => {
      refuse = true;
    },
    apply,
    select: (refs: VectorPointRef[]) => {
      session = { ...session!, selectedPoints: refs };
      controller.refreshAvailability();
    },
    selectEndpoints: () => {
      session = {
        ...session!,
        selectedPoints: [
          { pathId: 'p', pointIndex: 0 },
          { pathId: 'p', pointIndex: 2 },
        ],
      };
      controller.refreshAvailability();
    },
  };
};

describe('permanent vector editing', () => {
  it('starts neutral, nudges a selected path through history, then clears activation on Escape', () => {
    const h = harness();
    expect(h.state().activePathId).toBeNull();
    expect(h.controller.targetContextMenu({ x: 50, y: 0 })).toBe(true);
    h.controller.nudge(10, 0);
    expect(h.state().paths[0]!.points[0]!.anchor.x).toBe(10);
    expect(h.edits).toHaveLength(1);
    h.apply(h.edits[0]!.inverse);
    expect(h.state().paths[0]!.points[0]!.anchor.x).toBe(0);
    h.apply(h.edits[0]!.forward);
    expect(h.state().paths[0]!.points[0]!.anchor.x).toBe(10);
    h.controller.onKeyCommand({} as ToolContext, 'cancel');
    expect(h.state().activePathId).toBeNull();
  });
  it('keeps the individual transform frame during nudge and cancels an unfinished point drag', () => {
    const h = harness();
    h.controller.targetContextMenu({ x: 50, y: 0 });
    h.controller.transformPath();
    h.controller.nudge(1, 10);
    expect(h.state().pathTransform?.transform).toMatchObject({ x: 1, y: 10 });
    expect(h.state().paths[0]!.points[0]!.anchor).toEqual({ x: 1, y: 10 });
    h.controller.onKeyCommand({} as ToolContext, 'cancel');
    h.controller.onPointerDown({} as ToolContext, pointer(1, 10));
    h.controller.onPointerMove({ updateCursor: vi.fn() } as unknown as ToolContext, pointer(20, 20));
    h.controller.onPointerCancel();
    expect(h.state().paths[0]!.points[0]!.anchor).toEqual({ x: 1, y: 10 });
    expect(h.edits).toHaveLength(1);
  });
  it('uses viewport distance consistently for three-point Join under non-uniform scaling', () => {
    const h = harness();
    const layer = h.document().stacks.vector![0]!;
    if (layer.type !== 'vector') {
      throw new Error('Expected vector');
    }
    h.apply({
      type: 'updateCanvasLayerConfig',
      id: layer.id,
      config: {
        layerType: 'vector',
        paths: [
          {
            ...layer.paths[0]!,
            points: anchorsToBezierPoints([
              { x: 0, y: 0 },
              { x: 5, y: 5 },
              { x: 10, y: 0 },
            ]),
          },
        ],
      },
    });
    h.apply({
      type: 'updateCanvasLayer',
      id: layer.id,
      patch: { transform: { ...layer.transform, scaleX: 100, scaleY: 0.01 } },
    });
    h.selectEndpoints();
    expect(h.state().canJoin).toBe(true);
    h.controller.join();
    expect(h.state().paths[0]!.isClosed).toBe(true);
  });
  it('splits and rejoins endpoints, nudges only a selected point and replays history', () => {
    const h = harness();
    h.controller.targetContextMenu({ x: 100, y: 0 });
    h.controller.nudge(0, 10);
    expect(h.state().paths[0]!.points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 100, y: 10 },
      { x: 100, y: 100 },
    ]);
    h.controller.split();
    expect(h.state().paths).toHaveLength(2);
    const [a, b] = h.state().paths;
    h.select([
      { pathId: a!.id, pointIndex: a!.points.length - 1 },
      { pathId: b!.id, pointIndex: 0 },
    ]);
    expect(h.state().canJoin).toBe(true);
    h.controller.join();
    expect(h.state().paths).toHaveLength(1);
    expect(h.state().paths[0]!.points).toHaveLength(3);
    const last = h.edits.at(-1)!;
    h.apply(last.inverse);
    expect(h.state().paths).toHaveLength(2);
    h.apply(last.forward);
    expect(h.state().paths).toHaveLength(1);
  });
  it('disables three-point welding when endpoints are close on screen despite a large local distance', () => {
    const h = harness(),
      layer = h.document().stacks.vector![0]!;
    if (layer.type !== 'vector') {
      throw new Error('Expected vector');
    }
    h.apply({
      type: 'updateCanvasLayerConfig',
      id: 'v',
      config: {
        layerType: 'vector',
        paths: [
          {
            ...layer.paths[0]!,
            points: anchorsToBezierPoints([
              { x: 0, y: 0 },
              { x: 5, y: 5 },
              { x: 0, y: 10 },
            ]),
          },
        ],
      },
    });
    h.apply({
      type: 'updateCanvasLayer',
      id: 'v',
      patch: { transform: { ...layer.transform, scaleX: 100, scaleY: 0.01 } },
    });
    h.selectEndpoints();
    expect(h.state().canJoin).toBe(false);
    h.controller.join();
    expect(h.edits).toHaveLength(0);
  });
  it('reports refused edits, restores committed geometry, and extracts a copy without retargeting', () => {
    const h = harness();
    h.apply({
      type: 'updateCanvasLayerConfig',
      id: 'v',
      config: { layerType: 'vector', strokeColor: '#987654', strokeWidth: 19 },
    });
    h.controller.targetContextMenu({ x: 50, y: 0 });
    h.controller.extractPath();
    expect(h.document().selectedLayerId).toBe('v');
    expect(h.document().stacks.vector).toHaveLength(2);
    const copy = h.document().stacks.vector![0]!;
    if (copy.type !== 'vector') {
      throw new Error('Expected vector');
    }
    expect(copy).toMatchObject({ strokeColor: '#987654', strokeWidth: 19 });
    expect(copy.paths[0]!.id).not.toBe('p');
    expect(copy.paths[0]!.points).toEqual(h.state().paths[0]!.points);
    const loaded = normalizeCanvasDocumentContract(JSON.parse(JSON.stringify(h.document())));
    expect(loaded?.version).toBe(7);
    expect(loaded?.stacks.vector?.[0]).toEqual(copy);
    h.refuse();
    h.controller.nudge(10, 0);
    expect(h.refusals).toEqual(['over-budget']);
    expect(h.state().paths[0]!.points[0]!.anchor.x).toBe(0);
  });
});
