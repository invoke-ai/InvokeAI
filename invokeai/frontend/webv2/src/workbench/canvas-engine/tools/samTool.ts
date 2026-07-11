import type { SamVisualInput } from '@workbench/canvas-engine/engineStores';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import { canonicalizeDocumentSamPoint } from '@workbench/canvas-engine/samCoordinates';

import type { SamHitTarget } from './samHitTest';
import type { Tool, ToolContext } from './tool';

import { clipPointToRect, moveSamBbox, rectFromPoints, resizeSamBbox, samHitTest } from './samHitTest';

const PRIMARY_BUTTON = 1;
export const SAM_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  startInput: SamVisualInput;
  target: SamHitTarget | null;
  moved: boolean;
}

const cloneInput = (input: SamVisualInput): SamVisualInput => ({
  bbox: input.bbox ? { ...input.bbox } : null,
  excludePoints: input.excludePoints.map((point) => ({ ...point })),
  includePoints: input.includePoints.map((point) => ({ ...point })),
  type: 'visual',
});

const flippedLabel = (label: 'include' | 'exclude'): 'include' | 'exclude' =>
  label === 'include' ? 'exclude' : 'include';

export const createSamTool = (): Tool => {
  let gesture: GestureState | null = null;

  const publish = (ctx: ToolContext, input: SamVisualInput): void => {
    ctx.updateSamInput?.(input);
    ctx.invalidate({ overlay: true });
  };

  const updateGesture = (ctx: ToolContext, input: PointerInput, state: GestureState): void => {
    const snapshot = ctx.stores.samSession.get();
    if (!snapshot) {
      return;
    }
    const next = cloneInput(state.startInput);
    const delta = {
      x: input.documentPoint.x - state.startDoc.x,
      y: input.documentPoint.y - state.startDoc.y,
    };

    if (state.target?.kind === 'point') {
      const points = state.target.label === 'include' ? next.includePoints : next.excludePoints;
      points[state.target.index] = clipPointToRect(input.documentPoint, snapshot.sourceRect);
    } else if (state.target?.kind === 'bbox-body' && next.bbox) {
      next.bbox = moveSamBbox(next.bbox, delta.x, delta.y, snapshot.sourceRect);
    } else if (state.target?.kind === 'bbox-handle' && next.bbox) {
      next.bbox = resizeSamBbox({ bounds: snapshot.sourceRect, delta, handle: state.target.handle, start: next.bbox });
    } else {
      const created = rectFromPoints(state.startDoc, input.documentPoint, snapshot.sourceRect);
      next.bbox = created.width >= 1 && created.height >= 1 ? created : state.startInput.bbox;
    }
    publish(ctx, next);
  };

  const cancelGesture = (ctx: ToolContext): void => {
    const current = gesture;
    gesture = null;
    if (current?.moved) {
      publish(ctx, cloneInput(current.startInput));
    }
  };

  return {
    cursor: () => 'crosshair',
    id: 'sam',
    onDeactivate: (ctx) => cancelGesture(ctx),
    onPointerCancel: (ctx) => cancelGesture(ctx),
    onPointerDown: (ctx, input) => {
      const snapshot = ctx.stores.samSession.get();
      if (
        gesture ||
        !snapshot ||
        (input.buttons & PRIMARY_BUTTON) === 0 ||
        !canonicalizeDocumentSamPoint(input.documentPoint, snapshot.sourceRect)
      ) {
        return;
      }
      gesture = {
        moved: false,
        startDoc: input.documentPoint,
        startInput: cloneInput(snapshot.input),
        startScreen: input.screenPoint,
        target: samHitTest({
          bbox: snapshot.input.bbox,
          excludePoints: snapshot.input.excludePoints,
          includePoints: snapshot.input.includePoints,
          screenPoint: input.screenPoint,
          view: ctx.viewport.viewMatrix(1),
        }),
      };
    },
    onPointerMove: (ctx, input) => {
      if (!gesture) {
        return;
      }
      if (!gesture.moved) {
        const dx = input.screenPoint.x - gesture.startScreen.x;
        const dy = input.screenPoint.y - gesture.startScreen.y;
        if (Math.hypot(dx, dy) < SAM_DRAG_THRESHOLD_PX) {
          return;
        }
        gesture.moved = true;
      }
      updateGesture(ctx, input, gesture);
    },
    onPointerUp: (ctx, input) => {
      const current = gesture;
      gesture = null;
      if (!current) {
        return;
      }
      if (current.moved) {
        updateGesture(ctx, input, current);
        return;
      }

      const snapshot = ctx.stores.samSession.get();
      if (!snapshot) {
        return;
      }
      const next = cloneInput(current.startInput);
      if (current.target?.kind === 'point') {
        const points = current.target.label === 'include' ? next.includePoints : next.excludePoints;
        points.splice(current.target.index, 1);
      } else {
        const label = input.modifiers.shift ? flippedLabel(snapshot.pointLabel) : snapshot.pointLabel;
        const points = label === 'include' ? next.includePoints : next.excludePoints;
        points.push(clipPointToRect(input.documentPoint, snapshot.sourceRect));
      }
      publish(ctx, next);
    },
  };
};
