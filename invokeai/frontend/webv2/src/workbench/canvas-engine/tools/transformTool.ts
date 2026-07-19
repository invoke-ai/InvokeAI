/**
 * The transform tool: an interactive scale/rotate/move SESSION on a single layer.
 *
 * Interaction contract (CANVAS_PLAN Phase 5):
 * - **Session**: selecting the tool with an eligible layer selected (or clicking a
 *   layer) captures its committed transform and opens a session. The live preview
 *   flows through the engine's transform-override channel; a `transformSession`
 *   store exposes the layer id + live transform so the numeric options bar can
 *   render and edit it. The session survives multiple gestures — drag handles,
 *   drag to rotate/move, adjust numerics — until **Apply** or **Cancel**.
 * - **Gestures** (each a fresh pointer drag on the session's frame): a scale
 *   handle scales about the opposite handle (alt = center, shift = uniform on
 *   corners); a corner rotate zone rotates about the center (shift = 15° snap);
 *   the interior moves (shift = axis constrain). Pointer-move only updates the
 *   session preview — it never dispatches.
 * - **Apply** (`enter` / options button): the engine commits — a param edit for
 *   image layers, a pixel bake for paint layers — as ONE undoable entry.
 * - **Cancel** (`esc` / options button / a REAL tool switch): drops the
 *   preview, no dispatch. A mid-gesture pointercancel reverts just that drag,
 *   keeping the session; Escape aborts the whole session. A TEMPORARY tool
 *   switch (space/alt modifier-hold) is not a cancel — the session and its
 *   preview survive the hold and resume when it ends (see `onActivate`/
 *   `onDeactivate`'s `opts.temporary`). If the session's layer is deleted
 *   while held, the engine's layer-change teardown cancels it regardless of
 *   which tool is active.
 *
 * Locked/hidden layers get no session (same guard as the move tool). Zero React,
 * zero import-time side effects.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerTransform, TransformRect, TransformTarget } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import {
  applyMove,
  applyRotate,
  applyScale,
  resizeCursorForHandle,
  transformTargetAt,
} from '@workbench/canvas-engine/transform/transformMath';

import type { Tool, ToolContext } from './tool';

import { hittableLayerRect, topLayerAt } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const TRANSFORM_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  target: TransformTarget;
  /** The session transform captured at gesture start (revert target for cancel). */
  startTransform: LayerTransform;
  startPointerDoc: Vec2;
  startScreen: Vec2;
  /** The session layer's local content rect (off-origin aware). */
  rect: TransformRect;
  /** The cursor held for the duration of this gesture. */
  cursor: string;
  moved: boolean;
}

/** The cursor for a hovered/grabbed target, given the layer's current transform. */
const cursorForTarget = (transform: LayerTransform, target: TransformTarget): string => {
  if (target.kind === 'scale') {
    return resizeCursorForHandle(transform, target.handle);
  }
  return target.kind === 'rotate' ? 'grab' : 'move';
};

/** Creates a fresh transform tool with its own session/gesture state. */
export const createTransformTool = (): Tool => {
  let gesture: GestureState | null = null;
  // The cursor for the target under the pointer while idle (session but no drag).
  let hoverCursor: string | null = null;

  const isEligible = (layer: CanvasLayerContract, doc: NonNullable<ReturnType<ToolContext['getDocument']>>): boolean =>
    // Masks are MOVE-able (legacy parity) but not transform-able in this phase:
    // `applyTransform` has no mask bake path, so a transform session on a mask
    // would preview then no-op on Apply. Exclude them until that lands (Phase 7+).
    layer.isEnabled &&
    !layer.isLocked &&
    layer.type !== 'inpaint_mask' &&
    layer.type !== 'regional_guidance' &&
    hittableLayerRect(layer, doc) !== null;

  /** Hit-tests the active session's frame at a screen point (or `null`). */
  const targetAt = (ctx: ToolContext, screenPoint: Vec2): TransformTarget | null => {
    const session = ctx.stores.transformSession.get();
    const doc = ctx.getDocument();
    if (!session || !doc) {
      return null;
    }
    const layer = doc.layers.find((candidate) => candidate.id === session.layerId);
    const rect = layer ? hittableLayerRect(layer, doc) : null;
    if (!rect) {
      return null;
    }
    return transformTargetAt({
      point: screenPoint,
      rect,
      toScreen: (p) => ctx.viewport.documentToScreen(p),
      transform: session.transform,
    });
  };

  const nextTransform = (state: GestureState, input: PointerInput): LayerTransform => {
    const delta: Vec2 = {
      x: input.documentPoint.x - state.startPointerDoc.x,
      y: input.documentPoint.y - state.startPointerDoc.y,
    };
    switch (state.target.kind) {
      case 'move':
        return applyMove(state.startTransform, delta, input.modifiers.shift);
      case 'scale':
        return applyScale({
          alt: input.modifiers.alt,
          handle: state.target.handle,
          pointerDoc: input.documentPoint,
          shift: input.modifiers.shift,
          rect: state.rect,
          start: state.startTransform,
          startPointerDoc: state.startPointerDoc,
        });
      case 'rotate':
        return applyRotate({
          pointerDoc: input.documentPoint,
          shift: input.modifiers.shift,
          rect: state.rect,
          start: state.startTransform,
          startPointerDoc: state.startPointerDoc,
        });
    }
  };

  const endGesture = (): void => {
    gesture = null;
  };

  return {
    cursor: () => {
      if (gesture) {
        return gesture.cursor;
      }
      return hoverCursor ?? 'default';
    },
    id: 'transform',
    onActivate: (ctx, opts) => {
      if (opts?.temporary) {
        // Resuming from a modifier-hold switch (space→view, alt→colorPicker):
        // `onDeactivate` preserved the session (and its preview override)
        // across the hold, so there is nothing to (re)open here. Re-opening
        // from the current selection would stomp the live preview with the
        // layer's committed transform, discarding accumulated drags/numeric
        // edits. If the session's layer vanished mid-hold, the engine's
        // layer-change teardown already cancelled it — leave that alone too.
        return;
      }
      // Entering the tool on an eligible selected layer opens a session on it.
      const doc = ctx.getDocument();
      const selectedId = doc?.selectedLayerId;
      const selected = selectedId ? doc?.layers.find((layer) => layer.id === selectedId) : undefined;
      if (doc && selected && isEligible(selected, doc)) {
        ctx.beginTransformSession?.(selected.id);
      }
    },
    onDeactivate: (ctx, opts) => {
      hoverCursor = null;
      if (opts?.temporary) {
        // A modifier-hold switch (space/alt) must not discard an in-progress
        // session: the pipeline already suppresses temp switches mid-gesture,
        // so `gesture` is guaranteed null here — this only clears the idle
        // hover cursor. The session + preview override are left for
        // `onActivate` to resume when the hold ends. A REAL tool switch (below)
        // still cancels.
        endGesture();
        return;
      }
      // A real tool switch mid-session cancels it (drops the preview, no dispatch).
      endGesture();
      ctx.cancelTransform?.();
    },
    onKeyCommand: (ctx, command) => {
      if (gesture) {
        // A live drag holds pointer capture (its own pointerup/pointercancel
        // will end it); applying or cancelling now would tear the gesture down
        // out from under the still-open pointer session, freezing the preview
        // mid-drag. No-op instead — mirrors `applyTransform`'s own mid-gesture
        // guard (`pipeline.isGestureActive()`) for the same reason.
        return;
      }
      if (command === 'apply') {
        ctx.applyTransform?.();
      } else {
        ctx.cancelTransform?.();
      }
    },
    onPointerCancel: (ctx) => {
      // Revert just this drag (keep the session); Escape's session cancel is separate.
      if (gesture) {
        const session = ctx.stores.transformSession.get();
        if (session) {
          ctx.updateTransformSession?.(gesture.startTransform);
        }
        endGesture();
        ctx.invalidate({ overlay: true });
      }
    },
    onPointerDown: (ctx, input) => {
      if (gesture || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const doc = ctx.getDocument();
      if (!doc) {
        return;
      }
      const session = ctx.stores.transformSession.get();

      // 1) An open session: a press on its frame starts a scale/rotate/move gesture.
      if (session) {
        const layer = doc.layers.find((candidate) => candidate.id === session.layerId);
        const rect = layer ? hittableLayerRect(layer, doc) : null;
        const target = rect ? targetAt(ctx, input.screenPoint) : null;
        if (rect && target) {
          gesture = {
            cursor: target.kind === 'rotate' ? 'grabbing' : cursorForTarget(session.transform, target),
            moved: false,
            rect,
            startPointerDoc: input.documentPoint,
            startScreen: input.screenPoint,
            startTransform: session.transform,
            target,
          };
          return;
        }
      }

      // 2) No session, or a press off the session frame: adopt the top-most eligible
      //    layer under the pointer and start a move gesture on it. Empty space is a
      //    no-op (the current session, if any, persists).
      const hit = topLayerAt(doc, input.documentPoint, (layer) => isEligible(layer, doc));
      if (!hit) {
        return;
      }
      const rect = hittableLayerRect(hit, doc);
      if (!rect) {
        return;
      }
      ctx.beginTransformSession?.(hit.id);
      gesture = {
        cursor: 'move',
        moved: false,
        rect,
        startPointerDoc: input.documentPoint,
        startScreen: input.screenPoint,
        startTransform: hit.transform,
        target: { kind: 'move' },
      };
    },
    onPointerMove: (ctx, input) => {
      if (gesture) {
        if (!gesture.moved) {
          const dxs = input.screenPoint.x - gesture.startScreen.x;
          const dys = input.screenPoint.y - gesture.startScreen.y;
          if (Math.hypot(dxs, dys) < TRANSFORM_DRAG_THRESHOLD_PX) {
            return;
          }
          gesture.moved = true;
        }
        ctx.updateTransformSession?.(nextTransform(gesture, input));
        return;
      }
      // Idle hover over a session frame: reflect the target under the pointer.
      const session = ctx.stores.transformSession.get();
      if (!session) {
        if (hoverCursor !== null) {
          hoverCursor = null;
          ctx.updateCursor();
        }
        return;
      }
      const target = targetAt(ctx, input.screenPoint);
      const cursor = target ? cursorForTarget(session.transform, target) : 'default';
      if (cursor !== hoverCursor) {
        hoverCursor = cursor;
        ctx.updateCursor();
      }
    },
    onPointerUp: () => {
      // The session's live transform already reflects the drag; keep the session.
      endGesture();
    },
  };
};
