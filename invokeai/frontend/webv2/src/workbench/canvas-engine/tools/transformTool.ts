/**
 * The transform tool: an interactive scale/rotate/move SESSION on a single layer.
 *
 * Interaction contract (CANVAS_PLAN Phase 5):
 * - **Session**: selecting the tool with an eligible layer selected (or pressing
 *   the canvas with no session open) captures the SELECTED layer's committed
 *   transform and opens a session on it. A press never re-targets which layer is
 *   active — that is the layers panel's job. The live preview
 *   flows through the engine's transform-override channel; a `transformSession`
 *   store exposes the layer id + live transform so the numeric options bar can
 *   render and edit it. The session survives multiple gestures — drag handles,
 *   drag to rotate/move, adjust numerics — until **Apply** or **Cancel**.
 * - **Gestures** (each a fresh pointer drag on the session's frame): a scale
 *   handle scales about the opposite handle (alt = center, shift = uniform on
 *   corners); a corner rotate zone rotates about the center (shift = 15° snap);
 *   the interior moves (shift = axis constrain). Pointer-move only updates the
 *   session preview — it never dispatches.
 * - **Snap**: with the snap-to-grid setting on, a LAYER gesture lands its origin
 *   (move) or the dragged handle (scale) on the model grid. Alt bypasses the snap
 *   on a move; on a scale handle it does not, because alt already means
 *   scale-about-center there. Float gestures never snap — their transform is
 *   layer-local, so a document-space grid would skew rotated/scaled layers.
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

import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
import {
  applyMove,
  applyRotate,
  applyScale,
  resizeCursorForHandle,
  transformTargetAt,
} from '@workbench/canvas-engine/transform/transformMath';

import type { Tool, ToolContext } from './tool';

import { positionGrid } from './gridSnap';
import { hittableLayerRect, layerMatrix } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const TRANSFORM_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  target: TransformTarget;
  /** Whether this drag moves the layer session or a floating selection. */
  kind: 'layer' | 'float';
  /** The subject transform captured at gesture start (revert target for cancel). */
  startTransform: LayerTransform;
  /** The press point, in the SUBJECT's space (not necessarily document space). */
  startPointerDoc: Vec2;
  startScreen: Vec2;
  /** The subject's content rect, in its own space (off-origin aware). */
  rect: TransformRect;
  /** Converts a document-space pointer into the subject's space. */
  fromDocument: (point: Vec2) => Vec2;
  /** The cursor held for the duration of this gesture. */
  cursor: string;
  moved: boolean;
}

/** The cursor for a hovered/grabbed target, given the subject's current transform. */
const cursorForTarget = (transform: LayerTransform, target: TransformTarget): string => {
  if (target.kind === 'scale') {
    return resizeCursorForHandle(transform, target.handle);
  }
  return target.kind === 'rotate' ? 'grab' : 'move';
};

/**
 * What a transform gesture acts on. A layer works in DOCUMENT space; a floating
 * selection works in its layer's LOCAL space (that is where its pixels and its
 * transform live). The gesture math is space-agnostic as long as `rect`,
 * `transform`, and the pointer are all expressed in the same space — `toDocument`
 * / `fromDocument` bridge that space to the viewport, so both cases run through
 * exactly the same handle hit-testing and scale/rotate/move code.
 */
interface TransformSubject {
  readonly kind: 'layer' | 'float';
  readonly rect: TransformRect;
  readonly transform: LayerTransform;
  toDocument(point: Vec2): Vec2;
  fromDocument(point: Vec2): Vec2;
  /**
   * Rotation the subject's own space adds on top of `transform.rotation`, so a
   * resize cursor points the right way for a float on a rotated layer.
   */
  readonly spaceRotation: number;
}

const identityPoint = (point: Vec2): Vec2 => point;

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

  /**
   * What the tool currently acts on: a live floating selection takes precedence
   * over the layer session — the pixels are already detached, so the frame must
   * wrap them, not the layer they came from.
   */
  const subjectOf = (ctx: ToolContext): TransformSubject | null => {
    const doc = ctx.getDocument();
    if (!doc) {
      return null;
    }
    const float = ctx.getFloatingSelection?.() ?? null;
    if (float) {
      const layer = doc.layers.find((candidate) => candidate.id === float.layerId);
      if (!layer) {
        return null;
      }
      const matrix = layerMatrix(layer.transform);
      const inverse = invert(matrix);
      if (!inverse) {
        return null;
      }
      return {
        fromDocument: (point) => applyToPoint(inverse, point),
        kind: 'float',
        rect: float.pixels.rect,
        spaceRotation: layer.transform.rotation,
        toDocument: (point) => applyToPoint(matrix, point),
        transform: float.transform,
      };
    }
    const session = ctx.stores.transformSession.get();
    const layer = session ? doc.layers.find((candidate) => candidate.id === session.layerId) : undefined;
    const rect = session && layer ? hittableLayerRect(layer, doc) : null;
    if (!session || !rect) {
      return null;
    }
    return {
      fromDocument: identityPoint,
      kind: 'layer',
      rect,
      spaceRotation: 0,
      toDocument: identityPoint,
      transform: session.transform,
    };
  };

  /** Publishes a live transform to whichever subject the gesture is driving. */
  const publish = (ctx: ToolContext, kind: TransformSubject['kind'], transform: LayerTransform): void => {
    if (kind === 'float') {
      ctx.setFloatingTransform?.(transform);
    } else {
      ctx.updateTransformSession?.(transform);
    }
  };

  /** The cursor for a target, accounting for any rotation the subject's space adds. */
  const cursorFor = (subject: TransformSubject, transform: LayerTransform, target: TransformTarget): string =>
    cursorForTarget({ ...transform, rotation: transform.rotation + subject.spaceRotation }, target);

  /** Hit-tests the current subject's frame at a screen point (or `null`). */
  const targetAt = (ctx: ToolContext, subject: TransformSubject, screenPoint: Vec2): TransformTarget | null =>
    transformTargetAt({
      point: screenPoint,
      rect: subject.rect,
      toScreen: (p) => ctx.viewport.documentToScreen(subject.toDocument(p)),
      transform: subject.transform,
    });

  const nextTransform = (ctx: ToolContext, state: GestureState, input: PointerInput): LayerTransform => {
    // The pointer arrives in document space; the gesture math runs in the
    // subject's space, so convert before doing anything with it.
    const pointer = state.fromDocument(input.documentPoint);
    const delta: Vec2 = {
      x: pointer.x - state.startPointerDoc.x,
      y: pointer.y - state.startPointerDoc.y,
    };
    // A float works in its layer's LOCAL space, so a document-space grid would
    // skew it on a rotated/scaled layer — only layer gestures snap.
    const gridFor = (bypass: boolean): number => (state.kind === 'layer' ? positionGrid(ctx, bypass) : 0);
    switch (state.target.kind) {
      case 'move':
        return applyMove(state.startTransform, delta, input.modifiers.shift, gridFor(input.modifiers.alt));
      case 'scale':
        return applyScale({
          alt: input.modifiers.alt,
          // No alt bypass on a scale handle — alt already means scale-about-center.
          grid: gridFor(false),
          handle: state.target.handle,
          pointerDoc: pointer,
          shift: input.modifiers.shift,
          rect: state.rect,
          start: state.startTransform,
          startPointerDoc: state.startPointerDoc,
        });
      case 'rotate':
        return applyRotate({
          pointerDoc: pointer,
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
      // A live float IS the session — entering the tool frames the pixels in
      // flight, so no layer session is opened over the top of them.
      if (ctx.getFloatingSelection?.()) {
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
      // A float is NOT cancelled here — the engine banks it on a real switch, so
      // the pixels land rather than snapping back.
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
      // A float owns Apply/Cancel while it is in flight: baking it is what
      // "apply" means, and abandoning it is what "cancel" means.
      if (ctx.getFloatingSelection?.()) {
        if (command === 'apply') {
          ctx.commitFloatingSelection?.();
        } else {
          ctx.cancelFloatingSelection?.();
        }
        return;
      }
      if (command === 'apply') {
        ctx.applyTransform?.();
      } else {
        ctx.cancelTransform?.();
      }
    },
    onPointerCancel: (ctx) => {
      // Revert just this drag (keep the session / the float); Escape's own
      // session-level cancel is separate.
      if (gesture) {
        if (gesture.kind === 'float') {
          if (ctx.getFloatingSelection?.()) {
            ctx.setFloatingTransform?.(gesture.startTransform);
          }
        } else if (ctx.stores.transformSession.get()) {
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
      // 1) A framed subject (a float, or an open layer session): a press on its
      //    frame starts a scale/rotate/move gesture.
      const subject = subjectOf(ctx);
      if (subject) {
        const target = targetAt(ctx, subject, input.screenPoint);
        if (target) {
          gesture = {
            cursor: target.kind === 'rotate' ? 'grabbing' : cursorFor(subject, subject.transform, target),
            fromDocument: subject.fromDocument,
            kind: subject.kind,
            moved: false,
            rect: subject.rect,
            startPointerDoc: subject.fromDocument(input.documentPoint),
            startScreen: input.screenPoint,
            startTransform: subject.transform,
            target,
          };
          return;
        }
        // A press off the frame is a no-op; the subject persists.
        return;
      }

      // 2) Nothing framed yet: open a session on the SELECTED layer (the layers
      //    panel is the sole authority on which layer is active — a press never
      //    re-targets it) and start a move gesture.
      const selectedId = doc.selectedLayerId;
      const selected = selectedId ? doc.layers.find((layer) => layer.id === selectedId) : undefined;
      if (!selected || !isEligible(selected, doc)) {
        return;
      }
      const rect = hittableLayerRect(selected, doc);
      if (!rect) {
        return;
      }
      ctx.beginTransformSession?.(selected.id);
      gesture = {
        cursor: 'move',
        fromDocument: identityPoint,
        kind: 'layer',
        moved: false,
        rect,
        startPointerDoc: input.documentPoint,
        startScreen: input.screenPoint,
        startTransform: selected.transform,
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
        publish(ctx, gesture.kind, nextTransform(ctx, gesture, input));
        return;
      }
      // Idle hover over a framed subject: reflect the target under the pointer.
      const subject = subjectOf(ctx);
      if (!subject) {
        if (hoverCursor !== null) {
          hoverCursor = null;
          ctx.updateCursor();
        }
        return;
      }
      const target = targetAt(ctx, subject, input.screenPoint);
      const cursor = target ? cursorFor(subject, subject.transform, target) : 'default';
      if (cursor !== hoverCursor) {
        hoverCursor = cursor;
        ctx.updateCursor();
      }
    },
    onPointerUp: () => {
      // The subject's live transform already reflects the drag; keep it framed.
      endGesture();
    },
  };
};
