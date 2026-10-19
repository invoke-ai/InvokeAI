/**
 * Transform sessions frame the selected eligible layer without retargeting and span multiple gestures/numeric
 * edits. Handles scale about opposite anchors (Alt center, Shift uniform); rotate zones use Shift 15-degree snap;
 * interior moves with axis lock.
 *
 * Layer origins/handles snap to the model grid; Alt bypasses move snapping but retains center-scale meaning.
 * Local-space floats never snap. Apply makes one parameter edit or paint bake. Pointercancel reverts only its
 * drag; Escape/real switches cancel the session. Temporary switches preserve edits, and deletion always tears them
 * down.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerTransform, TransformRect, TransformTarget } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
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
 * Layer gestures use document space; floats use layer-local space. Convert pointers through to/fromDocument so
 * bounds, transforms and gesture math share one coordinate system.
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
    // Masks lack an applyTransform bake path, so reject sessions that would preview but fail to apply.
    isLeafEditable(lookupDocumentLeaf(doc, layer.id)) &&
    layer.type !== 'inpaint_mask' &&
    layer.type !== 'regional_guidance' &&
    hittableLayerRect(layer, doc) !== null;

  /** Live floats take precedence over layer sessions so the frame follows detached pixels. */
  const subjectOf = (ctx: ToolContext): TransformSubject | null => {
    const doc = ctx.getDocument();
    if (!doc) {
      return null;
    }
    const float = ctx.getFloatingSelection?.() ?? null;
    if (float) {
      const layer = getDocumentLayer(doc, float.layerId);
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
    const layer = session ? getDocumentLayer(doc, session.layerId) : undefined;
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
        // Temporary reactivation preserves existing edits and does not reopen from committed transforms. If
        // deletion cancelled the session mid-hold, leave it closed.
        return;
      }
      // A live float already owns the session; do not open a layer session over it.
      if (ctx.getFloatingSelection?.()) {
        return;
      }
      // Entering the tool on an eligible selected layer opens a session on it.
      const doc = ctx.getDocument();
      const selectedId = doc?.selectedLayerId;
      const selected = selectedId ? getDocumentLayer(doc, selectedId) : undefined;
      if (doc && selected && isEligible(selected, doc)) {
        ctx.beginTransformSession?.(selected.id);
      }
    },
    onDeactivate: (ctx, opts) => {
      hoverCursor = null;
      if (opts?.temporary) {
        // Temporary switches occur only between gestures and clear hover state while preserving session/preview.
        // Real switches cancel.
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
        // Ignore apply/cancel during captured drags to avoid tearing down the still-active pointer session.
        return;
      }
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

      // Open an unframed session on panel selection and begin movement without retargeting.
      const selectedId = doc.selectedLayerId;
      const selected = selectedId ? getDocumentLayer(doc, selectedId) : undefined;
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
