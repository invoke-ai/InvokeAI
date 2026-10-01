import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SamPreviewState } from '@workbench/canvas-engine/controllers/previewStateController';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type {
  ColorLoupeOverlay,
  OverlayCursor,
  OverlayState,
  TransformFrameOverlay,
} from '@workbench/canvas-engine/render/overlayRenderer';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { ToolId, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLayer, lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex, type CanvasNodeEntry } from '@workbench/canvas-engine/document/documentIndex';
import { isGroupNode } from '@workbench/canvas-engine/document/documentTree';
import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { hittableLayerRect, layerOutlineCorners } from '@workbench/canvas-engine/tools/moveHitTest';
import { transformOverlayGeometry } from '@workbench/canvas-engine/transform/transformMath';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';

/** The SAM preview's resting overlay opacity — it sits over the layer it describes. */
const SAM_PREVIEW_OPACITY = 0.45;
/** Pulse amplitude and full-cycle period, matching legacy's 1s yoyo tween. */
const SAM_PREVIEW_PULSE = 0.15;
const SAM_PULSE_PERIOD_MS = 2000;

const samPreviewOpacity = (pulseTime: number | null): number =>
  pulseTime === null
    ? SAM_PREVIEW_OPACITY
    : SAM_PREVIEW_OPACITY + SAM_PREVIEW_PULSE * Math.sin((pulseTime / SAM_PULSE_PERIOD_MS) * 2 * Math.PI);

export type LayerTransformOverrides = ReadonlyMap<
  string,
  { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
>;

export interface CreateOverlayFrameDeps {
  readonly stores: EngineStores;
  readonly selection: SelectionState;
  readonly transformOverrides: LayerTransformOverrides;
  readonly getActiveToolId: () => ToolId;
  readonly getFloatingSelection: () => FloatingSelection | null;
  readonly getOverlayCursor: () => OverlayCursor | null;
  /** The picker's loupe over `doc`, sampled for this frame, or null while hidden. */
  readonly getColorLoupe: (doc: CanvasDocumentContractV3) => ColorLoupeOverlay | null;
  readonly getAntsPhase: () => number;
  /** The clock while the SAM pulse animates, `null` for the static opacity (reduced motion, no preview). */
  readonly getSamPulseTime: () => number | null;
}

export interface OverlayFrame {
  /** Everything the overlay renderer draws this frame, gathered from live state. */
  describe(
    doc: CanvasDocumentContractV3,
    screen: OverlayScreen,
    floatFrame: FloatingSelectionFrame | null,
    samPreview: SamPreviewState | null
  ): OverlayState;
}

/** The overlay's screen: document→CSS transform, CSS viewport size and device-pixel ratio. */
export type OverlayScreen = Pick<OverlayState, 'dpr' | 'view' | 'viewportSize'>;

/**
 * Pure projection of engine state into overlay descriptors. Live previews replace committed geometry; stale frames
 * are suppressed. The overlay owns no interaction state.
 */
export const createOverlayFrame = (deps: CreateOverlayFrameDeps): OverlayFrame => {
  const { getActiveToolId, selection, stores, transformOverrides } = deps;

  /** The topmost leaf carrying a live override, resolved from the override ids rather than a leaf scan. */
  const firstOverriddenLayer = (doc: CanvasDocumentContractV3): CanvasLayerContract | null => {
    let first: CanvasNodeEntry | null = null;
    const index = getDocumentIndex(doc);
    for (const id of transformOverrides.keys()) {
      const entry = index.byId.get(id);
      if (entry && !isGroupNode(entry.node) && (first === null || entry.order < first.order)) {
        first = entry;
      }
    }
    return first ? (first.node as CanvasLayerContract) : null;
  };

  /**
   * The selected layer's bounds outline for the move tool. A layer mid-drag
   * (carrying a live override) wins over the committed selection so the marquee
   * tracks the preview rather than lagging a frame behind it.
   */
  const moveOutlineCorners = (doc: CanvasDocumentContractV3): readonly Vec2[] | null => {
    if (getActiveToolId() !== 'move') {
      return null;
    }
    const target =
      firstOverriddenLayer(doc) ?? (doc.selectedLayerId ? lookupDocumentLayer(doc, doc.selectedLayerId) : null);
    if (!target) {
      return null;
    }
    return layerOutlineCorners(target, doc, transformOverrides.get(target.id) ?? null);
  };

  /** The transform-tool frame (rotated bounds + handles + rotation nub), or `null`. */
  const transformFrame = (doc: CanvasDocumentContractV3): TransformFrameOverlay | null => {
    if (getActiveToolId() !== 'transform') {
      return null;
    }
    const float = deps.getFloatingSelection();
    if (float) {
      // The float frames its own pixels. Its rect and transform are LAYER-LOCAL,
      // so the resulting geometry is projected out through the layer's matrix to
      // reach the document space the overlay draws in.
      const floatLeaf = lookupDocumentLeaf(doc, float.layerId);
      if (!floatLeaf) {
        return null;
      }
      const toDocument = floatLeaf.worldTransform;
      const geometry = transformOverlayGeometry(float.transform, float.pixels.rect);
      return {
        center: applyToPoint(toDocument, geometry.center),
        corners: geometry.corners.map((point) => applyToPoint(toDocument, point)),
        handles: geometry.handles.map((point) => applyToPoint(toDocument, point)),
        rotationAnchor: applyToPoint(toDocument, geometry.rotationAnchor),
      };
    }
    const session = stores.transformSession.get();
    if (!session) {
      return null;
    }
    const layer = lookupDocumentLayer(doc, session.layerId);
    // Frame actual local content bounds, including off-origin pixels.
    const rect = layer ? hittableLayerRect(layer, doc) : null;
    if (!rect) {
      return null;
    }
    return transformOverlayGeometry(session.transform, rect);
  };

  return {
    describe: (doc, screen, floatFrame, samPreview) => {
      const activeTool = getActiveToolId();
      const bboxPreview = stores.bboxPreview.get();
      const samSession = stores.samInteraction.get();
      return {
        bbox: bboxPreview ?? doc.bbox,
        bboxHandles: activeTool === 'bbox',
        bboxOverlay: stores.bboxOverlay.get(),
        // The checker's darker square is the theme's canvas surround (`bg.inset`).
        bboxOverlayColor: stores.checkerColors.get().a,
        colorLoupe: deps.getColorLoupe(doc),
        cursor: deps.getOverlayCursor(),
        gradientPreview: stores.gradientPreview.get(),
        // Grid spans the viewport at bbox snap size, independent of document bounds.
        gridSize: stores.bboxGrid.get(),
        lassoPreview: stores.lassoPreview.get(),
        layerOutline: moveOutlineCorners(doc),
        // Overlay-only ants ticks use the float matrix to follow lifted pixels.
        marchingAnts: selection.hasSelection()
          ? { matrix: floatFrame?.ants ?? null, paths: selection.antsPaths(), phase: deps.getAntsPhase() }
          : null,
        marqueePreview: stores.marqueePreview.get(),
        ruleOfThirds: stores.ruleOfThirds.get(),
        samInput: samSession?.input.type === 'visual' ? samSession.input : null,
        samPreview: samPreview
          ? {
              opacity: samPreviewOpacity(deps.getSamPulseTime()),
              outline: samPreview.outline ?? null,
              phase: deps.getAntsPhase(),
              rect: samPreview.rect,
              surface: samPreview.data,
            }
          : null,
        shapePreview: stores.shapePreview.get(),
        // The passive bbox frame follows the setting, but always renders while
        // the bbox tool is active so its handles have a frame to attach to.
        showBbox: stores.showBbox.get() || activeTool === 'bbox',
        showGrid: stores.showGrid.get(),
        transformFrame: transformFrame(doc),
        ...screen,
      };
    },
  };
};
