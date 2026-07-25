import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { SamPreviewState } from '@workbench/canvas-engine/controllers/previewStateController';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type {
  OverlayCursor,
  OverlayState,
  TransformFrameOverlay,
} from '@workbench/canvas-engine/render/overlayRenderer';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Mat2d, ToolId, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { hittableLayerRect, layerMatrix, layerOutlineCorners } from '@workbench/canvas-engine/tools/moveHitTest';
import { transformOverlayGeometry } from '@workbench/canvas-engine/transform/transformMath';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';

/** The SAM preview's fixed overlay opacity — it sits over the layer it describes. */
const SAM_PREVIEW_OPACITY = 0.45;

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
  readonly getAntsPhase: () => number;
}

export interface OverlayFrame {
  /** Everything the overlay renderer draws this frame, gathered from live state. */
  describe(
    doc: CanvasDocumentContractV2,
    view: Mat2d,
    floatFrame: FloatingSelectionFrame | null,
    samPreview: SamPreviewState | null
  ): OverlayState;
}

/**
 * Projects engine state into the overlay renderer's descriptor.
 *
 * The overlay is chrome — bbox, grid, cursor ring, marching ants, tool
 * previews — and every part of it is derived, never owned. Keeping the
 * projection pure means the overlay's rules can be tested without a canvas:
 * which tool shows which affordance, that a live drag preview stands in for
 * committed state so the overlay tracks the gesture, and that a frame is
 * suppressed rather than drawn against stale geometry.
 */
export const createOverlayFrame = (deps: CreateOverlayFrameDeps): OverlayFrame => {
  const { getActiveToolId, selection, stores, transformOverrides } = deps;

  /**
   * The selected layer's bounds outline for the move tool. A layer mid-drag
   * (carrying a live override) wins over the committed selection so the marquee
   * tracks the preview rather than lagging a frame behind it.
   */
  const moveOutlineCorners = (doc: CanvasDocumentContractV2): readonly Vec2[] | null => {
    if (getActiveToolId() !== 'move') {
      return null;
    }
    const overridden = doc.layers.find((layer) => transformOverrides.has(layer.id));
    const target = overridden ?? doc.layers.find((layer) => layer.id === doc.selectedLayerId);
    if (!target) {
      return null;
    }
    return layerOutlineCorners(target, doc, transformOverrides.get(target.id) ?? null);
  };

  /** The transform-tool frame (rotated bounds + handles + rotation nub), or `null`. */
  const transformFrame = (doc: CanvasDocumentContractV2): TransformFrameOverlay | null => {
    if (getActiveToolId() !== 'transform') {
      return null;
    }
    const float = deps.getFloatingSelection();
    if (float) {
      // The float frames its own pixels. Its rect and transform are LAYER-LOCAL,
      // so the resulting geometry is projected out through the layer's matrix to
      // reach the document space the overlay draws in.
      const floatLayer = doc.layers.find((candidate) => candidate.id === float.layerId);
      if (!floatLayer) {
        return null;
      }
      const toDocument = layerMatrix(floatLayer.transform);
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
    const layer = doc.layers.find((candidate) => candidate.id === session.layerId);
    // The layer's LOCAL content rect (off-origin aware): the frame must wrap the
    // pixels where the compositor draws them, not an assumed origin-anchored box.
    const rect = layer ? hittableLayerRect(layer, doc) : null;
    if (!rect) {
      return null;
    }
    return transformOverlayGeometry(session.transform, rect);
  };

  return {
    describe: (doc, view, floatFrame, samPreview) => {
      const activeTool = getActiveToolId();
      // While the bbox tool drags, the transient preview stands in for the
      // committed frame so the rect and its handles track the gesture.
      const bboxPreview = stores.bboxPreview.get();
      const samSession = stores.samInteraction.get();
      return {
        bbox: bboxPreview ?? doc.bbox,
        bboxHandles: activeTool === 'bbox',
        bboxOverlay: stores.bboxOverlay.get(),
        cursor: deps.getOverlayCursor(),
        gradientPreview: stores.gradientPreview.get(),
        // The grid spans the whole viewport at the bbox snap size when the
        // setting is on; the document rect no longer bounds it.
        gridSize: stores.bboxGrid.get(),
        lassoPreview: stores.lassoPreview.get(),
        layerOutline: moveOutlineCorners(doc),
        // Ants advance via the animator's overlay-only ticks, and ride the
        // float's matrix so they track lifted pixels rather than the layer.
        marchingAnts: selection.hasSelection()
          ? { matrix: floatFrame?.ants ?? null, paths: selection.antsPaths(), phase: deps.getAntsPhase() }
          : null,
        marqueePreview: stores.marqueePreview.get(),
        ruleOfThirds: stores.ruleOfThirds.get(),
        samInput: samSession?.input.type === 'visual' ? samSession.input : null,
        samPreview: samPreview
          ? { opacity: SAM_PREVIEW_OPACITY, rect: samPreview.rect, surface: samPreview.data }
          : null,
        shapePreview: stores.shapePreview.get(),
        // The passive bbox frame follows the setting, but always renders while
        // the bbox tool is active so its handles have a frame to attach to.
        showBbox: stores.showBbox.get() || activeTool === 'bbox',
        showGrid: stores.showGrid.get(),
        transformFrame: transformFrame(doc),
        view,
      };
    },
  };
};
