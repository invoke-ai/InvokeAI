import type { CanvasEditRefusal, StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasBezierPathContract,
  CanvasBezierPointContract,
  CanvasDocumentContractV3,
  CanvasVectorLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform, TransformTarget } from '@workbench/canvas-engine/transform/transformMath';
import type { Mat2d, PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { VectorEditSession, VectorPointRef } from '@workbench/canvas-engine/vectorEditState';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { leafEditRefusal } from '@workbench/canvas-engine/document/layerEligibility';
import { createVectorLayer, nextVectorLayerName } from '@workbench/canvas-engine/document/vectorLayerFactories';
import {
  getBezierPathBounds,
  transformBezierPath,
  getBezierPointPullHandleType,
  setBezierPointHandle,
  setBezierPointType,
  smoothBezierPathPoints,
  splitBezierSegmentAt,
} from '@workbench/canvas-engine/math/bezierPath';
import { applyToPoint, getScale, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { vectorHandleLayout } from '@workbench/canvas-engine/math/vectorHandleLayout';
import { VectorHitTestCache } from '@workbench/canvas-engine/math/vectorHitTest';
import {
  canJoinVectorPathEndpoints,
  canSplitVectorPathAtPoints,
  deleteVectorPathPoints,
  joinVectorPathEndpoints,
  splitVectorPathAtPoints,
} from '@workbench/canvas-engine/math/vectorPathTopology';
import {
  applyMove,
  applyRotate,
  applyScale,
  isCornerHandle,
  layerTransformMatrix,
  resizeCursorForHandle,
  transformTargetAt,
} from '@workbench/canvas-engine/transform/transformMath';

import { addedVectorLayerMutations } from './editSteps';

interface Options {
  session: { get(): VectorEditSession | null; set(value: VectorEditSession | null): void };
  getDocument(): CanvasDocumentContractV3 | null;
  canEdit(): boolean;
  isGestureActive(): boolean;
  getToolId(): ToolId;
  commit(label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): StructuralCommitResult;
  invalidate(): void;
  reportRefusal?(refusal: CanvasEditRefusal): void;
  createId(): string;
  captureInsertionAnchor(layerId: string): CanvasNodeInsertionAnchor;
  viewport: Viewport;
}

type Snapshot = Pick<VectorEditSession, 'paths' | 'activePathId' | 'selectedPoints'>;
type Drag = {
  before: Snapshot;
  cancelBefore?: Snapshot;
  start: Vec2;
  screen: Vec2;
  kind: 'anchor' | 'path' | 'pull' | 'maybe-pull' | 'inHandle' | 'outHandle' | 'rect';
  ref?: VectorPointRef;
  additive?: boolean;
  missing?: 'inHandle' | 'outHandle';
  handleOrigin?: Vec2;
};
const sameRef = (a: VectorPointRef, b: VectorPointRef): boolean =>
  a.pathId === b.pathId && a.pointIndex === b.pointIndex;
const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
// Hit radii are CSS viewport pixels: project geometry through screenMatrix() before comparing.
const ANCHOR_HIT_RADIUS_PX = 12;
const HANDLE_HIT_RADIUS_PX = 12;
const snapshot = (state: VectorEditSession): Snapshot => ({
  paths: state.paths,
  activePathId: state.activePathId,
  selectedPoints: state.selectedPoints,
});
const samePaths = (a: CanvasBezierPathContract[], b: CanvasBezierPathContract[]): boolean =>
  a === b || JSON.stringify(a) === JSON.stringify(b);

/** Owns point selection and gesture previews; completed changes commit directly to document history. */
export class VectorEditingController implements Tool {
  readonly id = 'spline';
  private base: CanvasVectorLayerContract | null = null;
  private readonly hits = new VectorHitTestCache();
  private drag: Drag | null = null;
  private hoverPath = false;
  private transformOrigin: CanvasBezierPathContract | null = null;
  private transformDrag: { target: TransformTarget; start: Vec2; transform: LayerTransform; before: Snapshot } | null =
    null;
  private transformCursor: string | null = null;

  constructor(private readonly options: Options) {}

  hasSession = (): boolean => this.options.session.get() !== null;
  private availability(state: VectorEditSession): Pick<VectorEditSession, 'canSplit' | 'canJoin'> {
    return {
      canSplit: canSplitVectorPathAtPoints(state.paths, state.selectedPoints),
      canJoin: this.canJoinState(state),
    };
  }
  private canJoinState(state: VectorEditSession): boolean {
    if (!this.base) {
      return false;
    }
    const matrix = this.screenMatrix();
    return canJoinVectorPathEndpoints(state.paths, state.selectedPoints, 32, (point) => applyToPoint(matrix, point));
  }
  private reportRefusal(result: StructuralCommitResult): void {
    const { status } = result;
    if (status === 'committed') {
      return;
    }
    this.options.reportRefusal?.(
      status === 'busy' || status === 'gesture-active' || status === 'not-ready' || status === 'over-budget'
        ? status
        : 'not-ready'
    );
  }
  refreshAvailability = (): void => {
    const state = this.options.session.get();
    if (!state || !this.base) {
      return;
    }
    const next = this.availability(state);
    if (next.canSplit !== state.canSplit || next.canJoin !== state.canJoin) {
      this.publish(next);
    }
  };
  private publish(patch: Partial<VectorEditSession>): void {
    const state = this.options.session.get();
    if (!state) {
      return;
    }
    const next = { ...state, ...patch };
    this.options.session.set({ ...next, ...this.availability(next) });
    this.options.invalidate();
  }

  private record(keepTransform = false): void {
    if (!keepTransform && this.options.session.get()?.pathTransform) {
      this.transformOrigin = null;
      this.publish({ pathTransform: null });
    }
    const state = this.options.session.get();
    const base = this.base;
    if (!state || !base || samePaths(base.paths, state.paths)) {
      return;
    }
    this.base = { ...base, paths: state.paths };
    const result = this.options.commit(
      'Edit vector paths',
      {
        type: 'updateCanvasLayerConfig',
        id: base.id,
        config: { layerType: 'vector', paths: state.paths },
      },
      { type: 'updateCanvasLayerConfig', id: base.id, config: { layerType: 'vector', paths: base.paths } }
    );
    if (result.status !== 'committed') {
      this.reportRefusal(result);
      this.base = base;
      this.transformOrigin = null;
      this.transformCursor = null;
      this.publish({ paths: base.paths, selectedPoints: [], selectionRect: null, pathTransform: null });
    }
  }

  private ready(): boolean {
    return this.hasSession() && this.options.canEdit();
  }
  private actionReady(): boolean {
    return this.ready() && !this.options.isGestureActive();
  }
  private matrix(): Mat2d {
    return lookupDocumentLeaf(this.options.getDocument()!, this.base!.id)!.worldTransform;
  }
  private screenMatrix(): Mat2d {
    return multiply(this.options.viewport.viewMatrix(1), this.matrix());
  }

  begin = (layerId: string): boolean => {
    if (this.options.session.get()?.layerId === layerId) {
      return true;
    }
    this.discard();
    if (this.options.isGestureActive()) {
      return false;
    }
    const doc = this.options.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, layerId) : null;
    if (
      !leaf ||
      leaf.layer.type !== 'vector' ||
      leaf.documentHidden ||
      leafEditRefusal(leaf) ||
      !invert(leaf.worldTransform)
    ) {
      return false;
    }
    this.base = leaf.layer;
    const state: VectorEditSession = {
      layerId,
      paths: leaf.layer.paths,
      activePathId: null,
      selectedPoints: [],
      selectionRect: null,
      pathTransform: null,
      canSplit: false,
      canJoin: false,
    };
    this.options.session.set(state);
    this.options.invalidate();
    return true;
  };

  discard = (): void => {
    this.base = null;
    this.hits.clear();
    this.drag = null;
    this.hoverPath = false;
    this.transformOrigin = null;
    this.transformDrag = null;
    this.transformCursor = null;
    this.options.session.set(null);
    this.options.invalidate();
  };

  onActivate = (): void => {
    const id = this.options.getDocument()?.selectedLayerId;
    if (id) {
      this.begin(id);
    }
  };

  onDocumentChanged = (): void => {
    const doc = this.options.getDocument();
    const state = this.options.session.get();
    const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
    if (!leaf || leaf.layer.type !== 'vector' || leaf.documentHidden || leafEditRefusal(leaf)) {
      if (state) {
        this.discard();
      }
      return;
    }
    if (!state) {
      if (this.options.getToolId() === 'spline') {
        this.begin(leaf.id);
      }
      return;
    }
    if (state.layerId !== leaf.id) {
      this.begin(leaf.id);
      return;
    }
    if (leaf.layer.paths !== this.base?.paths || leaf.layer.transform !== this.base?.transform) {
      this.drag = null;
      this.transformDrag = null;
      this.transformOrigin = null;
      this.transformCursor = null;
      this.base = leaf.layer;
      const paths = leaf.layer.paths;
      this.publish({
        paths,
        activePathId: paths.some((path) => path.id === state.activePathId) ? state.activePathId : null,
        selectedPoints: state.selectedPoints.filter(
          (ref) => !!paths.find((path) => path.id === ref.pathId)?.points[ref.pointIndex]
        ),
        selectionRect: null,
        pathTransform: null,
      });
    }
  };

  private mapPoints(
    paths: CanvasBezierPathContract[],
    refs: VectorPointRef[],
    edit: (point: CanvasBezierPointContract, ref: VectorPointRef) => void
  ): CanvasBezierPathContract[] {
    return paths.map((path) => {
      const indices = refs.filter((ref) => ref.pathId === path.id);
      if (indices.length === 0) {
        return path;
      }
      const points = [...path.points];
      for (const ref of indices) {
        const point = points[ref.pointIndex];
        if (!point) {
          continue;
        }
        const copy = structuredClone(point);
        edit(copy, ref);
        points[ref.pointIndex] = copy;
      }
      return { ...path, points };
    });
  }
  /** Arrow deltas are document pixels, independent of the layer's scale and rotation. */
  nudge = (dx: number, dy: number): void => {
    if (!this.actionReady() || !Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) {
      return;
    }
    const state = this.options.session.get()!;
    if (!state.pathTransform && state.selectedPoints.length === 0 && state.activePathId === null) {
      return;
    }
    const inverse = invert(this.matrix());
    if (!inverse) {
      return;
    }
    const x = inverse.a * dx + inverse.c * dy;
    const y = inverse.b * dx + inverse.d * dy;
    if (state.pathTransform) {
      const frame = state.pathTransform;
      const origin = this.transformOrigin;
      if (!origin || origin.id !== frame.pathId) {
        return;
      }
      const transform = { ...frame.transform, x: frame.transform.x + x, y: frame.transform.y + y };
      const matrix = layerTransformMatrix(transform);
      const path = transformBezierPath(origin, matrix);
      this.publish({
        paths: state.paths.map((candidate) => (candidate.id === frame.pathId ? path : candidate)),
        pathTransform: { ...frame, transform },
      });
      this.record(true);
      return;
    }
    const refs =
      state.selectedPoints.length > 0
        ? state.selectedPoints
        : (state.paths.find((path) => path.id === state.activePathId)?.points ?? []).map((_, pointIndex) => ({
            pathId: state.activePathId!,
            pointIndex,
          }));
    this.publish({
      paths: this.mapPoints(state.paths, refs, (point) => {
        point.anchor = { x: point.anchor.x + x, y: point.anchor.y + y };
        if (point.inHandle) {
          point.inHandle = { x: point.inHandle.x + x, y: point.inHandle.y + y };
        }
        if (point.outHandle) {
          point.outHandle = { x: point.outHandle.x + x, y: point.outHandle.y + y };
        }
      }),
    });
    this.record();
  };
  setPointType = (type: CanvasBezierPointContract['type']): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    this.publish({
      paths: this.mapPoints(state.paths, state.selectedPoints, (point) => setBezierPointType(point, type)),
    });
    this.record();
  };
  smooth = (selectedOnly: boolean): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    const paths = state.paths.map((path) => {
      const refs = state.selectedPoints.filter((ref) => ref.pathId === path.id);
      if (selectedOnly ? refs.length === 0 : path.id !== state.activePathId) {
        return path;
      }
      const smoothed = smoothBezierPathPoints(path.points, path.isClosed);
      return {
        ...path,
        points: selectedOnly
          ? path.points.map((point, index) => (refs.some((ref) => ref.pointIndex === index) ? smoothed[index]! : point))
          : smoothed,
      };
    });
    this.publish({ paths });
    this.record();
  };
  deleteSelection = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    const paths =
      state.selectedPoints.length > 0
        ? deleteVectorPathPoints(state.paths, state.selectedPoints).paths
        : state.paths.filter((path) => path.id !== state.activePathId);
    this.publish({
      paths,
      selectedPoints: [],
      activePathId: paths.some((path) => path.id === state.activePathId) ? state.activePathId : null,
    });
    this.record();
  };
  extractPath = (): void => {
    if (!this.actionReady() || !this.base) {
      return;
    }
    const state = this.options.session.get()!;
    const base = this.base;
    const path = state.paths.find((candidate) => candidate.id === state.activePathId);
    const doc = this.options.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, base.id) : null;
    if (
      !path ||
      !doc ||
      !leaf ||
      leaf.layer.type !== 'vector' ||
      leaf.layer.paths !== base.paths ||
      leaf.layer.transform !== base.transform ||
      leaf.documentHidden ||
      leafEditRefusal(leaf)
    ) {
      return;
    }
    const layer = createVectorLayer({
      id: this.options.createId(),
      name: nextVectorLayerName(getDocumentIndex(doc).nodes.map((entry) => entry.node.name)),
    });
    layer.paths = [{ ...structuredClone(path), id: this.options.createId() }];
    layer.transform = { ...base.transform };
    layer.opacity = leaf.layer.opacity;
    layer.blendMode = leaf.layer.blendMode;
    layer.guideColor = leaf.layer.guideColor;
    layer.fillColor = leaf.layer.fillColor;
    layer.strokeColor = leaf.layer.strokeColor;
    layer.strokeWidth = leaf.layer.strokeWidth;
    layer.fillEnabled = leaf.layer.fillEnabled;
    layer.strokeEnabled = leaf.layer.strokeEnabled;
    layer.traceTaperEnds = leaf.layer.traceTaperEnds;
    layer.traceTaper = leaf.layer.traceTaper;
    // Export the active draft version without committing or changing the source draft.
    const mutations = addedVectorLayerMutations(
      layer,
      this.options.captureInsertionAnchor(base.id),
      doc.selectedLayerId,
      false
    );
    this.reportRefusal(this.options.commit('Extract vector path', mutations.forward, mutations.inverse));
  };
  copyPath = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    const index = state.paths.findIndex((path) => path.id === state.activePathId);
    if (index < 0) {
      return;
    }
    const copy = { ...structuredClone(state.paths[index]!), id: this.options.createId() };
    const paths = [...state.paths];
    paths.splice(index + 1, 0, copy);
    this.publish({
      paths,
      activePathId: copy.id,
      selectedPoints: copy.points.map((_, pointIndex) => ({ pathId: copy.id, pointIndex })),
    });
    this.record();
  };
  canSplit = (): boolean => {
    const state = this.options.session.get();
    return !!state && canSplitVectorPathAtPoints(state.paths, state.selectedPoints);
  };
  canJoin = (): boolean => {
    const state = this.options.session.get();
    return !!state && this.canJoinState(state);
  };
  split = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    if (!canSplitVectorPathAtPoints(state.paths, state.selectedPoints)) {
      return;
    }
    let selected: VectorPointRef | null = null;
    const paths = state.paths.flatMap((path) => {
      const result = splitVectorPathAtPoints(
        path,
        state.selectedPoints.filter((ref) => ref.pathId === path.id).map((ref) => ref.pointIndex),
        this.options.createId
      );
      if (!result) {
        return [path];
      }
      selected = { pathId: result.activePathId, pointIndex: result.activePointIndex };
      return result.paths;
    });
    this.publish({
      paths,
      activePathId: selected ? (selected as VectorPointRef).pathId : state.activePathId,
      selectedPoints: selected ? [selected] : [],
    });
    this.record();
  };
  join = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    if (!this.canJoinState(state)) {
      return;
    }
    const [from, to] = state.selectedPoints as [VectorPointRef, VectorPointRef];
    const a = state.paths.find((path) => path.id === from.pathId)!;
    const b = state.paths.find((path) => path.id === to.pathId)!;
    const matrix = this.screenMatrix();
    const weld =
      distance(
        applyToPoint(matrix, a.points[from.pointIndex]!.anchor),
        applyToPoint(matrix, b.points[to.pointIndex]!.anchor)
      ) <= 32;
    const result = joinVectorPathEndpoints(a, from.pointIndex, b, to.pointIndex, weld);
    if (!result) {
      return;
    }
    const paths = state.paths.flatMap((path) => (path.id === a.id ? [result.path] : path.id === b.id ? [] : [path]));
    this.publish({
      paths,
      activePathId: result.path.id,
      selectedPoints: [{ pathId: result.path.id, pointIndex: result.activePointIndex }],
    });
    this.record();
  };

  transformPath = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    const path = state.paths.find((candidate) => candidate.id === state.activePathId);
    if (!path || path.points.length < 2) {
      return;
    }
    const bounds = getBezierPathBounds(path.points, path.isClosed);
    this.transformOrigin = path;
    this.transformCursor = null;
    this.hoverPath = false;
    this.publish({
      selectedPoints: [],
      pathTransform: {
        pathId: path.id,
        rect: { ...bounds, width: Math.max(1, bounds.width), height: Math.max(1, bounds.height) },
        transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
      },
    });
  };
  private transformTarget(input: PointerInput): TransformTarget | null {
    const frame = this.options.session.get()?.pathTransform;
    if (!frame) {
      return null;
    }
    const matrix = this.matrix();
    return transformTargetAt({
      point: input.screenPoint,
      rect: frame.rect,
      transform: frame.transform,
      toScreen: (point) => this.options.viewport.documentToScreen(applyToPoint(matrix, point)),
    });
  }
  private cursorForTransform(target: TransformTarget | null): string | null {
    const frame = this.options.session.get()?.pathTransform;
    if (!target || !frame) {
      return null;
    }
    return target.kind === 'scale'
      ? resizeCursorForHandle(
          { ...frame.transform, rotation: frame.transform.rotation + (this.base?.transform.rotation ?? 0) },
          target.handle
        )
      : target.kind === 'rotate'
        ? 'grab'
        : 'move';
  }
  cursor = (): string => this.transformCursor ?? (this.drag?.kind === 'path' || this.hoverPath ? 'move' : 'crosshair');
  private updateHover(ctx: ToolContext, input: PointerInput): void {
    const state = this.options.session.get();
    let movable = false;
    if (state && this.ready() && !input.modifiers.shift) {
      const matrix = this.screenMatrix();
      const scale = this.options.viewport.getZoom() * getScale(this.matrix());
      const overAnchor = state.paths.some((path) =>
        this.hits.hitAnchor(path, matrix, input.screenPoint, scale, ANCHOR_HIT_RADIUS_PX)
      );
      const overHandle = state.selectedPoints.some((ref) => {
        const path = state.paths.find((path) => path.id === ref.pathId);
        if (!path?.points[ref.pointIndex]) {
          return false;
        }
        const handles = vectorHandleLayout(path, ref.pointIndex, matrix);
        return [handles.inHandle, handles.outHandle].some(
          (point) => !!point && distance(point, input.screenPoint) <= HANDLE_HIT_RADIUS_PX
        );
      });
      movable =
        !overAnchor &&
        !overHandle &&
        state.paths.some((path) => {
          const hit = this.nearSegment(path, input);
          return !!hit && hit.distance <= 10;
        });
    }
    if (movable !== this.hoverPath) {
      this.hoverPath = movable;
      ctx.updateCursor();
    }
  }
  private nearSegment(path: CanvasBezierPathContract, input: Pick<PointerInput, 'screenPoint'>) {
    const matrix = this.screenMatrix();
    return this.hits.find(path, matrix, input.screenPoint, this.options.viewport.getZoom() * getScale(this.matrix()));
  }
  targetContextMenu = (screenPoint: Vec2): boolean => {
    if (!this.actionReady()) {
      return false;
    }
    const state = this.options.session.get()!;
    const matrix = this.screenMatrix();
    let anchorHit: VectorPointRef | null = null;
    let pathHit: string | null = null;
    let anchorDistance = ANCHOR_HIT_RADIUS_PX;
    let pathDistance = 10;
    const paths = [
      ...state.paths.filter((path) => path.id === state.activePathId),
      ...state.paths.filter((path) => path.id !== state.activePathId),
    ];
    for (const path of paths) {
      path.points.forEach((point, pointIndex) => {
        const d = distance(applyToPoint(matrix, point.anchor), screenPoint);
        if (d <= anchorDistance && (!anchorHit || d < anchorDistance)) {
          anchorHit = { pathId: path.id, pointIndex };
          anchorDistance = d;
        }
      });
      const hit = this.nearSegment(path, { screenPoint });
      if (hit && hit.distance <= pathDistance && (!pathHit || hit.distance < pathDistance)) {
        pathHit = path.id;
        pathDistance = hit.distance;
      }
    }
    const ref = anchorHit as VectorPointRef | null;
    const pathId = ref?.pathId ?? pathHit;
    if (!pathId) {
      return false;
    }
    if (state.pathTransform && state.pathTransform.pathId !== pathId) {
      this.transformOrigin = null;
      this.transformCursor = null;
      this.publish({ pathTransform: null });
    }
    const selectedPoints = ref
      ? state.selectedPoints.some((selected) => sameRef(selected, ref))
        ? state.selectedPoints
        : [ref]
      : pathId === state.activePathId
        ? state.selectedPoints
        : [];
    this.publish({ activePathId: pathId, selectedPoints });
    return true;
  };
  deletePath = (): void => {
    if (!this.actionReady()) {
      return;
    }
    const state = this.options.session.get()!;
    const paths = state.paths.filter((path) => path.id !== state.activePathId);
    this.publish({
      paths,
      activePathId: paths[0]?.id ?? null,
      selectedPoints: state.selectedPoints.filter((ref) => ref.pathId !== state.activePathId),
    });
    this.record();
  };
  onPointerDown = (_ctx: ToolContext, input: PointerInput): void => {
    if (!this.ready() || (input.buttons & 1) === 0) {
      return;
    }
    const state = this.options.session.get()!;
    const inverse = invert(this.matrix())!;
    const local = applyToPoint(inverse, input.documentPoint);
    const matrix = this.screenMatrix();
    if (state.pathTransform) {
      const target = this.transformTarget(input);
      if (target && this.transformOrigin) {
        this.transformDrag = {
          target,
          start: local,
          transform: state.pathTransform.transform,
          before: snapshot(state),
        };
        this.transformCursor = this.cursorForTransform(target);
        return;
      }
      this.transformOrigin = null;
      this.transformCursor = null;
      this.publish({ pathTransform: null });
    }
    const begin = (kind: Drag['kind'], refs: VectorPointRef[], ref?: VectorPointRef, missing?: Drag['missing']) => {
      this.hoverPath = kind === 'path';
      const activePathId = ref?.pathId ?? state.activePathId;
      this.publish({ selectedPoints: refs, activePathId });
      this.drag = {
        kind,
        before: snapshot(this.options.session.get()!),
        start: local,
        screen: input.screenPoint,
        ref,
        missing,
      };
    };
    for (const ref of [...state.selectedPoints].reverse()) {
      const path = state.paths.find((path) => path.id === ref.pathId);
      const point = path?.points[ref.pointIndex];
      if (!path || !point) {
        continue;
      }
      const handles = vectorHandleLayout(path, ref.pointIndex, matrix);
      for (const handle of ['inHandle', 'outHandle'] as const) {
        const marker = handles[handle];
        if (marker && distance(marker, input.screenPoint) <= HANDLE_HIT_RADIUS_PX) {
          begin(handle, [...state.selectedPoints.filter((other) => !sameRef(ref, other)), ref], ref);
          const origin = point[handle] ?? applyToPoint(invert(matrix)!, marker);
          this.drag!.handleOrigin = origin;
          if (!point[handle]) {
            this.publish({
              paths: this.mapPoints(state.paths, [ref], (copy) => setBezierPointHandle(copy, handle, origin)),
            });
          }
          return;
        }
      }
    }
    const ordered = [
      ...state.paths.filter((path) => path.id === state.activePathId),
      ...state.paths.filter((path) => path.id !== state.activePathId),
    ];
    for (const path of ordered) {
      for (let pointIndex = 0; pointIndex < path.points.length; pointIndex += 1) {
        const point = path.points[pointIndex]!;
        if (distance(applyToPoint(matrix, point.anchor), input.screenPoint) > ANCHOR_HIT_RADIUS_PX) {
          continue;
        }
        const ref = { pathId: path.id, pointIndex };
        const selected = state.selectedPoints.some((other) => sameRef(ref, other));
        const refs =
          input.modifiers.ctrl || input.modifiers.meta || selected
            ? [...state.selectedPoints.filter((other) => !sameRef(ref, other)), ref]
            : [ref];
        const last = state.selectedPoints.at(-1);
        const repeated = last && sameRef(last, ref) && refs.length === 1;
        const missing =
          point.type === 'corner' && Boolean(point.inHandle) !== Boolean(point.outHandle)
            ? point.inHandle
              ? 'outHandle'
              : 'inHandle'
            : undefined;
        begin(
          repeated && !point.inHandle && !point.outHandle ? 'pull' : repeated && missing ? 'maybe-pull' : 'anchor',
          refs,
          ref,
          missing
        );
        return;
      }
    }
    const active = state.paths.find((path) => path.id === state.activePathId);
    if (input.modifiers.shift && active) {
      const hit = this.nearSegment(active, input);
      if (hit && hit.distance <= 10) {
        const points = active.points.map((point) => structuredClone(point));
        const from = points[hit.segmentIndex]!;
        const to = points[(hit.segmentIndex + 1) % points.length]!;
        const split = splitBezierSegmentAt(from, to, hit.t);
        from.outHandle = split.fromOutHandle;
        to.inHandle = split.toInHandle;
        const ref = { pathId: active.id, pointIndex: hit.segmentIndex + 1 };
        points.splice(ref.pointIndex, 0, split.insertPoint);
        this.publish({
          paths: state.paths.map((path) => (path.id === active.id ? { ...path, points } : path)),
          selectedPoints: [ref],
        });
        this.drag = {
          kind: 'anchor',
          before: snapshot(this.options.session.get()!),
          cancelBefore: snapshot(state),
          start: local,
          screen: input.screenPoint,
          ref,
        };
        return;
      }
    }
    for (const path of ordered) {
      const hit = this.nearSegment(path, input);
      if (hit && hit.distance <= 10) {
        begin('path', [], { pathId: path.id, pointIndex: 0 });
        return;
      }
    }
    const additive = input.modifiers.ctrl || input.modifiers.meta;
    this.hoverPath = false;
    this.drag = {
      kind: 'rect',
      before: snapshot(state),
      start: input.documentPoint,
      screen: input.screenPoint,
      additive,
    };
    this.publish({
      activePathId: additive ? state.activePathId : null,
      selectedPoints: additive ? state.selectedPoints : [],
      selectionRect: { ...input.documentPoint, width: 0, height: 0 },
    });
  };

  onPointerMove = (ctx: ToolContext, input: PointerInput): void => {
    const frame = this.options.session.get()?.pathTransform;
    if (frame && this.ready()) {
      const gesture = this.transformDrag;
      if (!gesture || !this.transformOrigin) {
        const cursor = this.cursorForTransform(this.transformTarget(input));
        if (cursor !== this.transformCursor) {
          this.transformCursor = cursor;
          ctx.updateCursor();
        }
        return;
      }
      const pointer = applyToPoint(invert(this.matrix())!, input.documentPoint);
      const args = {
        rect: frame.rect,
        start: gesture.transform,
        startPointerDoc: gesture.start,
        pointerDoc: pointer,
        shift: input.modifiers.shift,
      };
      const transform =
        gesture.target.kind === 'move'
          ? applyMove(
              gesture.transform,
              { x: pointer.x - gesture.start.x, y: pointer.y - gesture.start.y },
              input.modifiers.shift,
              0
            )
          : gesture.target.kind === 'scale'
            ? applyScale({
                ...args,
                alt: input.modifiers.alt,
                constrain: input.modifiers.shift && isCornerHandle(gesture.target.handle),
                grid: 0,
                handle: gesture.target.handle,
              })
            : applyRotate(args);
      const matrix = layerTransformMatrix(transform);
      const path = transformBezierPath(this.transformOrigin, matrix);
      this.publish({
        paths: gesture.before.paths.map((candidate) => (candidate.id === path.id ? path : candidate)),
        pathTransform: { ...frame, transform },
      });
      return;
    }
    const drag = this.drag;
    if (!drag) {
      this.updateHover(ctx, input);
      return;
    }
    if (!this.ready()) {
      return;
    }
    if (drag.kind === 'rect') {
      const x = Math.min(drag.screen.x, input.screenPoint.x),
        y = Math.min(drag.screen.y, input.screenPoint.y);
      const width = Math.abs(input.screenPoint.x - drag.screen.x),
        height = Math.abs(input.screenPoint.y - drag.screen.y);
      const matrix = this.screenMatrix();
      const refs = drag.before.paths.flatMap((path) =>
        path.points.flatMap((point, pointIndex) => {
          const p = applyToPoint(matrix, point.anchor);
          return p.x >= x && p.x <= x + width && p.y >= y && p.y <= y + height ? [{ pathId: path.id, pointIndex }] : [];
        })
      );
      const selectedPoints = drag.additive
        ? [
            ...drag.before.selectedPoints,
            ...refs.filter((ref) => !drag.before.selectedPoints.some((old) => sameRef(old, ref))),
          ]
        : refs;
      this.publish({
        selectedPoints,
        activePathId: selectedPoints.at(-1)?.pathId ?? (drag.additive ? drag.before.activePathId : null),
        selectionRect: {
          x: Math.min(drag.start.x, input.documentPoint.x),
          y: Math.min(drag.start.y, input.documentPoint.y),
          width: Math.abs(input.documentPoint.x - drag.start.x),
          height: Math.abs(input.documentPoint.y - drag.start.y),
        },
      });
      return;
    }
    const ref = drag.ref!;
    const path = drag.before.paths.find((candidate) => candidate.id === ref.pathId)!;
    const point = path.points[ref.pointIndex]!;
    const local = applyToPoint(invert(this.matrix())!, input.documentPoint);
    if (distance(drag.screen, input.screenPoint) < 3 && !drag.handleOrigin) {
      this.publish({ paths: drag.before.paths });
      return;
    }
    if (drag.kind === 'maybe-pull') {
      const handle = point[drag.missing === 'inHandle' ? 'outHandle' : 'inHandle']!;
      const dot =
        (local.x - point.anchor.x) * (handle.x - point.anchor.x) +
        (local.y - point.anchor.y) * (handle.y - point.anchor.y);
      drag.kind = dot < 0 ? drag.missing! : 'anchor';
    }
    if (drag.kind === 'anchor' || drag.kind === 'path') {
      const dx = local.x - drag.start.x,
        dy = local.y - drag.start.y;
      const refs =
        drag.kind === 'path'
          ? path.points.map((_, pointIndex) => ({ pathId: path.id, pointIndex }))
          : drag.before.selectedPoints;
      this.publish({
        paths: this.mapPoints(drag.before.paths, refs, (copy) => {
          copy.anchor = { x: copy.anchor.x + dx, y: copy.anchor.y + dy };
          if (copy.inHandle) {
            copy.inHandle = { x: copy.inHandle.x + dx, y: copy.inHandle.y + dy };
          }
          if (copy.outHandle) {
            copy.outHandle = { x: copy.outHandle.x + dx, y: copy.outHandle.y + dy };
          }
        }),
      });
    } else {
      const handle =
        drag.kind === 'pull'
          ? getBezierPointPullHandleType(path.points, path.isClosed, ref.pointIndex, local)
          : drag.kind === 'inHandle'
            ? 'inHandle'
            : 'outHandle';
      const target = drag.handleOrigin
        ? { x: drag.handleOrigin.x + local.x - drag.start.x, y: drag.handleOrigin.y + local.y - drag.start.y }
        : local;
      this.publish({
        paths: this.mapPoints(drag.before.paths, [ref], (copy) => setBezierPointHandle(copy, handle, target)),
      });
    }
  };
  onPointerUp = (): void => {
    if (this.transformDrag) {
      this.transformDrag = null;
      this.record(true);
      return;
    }
    if (!this.drag) {
      return;
    }
    this.drag = null;
    this.publish({ selectionRect: null });
    this.record();
  };
  onPointerCancel = (): void => {
    if (this.transformDrag) {
      const gesture = this.transformDrag;
      this.transformDrag = null;
      const frame = this.options.session.get()?.pathTransform;
      this.publish({
        ...gesture.before,
        ...(frame ? { pathTransform: { ...frame, transform: gesture.transform } } : {}),
      });
    }
    if (this.drag) {
      const before = this.drag.cancelBefore ?? this.drag.before;
      this.drag = null;
      this.publish({ ...before, selectionRect: null });
    }
  };
  onDeactivate = (_ctx: ToolContext, options?: { temporary?: boolean }): void => {
    this.onPointerCancel();
    if (!options?.temporary) {
      this.discard();
    }
  };
  onKeyCommand = (_ctx: ToolContext, command: 'apply' | 'cancel'): boolean => {
    if (!this.hasSession() || command !== 'cancel') {
      return false;
    }
    this.onPointerCancel();
    this.transformOrigin = null;
    this.transformCursor = null;
    this.publish({ activePathId: null, selectedPoints: [], selectionRect: null, pathTransform: null });
    return true;
  };
}
