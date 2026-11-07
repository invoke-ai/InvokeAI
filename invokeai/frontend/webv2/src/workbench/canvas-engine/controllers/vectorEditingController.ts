import type { StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasBezierPathContract, CanvasBezierPointContract, CanvasVectorLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { Mat2d, PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { VectorEditDestination, VectorEditSession, VectorPointRef } from '@workbench/canvas-engine/vectorEditState';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { leafEditRefusal } from '@workbench/canvas-engine/document/layerEligibility';
import { findNearestBezierPathSegment, getBezierPathHitSamplesPerSegment, getBezierPointPullHandleType, setBezierPointHandle, setBezierPointType, smoothBezierPathPoints, splitBezierSegmentAt } from '@workbench/canvas-engine/math/bezierPath';
import { applyToPoint, getScale, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { vectorHandleLayout } from '@workbench/canvas-engine/math/vectorHandleLayout';
import { canJoinVectorPathEndpoints, canSplitVectorPathAtPoints, deleteVectorPathPoints, joinVectorPathEndpoints, splitVectorPathAtPoints } from '@workbench/canvas-engine/math/vectorPathTopology';


interface Options {
  session: { get(): VectorEditSession | null; set(value: VectorEditSession | null): void };
  getDocument(): CanvasDocumentContractV3 | null;
  canEdit(): boolean;
  isGestureActive(): boolean;
  getToolId(): ToolId;
  setTool(id: ToolId): void;
  dispatch(mutation: CanvasProjectMutation): void;
  commit(label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): StructuralCommitResult;
  invalidate(): void;
  createId(): string;
  viewport: Viewport;
}

type Snapshot = Pick<VectorEditSession, 'paths' | 'activePathId' | 'selectedPoints'>;
type Drag = {
  before: Snapshot;
  start: Vec2;
  screen: Vec2;
  kind: 'anchor' | 'pull' | 'maybe-pull' | 'inHandle' | 'outHandle' | 'rect';
  ref?: VectorPointRef;
  additive?: boolean;
  missing?: 'inHandle' | 'outHandle';
  handleOrigin?: Vec2;
};
const sameRef = (a: VectorPointRef, b: VectorPointRef): boolean => a.pathId === b.pathId && a.pointIndex === b.pointIndex;
const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
// Hit radii are CSS viewport pixels: project geometry through screenMatrix() before comparing.
const ANCHOR_HIT_RADIUS_PX = 12;
const HANDLE_HIT_RADIUS_PX = 12;
const snapshot = (state: VectorEditSession): Snapshot => ({ paths: state.paths, activePathId: state.activePathId, selectedPoints: state.selectedPoints });
const samePaths = (a: CanvasBezierPathContract[], b: CanvasBezierPathContract[]): boolean => a === b || JSON.stringify(a) === JSON.stringify(b);

/** Draft geometry, point interaction and bounded local history belong to one engine's Edit session. */
export class VectorEditingController implements Tool {
  readonly id = 'move';
  private base: CanvasVectorLayerContract | null = null;
  private previousTool: ToolId = 'move';
  private history: Snapshot[] = [];
  private historyIndex = 0;
  private drag: Drag | null = null;

  constructor(private readonly options: Options) {}

  hasSession = (): boolean => this.options.session.get() !== null;
  private publish(patch: Partial<VectorEditSession>): void {
    const state = this.options.session.get();
    if (!state) { return; }
    this.options.session.set({ ...state, ...patch, canUndo: this.historyIndex > 0, canRedo: this.historyIndex < this.history.length - 1 });
    this.options.invalidate();
  }

  private record(): void {
    const state = this.options.session.get();
    if (!state || samePaths(this.history[this.historyIndex]!.paths, state.paths)) { return; }
    this.history = [...this.history.slice(0, this.historyIndex + 1), snapshot(state)].slice(-64);
    this.historyIndex = this.history.length - 1;
    this.publish({});
  }

  private ready(): boolean {
    return this.hasSession() && !this.options.session.get()!.pendingExit && this.options.canEdit();
  }
  private actionReady(): boolean { return this.ready() && !this.options.isGestureActive(); }
  private matrix(): Mat2d { return lookupDocumentLeaf(this.options.getDocument()!, this.base!.id)!.worldTransform; }
  private screenMatrix(): Mat2d { return multiply(this.options.viewport.viewMatrix(1), this.matrix()); }

  begin = (layerId: string): boolean => {
    if (this.hasSession() || !this.options.canEdit() || this.options.isGestureActive()) { return false; }
    const doc = this.options.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, layerId) : null;
    if (!leaf || leaf.layer.type !== 'vector' || leaf.documentHidden || leafEditRefusal(leaf) || leaf.layer.paths.length === 0 || !invert(leaf.worldTransform)) { return false; }
    this.base = leaf.layer;
    this.previousTool = this.options.getToolId();
    const state: VectorEditSession = {
      layerId, paths: leaf.layer.paths, activePathId: leaf.layer.paths[0]!.id, selectedPoints: [],
      selectionRect: null, canUndo: false, canRedo: false, pendingExit: null,
    };
    this.history = [snapshot(state)];
    this.historyIndex = 0;
    this.options.session.set(state);
    this.options.setTool('move');
    this.options.invalidate();
    return true;
  };

  private close(restoreTool: boolean): void {
    this.base = null;
    this.drag = null;
    this.history = [];
    this.options.session.set(null);
    this.options.invalidate();
    if (restoreTool) { this.options.setTool(this.previousTool); }
  }
  cancel = (): void => this.close(true);
  discard = (): void => this.close(false);
  apply = (): StructuralCommitResult | null => {
    const state = this.options.session.get();
    const base = this.base;
    if (!state || !base) { return null; }
    if (!this.options.canEdit() || this.options.isGestureActive()) { return { status: 'busy' }; }
    const doc = this.options.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, base.id) : null;
    if (!leaf || leaf.layer.type !== 'vector' || leaf.layer.paths !== base.paths || leaf.layer.transform !== base.transform || leaf.documentHidden || leafEditRefusal(leaf)) { return { status: 'stale' }; }
    if (samePaths(base.paths, state.paths)) { this.close(true); return null; }
    // Keep the draft recoverable on refusal; the mirror must not treat our own successful write as an external edit.
    this.base = { ...base, paths: state.paths };
    const result = this.options.commit('Edit vector paths', {
      type: 'updateCanvasLayerConfig', id: base.id, config: { layerType: 'vector', paths: state.paths },
    }, { type: 'updateCanvasLayerConfig', id: base.id, config: { layerType: 'vector', paths: base.paths } });
    if (result.status === 'committed') { this.close(true); } else { this.base = base; }
    return result;
  };

  reset = (): void => {
    if (!this.actionReady() || !this.base) { return; }
    this.publish({ paths: this.base.paths, activePathId: this.base.paths[0]?.id ?? null, selectedPoints: [], selectionRect: null });
    this.record();
  };
  undo = (): boolean => this.replay(-1);
  redo = (): boolean => this.replay(1);
  private replay(offset: number): boolean {
    const next = this.historyIndex + offset;
    if (!this.actionReady() || next < 0 || next >= this.history.length) { return false; }
    this.historyIndex = next;
    this.publish({ ...this.history[next]!, selectionRect: null });
    return true;
  }

  requestExit = (destination: VectorEditDestination): void => {
    if (this.hasSession() && !this.options.session.get()!.pendingExit) { this.publish({ pendingExit: destination }); }
  };
  resolveExit = (decision: 'apply' | 'discard' | 'cancel'): StructuralCommitResult | null => {
    const state = this.options.session.get();
    const destination = state?.pendingExit;
    if (!state || !destination) { return null; }
    if (decision === 'cancel') {
      if (destination.kind === 'layer') { this.options.dispatch({ type: 'setCanvasSelectedLayer', id: state.layerId }); }
      this.publish({ pendingExit: null });
      return null;
    }
    if (decision === 'apply') {
      const result = this.apply();
      if (result && result.status !== 'committed') { return result; }
    } else { this.close(true); }
    if (destination.kind === 'tool') { this.options.setTool(destination.toolId); }
    return null;
  };
  onDocumentChanged = (): void => {
    const state = this.options.session.get();
    if (!state || !this.base) { return; }
    const doc = this.options.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, state.layerId) : null;
    if (!leaf || leaf.layer.type !== 'vector' || leaf.layer.paths !== this.base.paths || leaf.layer.transform !== this.base.transform || leaf.documentHidden || leafEditRefusal(leaf)) {
      this.close(false);
    } else if (doc!.selectedLayerId !== state.layerId) {
      this.requestExit({ kind: 'layer', layerId: doc!.selectedLayerId });
    }
  };

  append = (layerId: string, path: CanvasBezierPathContract): boolean => {
    const state = this.options.session.get();
    if (!state || state.layerId !== layerId) { return false; }
    if (this.ready()) {
      this.publish({ paths: [...state.paths, path], activePathId: path.id, selectedPoints: [] });
      this.record();
    }
    return true;
  };

  private mapPoints(paths: CanvasBezierPathContract[], refs: VectorPointRef[], edit: (point: CanvasBezierPointContract, ref: VectorPointRef) => void): CanvasBezierPathContract[] {
    return paths.map((path) => {
      const indices = refs.filter((ref) => ref.pathId === path.id);
      if (indices.length === 0) { return path; }
      const points = [...path.points];
      for (const ref of indices) {
        const point = points[ref.pointIndex];
        if (!point) { continue; }
        const copy = structuredClone(point);
        edit(copy, ref);
        points[ref.pointIndex] = copy;
      }
      return { ...path, points };
    });
  }
  setPointType = (type: CanvasBezierPointContract['type']): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    this.publish({ paths: this.mapPoints(state.paths, state.selectedPoints, (point) => setBezierPointType(point, type)) });
    this.record();
  };
  smooth = (selectedOnly: boolean): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    const paths = state.paths.map((path) => {
      const refs = state.selectedPoints.filter((ref) => ref.pathId === path.id);
      if (selectedOnly ? refs.length === 0 : path.id !== state.activePathId) { return path; }
      const smoothed = smoothBezierPathPoints(path.points, path.isClosed);
      return { ...path, points: selectedOnly ? path.points.map((point, index) => refs.some((ref) => ref.pointIndex === index) ? smoothed[index]! : point) : smoothed };
    });
    this.publish({ paths }); this.record();
  };
  deleteSelection = (): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    const paths = state.selectedPoints.length > 0
      ? deleteVectorPathPoints(state.paths, state.selectedPoints).paths
      : state.paths.filter((path) => path.id !== state.activePathId);
    this.publish({ paths, selectedPoints: [], activePathId: paths.some((path) => path.id === state.activePathId) ? state.activePathId : paths[0]?.id ?? null });
    this.record();
  };
  copyPath = (): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    const index = state.paths.findIndex((path) => path.id === state.activePathId);
    if (index < 0) { return; }
    const copy = { ...structuredClone(state.paths[index]!), id: this.options.createId() };
    const paths = [...state.paths]; paths.splice(index + 1, 0, copy);
    this.publish({ paths, activePathId: copy.id, selectedPoints: copy.points.map((_, pointIndex) => ({ pathId: copy.id, pointIndex })) });
    this.record();
  };
  canSplit = (): boolean => {
    const state = this.options.session.get();
    return !!state && canSplitVectorPathAtPoints(state.paths, state.selectedPoints);
  };
  canJoin = (): boolean => {
    const state = this.options.session.get();
    return !!state && !!this.base && canJoinVectorPathEndpoints(state.paths, state.selectedPoints, 32 / (this.options.viewport.getZoom() * getScale(this.matrix())));
  };
  split = (): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    if (!canSplitVectorPathAtPoints(state.paths, state.selectedPoints)) { return; }
    let selected: VectorPointRef | null = null;
    const paths = state.paths.flatMap((path) => {
      const result = splitVectorPathAtPoints(path, state.selectedPoints.filter((ref) => ref.pathId === path.id).map((ref) => ref.pointIndex), this.options.createId);
      if (!result) { return [path]; }
      selected = { pathId: result.activePathId, pointIndex: result.activePointIndex };
      return result.paths;
    });
    this.publish({ paths, activePathId: selected ? (selected as VectorPointRef).pathId : state.activePathId, selectedPoints: selected ? [selected] : [] });
    this.record();
  };
  join = (): void => {
    if (!this.actionReady()) { return; }
    const state = this.options.session.get()!;
    const radius = 32 / (this.options.viewport.getZoom() * getScale(this.matrix()));
    if (!canJoinVectorPathEndpoints(state.paths, state.selectedPoints, radius)) { return; }
    const [from, to] = state.selectedPoints as [VectorPointRef, VectorPointRef];
    const a = state.paths.find((path) => path.id === from.pathId)!;
    const b = state.paths.find((path) => path.id === to.pathId)!;
    const matrix = this.screenMatrix();
    const weld = distance(applyToPoint(matrix, a.points[from.pointIndex]!.anchor), applyToPoint(matrix, b.points[to.pointIndex]!.anchor)) <= 32;
    const result = joinVectorPathEndpoints(a, from.pointIndex, b, to.pointIndex, weld);
    if (!result) { return; }
    const paths = state.paths.flatMap((path) => path.id === a.id ? [result.path] : path.id === b.id ? [] : [path]);
    this.publish({ paths, activePathId: result.path.id, selectedPoints: [{ pathId: result.path.id, pointIndex: result.activePointIndex }] });
    this.record();
  };

  cursor = (): string => this.ready() ? 'crosshair' : 'not-allowed';
  private nearSegment(path: CanvasBezierPathContract, input: PointerInput) {
    const matrix = this.screenMatrix();
    const projected = path.points.map((point) => ({
      anchor: applyToPoint(matrix, point.anchor),
      inHandle: point.inHandle ? applyToPoint(matrix, point.inHandle) : null,
      outHandle: point.outHandle ? applyToPoint(matrix, point.outHandle) : null,
    }));
    return findNearestBezierPathSegment(projected, path.isClosed, input.screenPoint, getBezierPathHitSamplesPerSegment(this.options.viewport.getZoom() * getScale(this.matrix())));
  }
  onPointerDown = (_ctx: ToolContext, input: PointerInput): void => {
    if (!this.ready() || (input.buttons & 1) === 0) { return; }
    const state = this.options.session.get()!;
    const inverse = invert(this.matrix())!;
    const local = applyToPoint(inverse, input.documentPoint);
    const matrix = this.screenMatrix();
    const begin = (kind: Drag['kind'], refs: VectorPointRef[], ref?: VectorPointRef, missing?: Drag['missing']) => {
      const activePathId = ref?.pathId ?? state.activePathId;
      this.publish({ selectedPoints: refs, activePathId });
      this.drag = { kind, before: snapshot(this.options.session.get()!), start: local, screen: input.screenPoint, ref, missing };
    };
    for (const ref of [...state.selectedPoints].reverse()) {
      const path = state.paths.find((path) => path.id === ref.pathId);
      const point = path?.points[ref.pointIndex];
      if (!path || !point) { continue; }
      const handles = vectorHandleLayout(path, ref.pointIndex, matrix);
      for (const handle of ['inHandle', 'outHandle'] as const) {
        const marker = handles[handle];
        if (marker && distance(marker, input.screenPoint) <= HANDLE_HIT_RADIUS_PX) {
          begin(handle, [...state.selectedPoints.filter((other) => !sameRef(ref, other)), ref], ref);
          const origin = point[handle] ?? applyToPoint(invert(matrix)!, marker);
          this.drag!.handleOrigin = origin;
          if (!point[handle]) {
            this.publish({ paths: this.mapPoints(state.paths, [ref], (copy) => setBezierPointHandle(copy, handle, origin)) });
          }
          return;
        }
      }
    }
    const ordered = [...state.paths.filter((path) => path.id === state.activePathId), ...state.paths.filter((path) => path.id !== state.activePathId)];
    for (const path of ordered) {
      for (let pointIndex = 0; pointIndex < path.points.length; pointIndex += 1) {
        const point = path.points[pointIndex]!;
        if (distance(applyToPoint(matrix, point.anchor), input.screenPoint) > ANCHOR_HIT_RADIUS_PX) { continue; }
        const ref = { pathId: path.id, pointIndex };
        const selected = state.selectedPoints.some((other) => sameRef(ref, other));
        const refs = input.modifiers.ctrl || input.modifiers.meta || selected
          ? [...state.selectedPoints.filter((other) => !sameRef(ref, other)), ref] : [ref];
        const last = state.selectedPoints.at(-1);
        const repeated = last && sameRef(last, ref) && refs.length === 1;
        const missing = point.type === 'corner' && Boolean(point.inHandle) !== Boolean(point.outHandle)
          ? point.inHandle ? 'outHandle' : 'inHandle' : undefined;
        begin(repeated && !point.inHandle && !point.outHandle ? 'pull' : repeated && missing ? 'maybe-pull' : 'anchor', refs, ref, missing);
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
        from.outHandle = split.fromOutHandle; to.inHandle = split.toInHandle;
        const ref = { pathId: active.id, pointIndex: hit.segmentIndex + 1 };
        points.splice(ref.pointIndex, 0, split.insertPoint);
        this.publish({ paths: state.paths.map((path) => path.id === active.id ? { ...path, points } : path), selectedPoints: [ref] });
        this.record();
        this.drag = { kind: 'anchor', before: snapshot(this.options.session.get()!), start: local, screen: input.screenPoint, ref };
        return;
      }
    }
    for (const path of ordered) {
      const hit = this.nearSegment(path, input);
      if (hit && hit.distance <= 10) { this.publish({ activePathId: path.id, selectedPoints: [] }); return; }
    }
    const additive = input.modifiers.ctrl || input.modifiers.meta;
    this.drag = { kind: 'rect', before: snapshot(state), start: input.documentPoint, screen: input.screenPoint, additive };
    this.publish({ selectedPoints: additive ? state.selectedPoints : [], selectionRect: { ...input.documentPoint, width: 0, height: 0 } });
  };

  onPointerMove = (_ctx: ToolContext, input: PointerInput): void => {
    const drag = this.drag;
    if (!drag || !this.ready()) { return; }
    if (drag.kind === 'rect') {
      const x = Math.min(drag.screen.x, input.screenPoint.x), y = Math.min(drag.screen.y, input.screenPoint.y);
      const width = Math.abs(input.screenPoint.x - drag.screen.x), height = Math.abs(input.screenPoint.y - drag.screen.y);
      const matrix = this.screenMatrix();
      const refs = drag.before.paths.flatMap((path) => path.points.flatMap((point, pointIndex) => {
        const p = applyToPoint(matrix, point.anchor);
        return p.x >= x && p.x <= x + width && p.y >= y && p.y <= y + height ? [{ pathId: path.id, pointIndex }] : [];
      }));
      const selectedPoints = drag.additive ? [...drag.before.selectedPoints, ...refs.filter((ref) => !drag.before.selectedPoints.some((old) => sameRef(old, ref)))] : refs;
      this.publish({ selectedPoints, activePathId: selectedPoints.at(-1)?.pathId ?? drag.before.activePathId,
        selectionRect: { x: Math.min(drag.start.x, input.documentPoint.x), y: Math.min(drag.start.y, input.documentPoint.y), width: Math.abs(input.documentPoint.x - drag.start.x), height: Math.abs(input.documentPoint.y - drag.start.y) } });
      return;
    }
    const ref = drag.ref!;
    const path = drag.before.paths.find((candidate) => candidate.id === ref.pathId)!;
    const point = path.points[ref.pointIndex]!;
    const local = applyToPoint(invert(this.matrix())!, input.documentPoint);
    if (distance(drag.screen, input.screenPoint) < 3 && !drag.handleOrigin) { this.publish({ paths: drag.before.paths }); return; }
    if (drag.kind === 'maybe-pull') {
      const handle = point[drag.missing === 'inHandle' ? 'outHandle' : 'inHandle']!;
      const dot = (local.x - point.anchor.x) * (handle.x - point.anchor.x) + (local.y - point.anchor.y) * (handle.y - point.anchor.y);
      drag.kind = dot < 0 ? drag.missing! : 'anchor';
    }
    if (drag.kind === 'anchor') {
      const dx = local.x - drag.start.x, dy = local.y - drag.start.y;
      this.publish({ paths: this.mapPoints(drag.before.paths, drag.before.selectedPoints, (copy) => {
        copy.anchor = { x: copy.anchor.x + dx, y: copy.anchor.y + dy };
        if (copy.inHandle) { copy.inHandle = { x: copy.inHandle.x + dx, y: copy.inHandle.y + dy }; }
        if (copy.outHandle) { copy.outHandle = { x: copy.outHandle.x + dx, y: copy.outHandle.y + dy }; }
      }) });
    } else {
      const handle = drag.kind === 'pull' ? getBezierPointPullHandleType(path.points, path.isClosed, ref.pointIndex, local) : drag.kind === 'inHandle' ? 'inHandle' : 'outHandle';
      const target = drag.handleOrigin
        ? { x: drag.handleOrigin.x + local.x - drag.start.x, y: drag.handleOrigin.y + local.y - drag.start.y }
        : local;
      this.publish({ paths: this.mapPoints(drag.before.paths, [ref], (copy) => setBezierPointHandle(copy, handle, target)) });
    }
  };
  onPointerUp = (): void => {
    if (!this.drag) { return; }
    this.drag = null; this.publish({ selectionRect: null }); this.record();
  };
  onPointerCancel = (): void => {
    if (this.drag) { const before = this.drag.before; this.drag = null; this.publish({ ...before, selectionRect: null }); }
  };
  onDeactivate = (): void => this.onPointerCancel();
  onKeyCommand = (_ctx: ToolContext, command: 'apply' | 'cancel'): void => { if (command === 'apply') { this.apply(); } else { this.cancel(); } };
}
