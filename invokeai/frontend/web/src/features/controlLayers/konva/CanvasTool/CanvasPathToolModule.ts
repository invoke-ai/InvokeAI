import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { addCoords, getPrefixedId, offsetCoord } from 'features/controlLayers/konva/util';
import type {
  CanvasBezierPathState,
  CanvasEntityIdentifier,
  Coordinate,
  Tool,
} from 'features/controlLayers/store/types';
import { getBezierPathState } from 'features/controlLayers/store/util';
import type { BezierPointType } from 'features/controlLayers/util/bezierPath';
import {
  anchorsToBezierPoints,
  buildBezierPathData,
  findNearestBezierPathSegment,
  getBezierPathHitSamplesPerSegment,
  getBezierPointPullHandleType,
  setBezierPointHandle,
  setBezierPointTypes,
  smoothBezierPathPoints,
  splitBezierSegmentAt,
} from 'features/controlLayers/util/bezierPath';
import {
  canJoinVectorPathEndpoints,
  canSplitVectorPathAtPoints,
  deleteVectorPathPoints,
  joinVectorPathEndpoints,
  splitVectorPathAtPoints,
} from 'features/controlLayers/util/vectorPathTopology';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasPathToolModuleConfig = {
  ANCHOR_RADIUS_PX: number;
  ANCHOR_STROKE_WIDTH_PX: number;
  START_POINT_HOVER_RADIUS_DELTA_PX: number;
  PATH_CLOSE_RADIUS_PX: number;
  HANDLE_RADIUS_PX: number;
  HANDLE_STROKE_WIDTH_PX: number;
  HANDLE_LINE_WIDTH_PX: number;
  HANDLE_PULL_INTENT_THRESHOLD_PX: number;
  PATH_HIT_TOLERANCE_PX: number;
  JOIN_POINT_WELD_RADIUS_PX: number;
  PREVIEW_STROKE_COLOR: string;
  PREVIEW_STROKE_WIDTH_PX: number;
  PREVIEW_POINT_FILL: string;
  PREVIEW_POINT_STROKE: string;
  EDIT_PATH_STROKE_COLOR: string;
  EDIT_ACTIVE_POINT_FILL: string;
  EDIT_ACTIVE_POINT_STROKE: string;
  EDIT_SELECTED_POINT_FILL: string;
  EDIT_SELECTED_POINT_STROKE: string;
  EDIT_INACTIVE_POINT_FILL: string;
  EDIT_INACTIVE_POINT_STROKE: string;
  EDIT_HANDLE_FILL: string;
  EDIT_HANDLE_STROKE: string;
  EDIT_HANDLE_LINE: string;
  EDIT_SELECTION_RECT_FILL: string;
  EDIT_SELECTION_RECT_STROKE: string;
};

type PathHandleType = 'inHandle' | 'outHandle';
type PointSelectionMode = 'replace' | 'add';
type CanvasPathPointRef = { pathId: string; pointIndex: number };

type CanvasPathEditDragTarget =
  | { pathId: string; pointIndex: number; type: 'anchor' }
  | {
      pathId: string;
      pointIndex: number;
      type: 'anchorOrHandle';
      missingHandleType: PathHandleType;
      startPointer: Coordinate;
    }
  | { pathId: string; pointIndex: number; type: 'pullHandles'; handleType: PathHandleType | null }
  | { pathId: string; pointIndex: number; type: PathHandleType }
  | {
      type: 'selectionRect';
      start: Coordinate;
      end: Coordinate;
      mode: PointSelectionMode;
      activePathId: string;
      initialSelectedPoints: CanvasPathPointRef[];
    };

type CanvasPathEditSnapshot = {
  paths: CanvasBezierPathState[];
  activePathId: string | null;
  activePointIndex: number | null;
  selectedPoints: CanvasPathPointRef[];
  activeHandle: PathHandleType | null;
};

type CanvasPathEditSession = {
  id: string;
  entityIdentifier: CanvasEntityIdentifier<'vector_layer'>;
  previousBaseTool: Tool;
  snapshotPaths: CanvasBezierPathState[];
  activePathId: string | null;
  activePointIndex: number | null;
  selectedPoints: CanvasPathPointRef[];
  activeHandle: PathHandleType | null;
  dragTarget: CanvasPathEditDragTarget | null;
  history: CanvasPathEditSnapshot[];
  historyIndex: number;
};

const MAX_EDIT_HISTORY = 64;

const DEFAULT_CONFIG: CanvasPathToolModuleConfig = {
  ANCHOR_RADIUS_PX: 4,
  ANCHOR_STROKE_WIDTH_PX: 2,
  START_POINT_HOVER_RADIUS_DELTA_PX: 2,
  PATH_CLOSE_RADIUS_PX: 10,
  HANDLE_RADIUS_PX: 3.5,
  HANDLE_STROKE_WIDTH_PX: 1.5,
  HANDLE_LINE_WIDTH_PX: 1,
  HANDLE_PULL_INTENT_THRESHOLD_PX: 3,
  PATH_HIT_TOLERANCE_PX: 10,
  JOIN_POINT_WELD_RADIUS_PX: 12,
  PREVIEW_STROKE_COLOR: 'rgba(90, 175, 255, 1)',
  PREVIEW_STROKE_WIDTH_PX: 1.5,
  PREVIEW_POINT_FILL: 'rgba(255, 255, 255, 1)',
  PREVIEW_POINT_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_PATH_STROKE_COLOR: 'rgba(90, 175, 255, 1)',
  EDIT_ACTIVE_POINT_FILL: 'rgba(90, 175, 255, 1)',
  EDIT_ACTIVE_POINT_STROKE: 'rgba(62, 214, 132, 1)',
  EDIT_SELECTED_POINT_FILL: 'rgba(90, 175, 255, 0.55)',
  EDIT_SELECTED_POINT_STROKE: 'rgba(255, 255, 255, 0.9)',
  EDIT_INACTIVE_POINT_FILL: 'rgba(255, 255, 255, 0.95)',
  EDIT_INACTIVE_POINT_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_HANDLE_FILL: 'rgba(255, 255, 255, 1)',
  EDIT_HANDLE_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_HANDLE_LINE: 'rgba(90, 175, 255, 0.75)',
  EDIT_SELECTION_RECT_FILL: 'rgba(90, 175, 255, 0.12)',
  EDIT_SELECTION_RECT_STROKE: 'rgba(90, 175, 255, 0.9)',
};

const getDistance = (a: Coordinate, b: Coordinate) => Math.hypot(a.x - b.x, a.y - b.y);
const getPointRefKey = (ref: CanvasPathPointRef) => `${ref.pathId}:${ref.pointIndex}`;
const isSamePointRef = (a: CanvasPathPointRef, b: CanvasPathPointRef) =>
  a.pathId === b.pathId && a.pointIndex === b.pointIndex;
const groupPointRefsByPath = (pointRefs: CanvasPathPointRef[]): Map<string, CanvasPathPointRef[]> => {
  const grouped = new Map<string, CanvasPathPointRef[]>();
  for (const pointRef of pointRefs) {
    const refs = grouped.get(pointRef.pathId) ?? [];
    refs.push(pointRef);
    grouped.set(pointRef.pathId, refs);
  }
  return grouped;
};

export class CanvasPathToolModule extends CanvasModuleBase {
  readonly type = 'path_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasPathToolModuleConfig = DEFAULT_CONFIG;

  $editSession = atom<CanvasPathEditSession | null>(null);
  $isExitConfirmationOpen = atom(false);

  private activeEntityIdentifier: CanvasEntityIdentifier<'vector_layer'> | null = null;
  private points: Coordinate[] = [];
  private previewPoint: Coordinate | null = null;

  konva: {
    group: Konva.Group;
    previewPath: Konva.Path;
    previewAnchorsGroup: Konva.Group;
    editPath: Konva.Path;
    editSelectionRect: Konva.Rect;
    editAnchorsGroup: Konva.Group;
    editHandlesGroup: Konva.Group;
  };

  constructor(parent: CanvasToolModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      previewPath: new Konva.Path({
        name: `${this.type}:preview_path`,
        listening: false,
        stroke: this.config.PREVIEW_STROKE_COLOR,
        strokeWidth: this.config.PREVIEW_STROKE_WIDTH_PX,
        fillEnabled: false,
        lineCap: 'round',
        lineJoin: 'round',
        visible: false,
        perfectDrawEnabled: false,
      }),
      previewAnchorsGroup: new Konva.Group({
        name: `${this.type}:preview_anchors_group`,
        listening: false,
        visible: false,
      }),
      editPath: new Konva.Path({
        name: `${this.type}:edit_path`,
        listening: false,
        stroke: this.config.EDIT_PATH_STROKE_COLOR,
        strokeWidth: this.config.PREVIEW_STROKE_WIDTH_PX,
        fillEnabled: false,
        lineCap: 'round',
        lineJoin: 'round',
        visible: false,
        perfectDrawEnabled: false,
      }),
      editSelectionRect: new Konva.Rect({
        name: `${this.type}:edit_selection_rect`,
        listening: false,
        visible: false,
        fill: this.config.EDIT_SELECTION_RECT_FILL,
        stroke: this.config.EDIT_SELECTION_RECT_STROKE,
        perfectDrawEnabled: false,
      }),
      editAnchorsGroup: new Konva.Group({
        name: `${this.type}:edit_anchors_group`,
        listening: false,
        visible: false,
      }),
      editHandlesGroup: new Konva.Group({
        name: `${this.type}:edit_handles_group`,
        listening: false,
        visible: false,
      }),
    };

    this.konva.group.add(
      this.konva.previewPath,
      this.konva.previewAnchorsGroup,
      this.konva.editPath,
      this.konva.editSelectionRect,
      this.konva.editHandlesGroup,
      this.konva.editAnchorsGroup
    );
  }

  hasActiveCreateSession = (): boolean => {
    return this.points.length > 0;
  };

  hasActiveEditSession = (): boolean => {
    return Boolean(this.$editSession.get());
  };

  getCanStartEdit = (entityIdentifier: CanvasEntityIdentifier<'vector_layer'>): boolean => {
    const adapter = this.manager.getAdapter(entityIdentifier);
    return Boolean(
      adapter &&
      adapter.state.type === 'vector_layer' &&
      adapter.state.paths.length > 0 &&
      !this.manager.$isBusy.get() &&
      !adapter.$isDisabled.get() &&
      !adapter.$isEntityTypeHidden.get() &&
      !adapter.$isLocked.get()
    );
  };

  getCanMutateEditSession = (): boolean => {
    const session = this.$editSession.get();
    if (!session || this.manager.$isBusy.get()) {
      return false;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (
      !selectedEntity ||
      selectedEntity.entityIdentifier.id !== session.entityIdentifier.id ||
      selectedEntity.entityIdentifier.type !== session.entityIdentifier.type
    ) {
      return false;
    }

    const adapter = this.manager.getAdapter(session.entityIdentifier);
    return Boolean(
      adapter &&
      adapter.state.type === 'vector_layer' &&
      !adapter.$isDisabled.get() &&
      !adapter.$isEntityTypeHidden.get() &&
      !adapter.$isLocked.get()
    );
  };

  hasActiveEditDragSession = (): boolean => {
    return Boolean(this.$editSession.get()?.dragTarget);
  };

  hasActiveSession = (): boolean => {
    return this.hasActiveCreateSession() || this.hasActiveEditSession();
  };

  startEdit = (entityIdentifier: CanvasEntityIdentifier<'vector_layer'>) => {
    const adapter = this.manager.getAdapter(entityIdentifier);
    if (!adapter || adapter.state.type !== 'vector_layer' || !this.getCanStartEdit(entityIdentifier)) {
      return;
    }

    const existingSession = this.$editSession.get();
    if (
      existingSession &&
      existingSession.entityIdentifier.id === entityIdentifier.id &&
      existingSession.entityIdentifier.type === entityIdentifier.type
    ) {
      this.activateEditTool();
      this.render();
      return;
    }

    const previousBaseTool = existingSession?.previousBaseTool ?? this.getPreviousBaseTool();
    if (existingSession) {
      this.acceptEditSession(false);
    }

    this.resetCreateState();
    const snapshotPaths = deepClone(adapter.state.paths);
    const activePathId = snapshotPaths[0]?.id ?? null;
    this.$editSession.set({
      id: getPrefixedId('path_edit_session'),
      entityIdentifier,
      previousBaseTool,
      snapshotPaths,
      activePathId,
      activePointIndex: null,
      selectedPoints: [],
      activeHandle: null,
      dragTarget: null,
      history: [
        {
          paths: deepClone(snapshotPaths),
          activePathId,
          activePointIndex: null,
          selectedPoints: [],
          activeHandle: null,
        },
      ],
      historyIndex: 0,
    });
    this.activateEditTool();
    this.render();
  };

  acceptEditSession = (restoreTool = true) => {
    const session = this.$editSession.get();
    this.$isExitConfirmationOpen.set(false);
    this.$editSession.set(null);
    if (session && restoreTool) {
      this.restorePreviousTool(session.previousBaseTool);
    }
    this.render();
  };

  cancelToolChange = () => {
    if (!this.hasActiveEditSession()) {
      this.$isExitConfirmationOpen.set(false);
      return;
    }

    this.$isExitConfirmationOpen.set(false);
    this.activateEditTool();
    this.render();
  };

  resetEditSession = () => {
    const session = this.$editSession.get();
    if (!session || !this.getCanMutateEditSession()) {
      return;
    }

    const paths = deepClone(session.snapshotPaths);
    const activePathId = paths.some((path) => path.id === session.activePathId)
      ? session.activePathId
      : (paths[0]?.id ?? null);
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId,
      activePointIndex: null,
      selectedPoints: [],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  setActivePointType = (pointType: BezierPointType) => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId || session.activePointIndex === null) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const path = nextPaths.find((candidate) => candidate.id === session.activePathId);
    if (!path || !path.points[session.activePointIndex]) {
      return;
    }

    const selectedPoints =
      session.selectedPoints.length > 0
        ? session.selectedPoints
        : [{ pathId: session.activePathId, pointIndex: session.activePointIndex }];
    const selectedPointsByPath = groupPointRefsByPath(selectedPoints);
    for (const [pathId, pointRefs] of selectedPointsByPath) {
      const selectedPath = nextPaths.find((candidate) => candidate.id === pathId);
      if (!selectedPath) {
        continue;
      }
      const isActivePath = pathId === session.activePathId;
      setBezierPointTypes(
        selectedPath.points,
        pointRefs.map((pointRef) => pointRef.pointIndex),
        pointType,
        isActivePath ? session.activePointIndex : null,
        isActivePath ? session.activeHandle : null
      );
    }
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.recordEditHistory();
    this.render();
  };

  smoothActivePath = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const path = nextPaths.find((candidate) => candidate.id === session.activePathId);
    if (!path || path.points.length < 2) {
      return;
    }

    path.points = smoothBezierPathPoints(path.points, path.isClosed);
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.recordEditHistory();
    this.render();
  };

  smoothSelectedPoints = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || session.selectedPoints.length === 0) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const selectedPointsByPath = groupPointRefsByPath(session.selectedPoints);
    for (const [pathId, pointRefs] of selectedPointsByPath) {
      const path = nextPaths.find((candidate) => candidate.id === pathId);
      if (!path || path.points.length < 2) {
        continue;
      }
      const smoothedPoints = smoothBezierPathPoints(path.points, path.isClosed);
      for (const pointRef of pointRefs) {
        const smoothedPoint = smoothedPoints[pointRef.pointIndex];
        if (smoothedPoint) {
          path.points[pointRef.pointIndex] = smoothedPoint;
        }
      }
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.recordEditHistory();
    this.render();
  };

  deleteActivePath = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId) {
      return;
    }

    const deletedPathIndex = activeEntity.state.paths.findIndex((path) => path.id === session.activePathId);
    if (deletedPathIndex === -1) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    nextPaths.splice(deletedPathIndex, 1);
    const nextActivePath = nextPaths[Math.min(deletedPathIndex, nextPaths.length - 1)] ?? null;

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId: nextActivePath?.id ?? null,
      activePointIndex: null,
      selectedPoints: [],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  copyActivePath = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId) {
      return;
    }

    const activePathIndex = activeEntity.state.paths.findIndex((path) => path.id === session.activePathId);
    const activePath = activeEntity.state.paths[activePathIndex];
    if (!activePath) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const copiedPath = deepClone(activePath);
    copiedPath.id = getPrefixedId('bezier_path');
    nextPaths.splice(activePathIndex + 1, 0, copiedPath);

    const activePointIndex =
      session.activePointIndex !== null && copiedPath.points[session.activePointIndex]
        ? session.activePointIndex
        : copiedPath.points.length > 0
          ? 0
          : null;
    const selectedPoints = copiedPath.points.map((_, pointIndex) => ({ pathId: copiedPath.id, pointIndex }));

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId: copiedPath.id,
      activePointIndex,
      selectedPoints,
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  extractActivePath = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId) {
      return;
    }

    this.manager.stateApi.extractVectorPath({
      entityIdentifier: session.entityIdentifier,
      pathId: session.activePathId,
    });
    this.acceptEditSession();
  };

  startTransformActivePath = async () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !session.activePathId) {
      return;
    }

    await activeEntity.transformer.startTransform({ vectorPathId: session.activePathId });
  };

  applyActivePathTransform = (matrix: [number, number, number, number, number, number]) => {
    const session = this.$editSession.get();
    const adapter = session ? this.manager.getAdapter(session.entityIdentifier) : null;
    if (!session || !adapter || adapter.state.type !== 'vector_layer' || !session.activePathId) {
      return;
    }

    this.manager.stateApi.transformVectorPath({
      entityIdentifier: session.entityIdentifier,
      pathId: session.activePathId,
      matrix,
      undoGroup: session.id,
    });
    this.recordEditHistory();
  };

  canSplitSelectedPoints = (): boolean => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return false;
    }
    const selectedPoints = this.getSelectedPointRefs(session);
    return canSplitVectorPathAtPoints(activeEntity.state.paths, selectedPoints);
  };

  splitSelectedPoints = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !this.canSplitSelectedPoints()) {
      return;
    }
    const selectedPoints = this.getSelectedPointRefs(session);
    const selectedPointsByPath = groupPointRefsByPath(selectedPoints);
    const lastSelectedPoint = selectedPoints.at(-1);
    const nextPaths: CanvasBezierPathState[] = [];
    let activePathId = session.activePathId;
    let activePointIndex = session.activePointIndex;
    let didSplit = false;

    for (const path of activeEntity.state.paths) {
      const pointRefs = selectedPointsByPath.get(path.id);
      if (!pointRefs) {
        nextPaths.push(deepClone(path));
        continue;
      }
      const splitResult = splitVectorPathAtPoints(
        path,
        pointRefs.map((pointRef) => pointRef.pointIndex),
        () => getPrefixedId('bezier_path')
      );
      if (!splitResult) {
        nextPaths.push(deepClone(path));
        continue;
      }
      nextPaths.push(...splitResult.paths);
      didSplit = true;
      if (lastSelectedPoint?.pathId === path.id || activePathId === null) {
        activePathId = splitResult.activePathId;
        activePointIndex = splitResult.activePointIndex;
      }
    }

    if (!didSplit || activePathId === null || activePointIndex === null) {
      return;
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId,
      activePointIndex,
      selectedPoints: [{ pathId: activePathId, pointIndex: activePointIndex }],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  canJoinSelectedEndpoints = (): boolean => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return false;
    }
    return canJoinVectorPathEndpoints(
      activeEntity.state.paths,
      session.selectedPoints,
      this.manager.stage.unscale(this.config.JOIN_POINT_WELD_RADIUS_PX)
    );
  };

  joinSelectedEndpoints = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity || !this.canJoinSelectedEndpoints()) {
      return;
    }
    const [sourceRef, targetRef] = session.selectedPoints;
    if (!sourceRef || !targetRef) {
      return;
    }
    const sourcePathIndex = activeEntity.state.paths.findIndex((path) => path.id === sourceRef.pathId);
    const targetPathIndex = activeEntity.state.paths.findIndex((path) => path.id === targetRef.pathId);
    const sourcePath = activeEntity.state.paths[sourcePathIndex];
    const targetPath = activeEntity.state.paths[targetPathIndex];
    const sourcePoint = sourcePath?.points[sourceRef.pointIndex];
    const targetPoint = targetPath?.points[targetRef.pointIndex];
    if (!sourcePath || !targetPath || !sourcePoint || !targetPoint) {
      return;
    }

    const shouldWeld =
      getDistance(sourcePoint.anchor, targetPoint.anchor) <=
      this.manager.stage.unscale(this.config.JOIN_POINT_WELD_RADIUS_PX);
    const joinResult = joinVectorPathEndpoints(
      sourcePath,
      sourceRef.pointIndex,
      targetPath,
      targetRef.pointIndex,
      shouldWeld
    );
    if (!joinResult) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    if (sourcePath.id === targetPath.id) {
      nextPaths.splice(sourcePathIndex, 1, joinResult.path);
    } else {
      const insertIndex = Math.min(sourcePathIndex, targetPathIndex);
      nextPaths.splice(Math.max(sourcePathIndex, targetPathIndex), 1);
      nextPaths.splice(Math.min(sourcePathIndex, targetPathIndex), 1);
      nextPaths.splice(insertIndex, 0, joinResult.path);
    }
    const activePathId = joinResult.path.id;
    const activePointIndex = joinResult.activePointIndex;

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId,
      activePointIndex,
      selectedPoints: [{ pathId: activePathId, pointIndex: activePointIndex }],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  canUndoEditSession = (): boolean => {
    const session = this.$editSession.get();
    return Boolean(session && session.historyIndex > 0);
  };

  canRedoEditSession = (): boolean => {
    const session = this.$editSession.get();
    return Boolean(session && session.historyIndex < session.history.length - 1);
  };

  undoEditSession = () => {
    const session = this.$editSession.get();
    if (!session || !this.getCanMutateEditSession() || session.historyIndex <= 0) {
      return;
    }

    this.restoreEditSnapshot(session, session.historyIndex - 1);
  };

  redoEditSession = () => {
    const session = this.$editSession.get();
    if (!session || !this.getCanMutateEditSession() || session.historyIndex >= session.history.length - 1) {
      return;
    }

    this.restoreEditSnapshot(session, session.historyIndex + 1);
  };

  addPathToEditSession = (path: CanvasBezierPathState): boolean => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return false;
    }

    const nextPaths = [...deepClone(activeEntity.state.paths), deepClone(path)];
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId: path.id,
      activePointIndex: null,
      selectedPoints: [],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
    return true;
  };

  onToolChanged = () => {
    const tool = this.parent.$tool.get();
    if ((tool === 'move' || tool === 'rect') && this.hasActiveEditSession()) {
      this.resetCreateState();
      this.render();
      return;
    }
    if (tool !== 'path' && !this.isTemporaryToolSwitch(tool, this.parent.$baseTool.get())) {
      if (this.hasActiveEditSession()) {
        this.$isExitConfirmationOpen.set(true);
        this.render();
        return;
      }
      this.resetCreateState();
    }
  };

  syncCursorStyle = () => {
    this.manager.stage.setCursor(this.parent.getCanDraw() ? 'crosshair' : 'not-allowed');
  };

  render = () => {
    const tool = this.parent.$tool.get();
    const isTemporaryToolSwitch = this.isTemporaryToolSwitch(tool, this.parent.$baseTool.get());

    if (tool !== 'path' && !isTemporaryToolSwitch) {
      this.hideCreatePreview();
      this.hideEditOverlay();
      return;
    }

    if (tool === 'path' || (tool === 'move' && this.hasActiveEditSession())) {
      this.syncCursorStyle();
    }

    if (this.hasActiveEditSession()) {
      this.hideCreatePreview();
      this.syncEditOverlay();
      return;
    }

    this.hideEditOverlay();
    this.syncCreatePreview();
  };

  cancel = () => {
    if (this.hasActiveEditSession()) {
      this.discardEditSession();
      return;
    }

    this.resetCreateState();
    this.render();
  };

  commit = () => {
    if (this.hasActiveEditSession()) {
      this.acceptEditSession();
      return;
    }

    this.commitCreateSession(false);
  };

  private commitCreateSession = (isClosed: boolean) => {
    const activeEntity = this.getCreateEntityAdapter();

    if (!activeEntity || this.points.length < 2) {
      this.resetCreateState();
      this.render();
      return;
    }

    this.manager.stateApi.addVectorPath({
      entityIdentifier: activeEntity.entityIdentifier,
      path: getBezierPathState(getPrefixedId('bezier_path'), {
        points: anchorsToBezierPoints(this.points),
        isClosed,
      }),
    });

    this.resetCreateState();
    this.render();
  };

  onStagePointerDown = (e: KonvaEventObject<PointerEvent>) => {
    if (this.hasActiveEditSession()) {
      this.onEditPointerDown(e);
      return;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!selectedEntity || selectedEntity.state.type !== 'vector_layer' || !cursorPos || e.evt.button !== 0) {
      return;
    }

    if (
      this.activeEntityIdentifier &&
      (this.activeEntityIdentifier.id !== selectedEntity.entityIdentifier.id ||
        this.activeEntityIdentifier.type !== selectedEntity.entityIdentifier.type)
    ) {
      this.resetCreateState();
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, selectedEntity.state.position);
    if (this.getIsClosingCreatePath(point)) {
      this.commitCreateSession(true);
      return;
    }

    const pathPoint = this.getPathPoint(point, e.evt.shiftKey);

    this.activeEntityIdentifier = { id: selectedEntity.state.id, type: 'vector_layer' };
    this.points = [...this.points, pathPoint];
    this.previewPoint = pathPoint;
    this.render();
  };

  onStagePointerMove = (e: KonvaEventObject<PointerEvent>) => {
    if (this.hasActiveEditSession()) {
      this.onEditPointerMove(e.evt);
      return;
    }

    const activeEntity = this.getCreateEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!activeEntity || !cursorPos || !this.hasActiveCreateSession()) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
    this.previewPoint = this.getIsClosingCreatePath(point) ? point : this.getPathPoint(point, e.evt.shiftKey);
    this.render();
  };

  onStagePointerUp = (_e: KonvaEventObject<PointerEvent>) => {
    if (!this.hasActiveEditSession()) {
      return;
    }

    this.clearEditDragTarget();
  };

  onWindowPointerMove = (e: PointerEvent) => {
    if (!this.hasActiveEditDragSession()) {
      return;
    }

    this.onEditPointerMove(e);
  };

  onWindowPointerUp = () => {
    this.clearEditDragTarget();
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      activeEntityIdentifier: this.activeEntityIdentifier,
      points: this.points,
      previewPoint: this.previewPoint,
      editSession: this.$editSession.get(),
    };
  };

  private getCreateEntityAdapter = () => {
    if (!this.activeEntityIdentifier) {
      return null;
    }

    const adapter = this.manager.getAdapter(this.activeEntityIdentifier);
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return null;
    }

    return adapter;
  };

  private getEditSessionAdapter = () => {
    const session = this.$editSession.get();
    if (!session || !this.getCanMutateEditSession()) {
      return null;
    }

    const adapter = this.manager.getAdapter(session.entityIdentifier);
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return null;
    }

    return adapter;
  };

  private getSelectedPointRefs = (session: CanvasPathEditSession): CanvasPathPointRef[] => {
    if (session.selectedPoints.length > 0) {
      return session.selectedPoints;
    }
    if (session.activePathId && session.activePointIndex !== null) {
      return [{ pathId: session.activePathId, pointIndex: session.activePointIndex }];
    }
    return [];
  };

  private recordEditHistory = () => {
    const session = this.$editSession.get();
    const adapter = session ? this.manager.getAdapter(session.entityIdentifier) : null;
    if (!session || !adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    const currentSnapshot = session.history[session.historyIndex];
    if (currentSnapshot && JSON.stringify(currentSnapshot.paths) === JSON.stringify(adapter.state.paths)) {
      return;
    }

    const snapshot: CanvasPathEditSnapshot = {
      paths: deepClone(adapter.state.paths),
      activePathId: session.activePathId,
      activePointIndex: session.activePointIndex,
      selectedPoints: session.selectedPoints.map((pointRef) => ({ ...pointRef })),
      activeHandle: session.activeHandle,
    };
    const history = [...session.history.slice(0, session.historyIndex + 1), snapshot];
    if (history.length > MAX_EDIT_HISTORY) {
      history.shift();
    }
    this.$editSession.set({ ...session, history, historyIndex: history.length - 1 });
  };

  private restoreEditSnapshot = (session: CanvasPathEditSession, historyIndex: number) => {
    const snapshot = session.history[historyIndex];
    if (!snapshot) {
      return;
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: deepClone(snapshot.paths),
      undoGroup: session.id,
    });
    this.$editSession.set({
      ...session,
      activePathId: snapshot.activePathId,
      activePointIndex: snapshot.activePointIndex,
      selectedPoints: snapshot.selectedPoints.map((pointRef) => ({ ...pointRef })),
      activeHandle: snapshot.activeHandle,
      dragTarget: null,
      historyIndex,
    });
    this.render();
  };

  discardEditSession = (restoreTool = true) => {
    const session = this.$editSession.get();
    if (!session) {
      return;
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: deepClone(session.snapshotPaths),
      undoGroup: session.id,
    });
    this.$isExitConfirmationOpen.set(false);
    this.$editSession.set(null);
    if (restoreTool) {
      this.restorePreviousTool(session.previousBaseTool);
    }
    this.render();
  };

  private getEntityRelativePoint = (point: Coordinate, position: Coordinate): Coordinate => {
    return offsetCoord(point, position);
  };

  private getPathPoint = (point: Coordinate, shouldSnap: boolean): Coordinate => {
    if (!shouldSnap) {
      return point;
    }

    const lastPoint = this.points.at(-1);
    if (!lastPoint) {
      return point;
    }

    const dx = point.x - lastPoint.x;
    const dy = point.y - lastPoint.y;
    const distance = Math.hypot(dx, dy);
    if (distance === 0) {
      return point;
    }

    const snapAngle = Math.PI / 4;
    const angle = Math.atan2(dy, dx);
    const snappedAngle = Math.round(angle / snapAngle) * snapAngle;

    return {
      x: lastPoint.x + Math.cos(snappedAngle) * distance,
      y: lastPoint.y + Math.sin(snappedAngle) * distance,
    };
  };

  private getPathCloseRadius = (): number => {
    return this.manager.stage.unscale(this.config.PATH_CLOSE_RADIUS_PX);
  };

  private getIsClosingCreatePath = (point: Coordinate): boolean => {
    const startPoint = this.points[0];
    if (!startPoint || this.points.length < 3) {
      return false;
    }

    return getDistance(point, startPoint) <= this.getPathCloseRadius();
  };

  private syncCreatePreview = () => {
    const activeEntity = this.getCreateEntityAdapter();
    if (!activeEntity || this.points.length === 0) {
      this.hideCreatePreview();
      return;
    }

    const entityPosition = activeEntity.state.position;
    const isClosingPath = this.previewPoint ? this.getIsClosingCreatePath(this.previewPoint) : false;
    const previewAnchors = isClosingPath
      ? [...this.points]
      : this.previewPoint
        ? [...this.points, this.previewPoint]
        : [...this.points];
    const previewPoints = previewAnchors.map((point) => addCoords(point, entityPosition));
    const data = buildBezierPathData(anchorsToBezierPoints(previewPoints), isClosingPath);
    const anchorRadius = this.manager.stage.unscale(this.config.ANCHOR_RADIUS_PX);
    const anchorStrokeWidth = this.manager.stage.unscale(this.config.ANCHOR_STROKE_WIDTH_PX);
    const startPointHoverRadiusDelta = this.manager.stage.unscale(this.config.START_POINT_HOVER_RADIUS_DELTA_PX);

    this.konva.previewPath.setAttrs({
      data,
      visible: Boolean(data),
      strokeWidth: this.manager.stage.unscale(this.config.PREVIEW_STROKE_WIDTH_PX),
    });

    this.konva.previewAnchorsGroup.destroyChildren();
    for (let pointIndex = 0; pointIndex < this.points.length; pointIndex += 1) {
      const point = this.points[pointIndex];
      if (!point) {
        continue;
      }
      const stagePoint = addCoords(point, entityPosition);
      this.konva.previewAnchorsGroup.add(
        new Konva.Circle({
          x: stagePoint.x,
          y: stagePoint.y,
          radius: anchorRadius + (pointIndex === 0 && isClosingPath ? startPointHoverRadiusDelta : 0),
          fill: this.config.PREVIEW_POINT_FILL,
          stroke: this.config.PREVIEW_POINT_STROKE,
          strokeWidth: anchorStrokeWidth,
          listening: false,
          perfectDrawEnabled: false,
        })
      );
    }
    this.konva.previewAnchorsGroup.visible(this.points.length > 0);
  };

  private syncEditOverlay = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      this.hideEditOverlay();
      return;
    }

    const activePath =
      activeEntity.state.paths.find((path) => path.id === session.activePathId) ?? activeEntity.state.paths[0] ?? null;
    if (!activePath || activePath.points.length < 2) {
      this.hideEditOverlay();
      return;
    }

    const entityPosition = activeEntity.state.position;
    const data = buildBezierPathData(
      activePath.points.map((point) => ({
        ...point,
        anchor: addCoords(point.anchor, entityPosition),
        inHandle: point.inHandle ? addCoords(point.inHandle, entityPosition) : null,
        outHandle: point.outHandle ? addCoords(point.outHandle, entityPosition) : null,
      })),
      activePath.isClosed
    );

    this.konva.editPath.setAttrs({
      data,
      visible: Boolean(data),
      strokeWidth: this.manager.stage.unscale(this.config.PREVIEW_STROKE_WIDTH_PX * 1.5),
    });

    const anchorRadius = this.manager.stage.unscale(this.config.ANCHOR_RADIUS_PX);
    const anchorStrokeWidth = this.manager.stage.unscale(this.config.ANCHOR_STROKE_WIDTH_PX);
    const handleRadius = this.manager.stage.unscale(this.config.HANDLE_RADIUS_PX);
    const handleStrokeWidth = this.manager.stage.unscale(this.config.HANDLE_STROKE_WIDTH_PX);
    const handleLineWidth = this.manager.stage.unscale(this.config.HANDLE_LINE_WIDTH_PX);

    if (session.dragTarget?.type === 'selectionRect') {
      const start = addCoords(session.dragTarget.start, entityPosition);
      const end = addCoords(session.dragTarget.end, entityPosition);
      this.konva.editSelectionRect.setAttrs({
        x: Math.min(start.x, end.x),
        y: Math.min(start.y, end.y),
        width: Math.abs(end.x - start.x),
        height: Math.abs(end.y - start.y),
        strokeWidth: this.manager.stage.unscale(1),
        dash: [this.manager.stage.unscale(4), this.manager.stage.unscale(3)],
        visible: true,
      });
    } else {
      this.konva.editSelectionRect.visible(false);
    }

    this.konva.editAnchorsGroup.destroyChildren();
    const selectedPointKeys = new Set(session.selectedPoints.map(getPointRefKey));
    for (const path of activeEntity.state.paths) {
      path.points.forEach((point, pointIndex) => {
        const pointRef = { pathId: path.id, pointIndex };
        const isActivePoint = path.id === session.activePathId && pointIndex === session.activePointIndex;
        const isSelectedPoint = selectedPointKeys.has(getPointRefKey(pointRef));
        const stagePoint = addCoords(point.anchor, entityPosition);
        this.konva.editAnchorsGroup.add(
          new Konva.Circle({
            x: stagePoint.x,
            y: stagePoint.y,
            radius: anchorRadius,
            fill: isActivePoint
              ? this.config.EDIT_ACTIVE_POINT_FILL
              : isSelectedPoint
                ? this.config.EDIT_SELECTED_POINT_FILL
                : this.config.EDIT_INACTIVE_POINT_FILL,
            stroke: isActivePoint
              ? this.config.EDIT_ACTIVE_POINT_STROKE
              : isSelectedPoint
                ? this.config.EDIT_SELECTED_POINT_STROKE
                : this.config.EDIT_INACTIVE_POINT_STROKE,
            strokeWidth: anchorStrokeWidth,
            listening: false,
            perfectDrawEnabled: false,
          })
        );
      });
    }
    this.konva.editAnchorsGroup.visible(activeEntity.state.paths.some((path) => path.points.length > 0));

    this.konva.editHandlesGroup.destroyChildren();
    const handlePointRefs = session.selectedPoints.map((pointRef) => ({ ...pointRef }));
    if (session.activePathId && session.activePointIndex !== null) {
      const activePointRef = { pathId: session.activePathId, pointIndex: session.activePointIndex };
      if (!handlePointRefs.some((pointRef) => isSamePointRef(pointRef, activePointRef))) {
        handlePointRefs.push(activePointRef);
      }
    }
    handlePointRefs.forEach((pointRef) => {
      const path = activeEntity.state.paths.find((candidate) => candidate.id === pointRef.pathId);
      const bezierPoint = path?.points[pointRef.pointIndex];
      if (!bezierPoint) {
        return;
      }

      const anchor = addCoords(bezierPoint.anchor, entityPosition);
      const handles: Array<{ type: PathHandleType; point: Coordinate | null }> = [
        { type: 'inHandle', point: bezierPoint.inHandle ? addCoords(bezierPoint.inHandle, entityPosition) : null },
        { type: 'outHandle', point: bezierPoint.outHandle ? addCoords(bezierPoint.outHandle, entityPosition) : null },
      ];

      handles.forEach(({ point }) => {
        if (!point) {
          return;
        }
        this.konva.editHandlesGroup.add(
          new Konva.Line({
            points: [anchor.x, anchor.y, point.x, point.y],
            stroke: this.config.EDIT_HANDLE_LINE,
            strokeWidth: handleLineWidth,
            listening: false,
            perfectDrawEnabled: false,
          })
        );
        this.konva.editHandlesGroup.add(
          new Konva.Circle({
            x: point.x,
            y: point.y,
            radius: handleRadius,
            fill: this.config.EDIT_HANDLE_FILL,
            stroke: this.config.EDIT_HANDLE_STROKE,
            strokeWidth: handleStrokeWidth,
            listening: false,
            perfectDrawEnabled: false,
          })
        );
      });
    });
    this.konva.editHandlesGroup.visible(handlePointRefs.length > 0);
  };

  private hideCreatePreview = () => {
    this.konva.previewPath.visible(false);
    this.konva.previewAnchorsGroup.visible(false);
    this.konva.previewAnchorsGroup.destroyChildren();
  };

  private hideEditOverlay = () => {
    this.konva.editPath.visible(false);
    this.konva.editSelectionRect.visible(false);
    this.konva.editAnchorsGroup.visible(false);
    this.konva.editAnchorsGroup.destroyChildren();
    this.konva.editHandlesGroup.visible(false);
    this.konva.editHandlesGroup.destroyChildren();
  };

  private activateEditTool = () => {
    this.parent.setBaseTool('move');
    this.parent.clearTemporaryToolHotkeys();
  };

  private getPreviousBaseTool = (): Tool => {
    const baseTool = this.parent.$baseTool.get();
    return baseTool === 'path' ? 'rect' : baseTool;
  };

  private restorePreviousTool = (tool: Tool) => {
    if (this.parent.$baseTool.get() !== 'path' && this.parent.$baseTool.get() !== 'move') {
      return;
    }
    this.parent.setBaseTool(tool);
    this.parent.clearTemporaryToolHotkeys();
  };

  private isTemporaryToolSwitch = (tool: Tool, baseTool: Tool) => {
    const preservesPathSession =
      baseTool === 'path' || ((baseTool === 'move' || baseTool === 'rect') && this.hasActiveEditSession());
    return preservesPathSession && (tool === 'view' || tool === 'colorPicker' || tool === baseTool);
  };

  private resetCreateState = () => {
    this.activeEntityIdentifier = null;
    this.points = [];
    this.previewPoint = null;
    this.hideCreatePreview();
  };

  private onEditPointerDown = (e: KonvaEventObject<PointerEvent>) => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!session || !activeEntity || !cursorPos || e.evt.button !== 0) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
    const anchorHitRadius = this.manager.stage.unscale(this.config.ANCHOR_RADIUS_PX + 4);
    const handleHitRadius = this.manager.stage.unscale(this.config.HANDLE_RADIUS_PX + 4);
    const pathHitTolerance = this.manager.stage.unscale(this.config.PATH_HIT_TOLERANCE_PX);
    const pathHitSamplesPerSegment = getBezierPathHitSamplesPerSegment(this.manager.stage.getScale());
    const activePath =
      activeEntity.state.paths.find((path) => path.id === session.activePathId) ?? activeEntity.state.paths[0] ?? null;

    if (activePath) {
      const activePointRef =
        session.activePointIndex === null ? null : { pathId: activePath.id, pointIndex: session.activePointIndex };
      const handlePointRefs = [
        ...(activePointRef ? [activePointRef] : []),
        ...session.selectedPoints.filter((pointRef) => !activePointRef || !isSamePointRef(pointRef, activePointRef)),
      ];
      for (const pointRef of handlePointRefs) {
        const handlePath = activeEntity.state.paths.find((candidate) => candidate.id === pointRef.pathId);
        if (!handlePath) {
          continue;
        }
        const handleHit = this.findHandleHit(handlePath, pointRef.pointIndex, point, handleHitRadius);
        if (!handleHit) {
          continue;
        }

        const selectedPoints = session.selectedPoints.some((selectedPoint) => isSamePointRef(selectedPoint, pointRef))
          ? [...session.selectedPoints.filter((selectedPoint) => !isSamePointRef(selectedPoint, pointRef)), pointRef]
          : [pointRef];
        this.$editSession.set({
          ...session,
          activePathId: pointRef.pathId,
          activePointIndex: pointRef.pointIndex,
          selectedPoints,
          activeHandle: handleHit,
          dragTarget: {
            pathId: pointRef.pathId,
            pointIndex: pointRef.pointIndex,
            type: handleHit,
          },
        });
        this.render();
        return;
      }
    }

    const anchorHit = this.findAnchorHit(activeEntity.state.paths, point, anchorHitRadius, session.activePathId);
    if (anchorHit) {
      const clickedPath = activeEntity.state.paths.find((path) => path.id === anchorHit.pathId);
      const clickedPoint = clickedPath?.points[anchorHit.pointIndex];
      const clickedPointRef = { pathId: anchorHit.pathId, pointIndex: anchorHit.pointIndex };
      const isPointSelected = session.selectedPoints.some((pointRef) => isSamePointRef(pointRef, clickedPointRef));
      const isAdditiveSelection = e.evt.ctrlKey || e.evt.metaKey;
      const selectedPoints = isAdditiveSelection
        ? [...session.selectedPoints.filter((pointRef) => !isSamePointRef(pointRef, clickedPointRef)), clickedPointRef]
        : isPointSelected
          ? [
              ...session.selectedPoints.filter((pointRef) => !isSamePointRef(pointRef, clickedPointRef)),
              clickedPointRef,
            ]
          : [clickedPointRef];
      const shouldPullHandles =
        Boolean(clickedPoint) &&
        session.activePathId === anchorHit.pathId &&
        session.activePointIndex === anchorHit.pointIndex &&
        selectedPoints.length <= 1 &&
        !clickedPoint?.inHandle &&
        !clickedPoint?.outHandle;
      const missingCornerHandleType =
        clickedPath && clickedPoint ? this.getMissingCornerHandleType(clickedPath, anchorHit.pointIndex) : null;
      let dragTarget: CanvasPathEditDragTarget;
      if (shouldPullHandles) {
        dragTarget = {
          pathId: anchorHit.pathId,
          pointIndex: anchorHit.pointIndex,
          type: 'pullHandles',
          handleType: null,
        };
      } else if (
        missingCornerHandleType &&
        session.activePathId === anchorHit.pathId &&
        session.activePointIndex === anchorHit.pointIndex &&
        selectedPoints.length <= 1
      ) {
        dragTarget = {
          pathId: anchorHit.pathId,
          pointIndex: anchorHit.pointIndex,
          type: 'anchorOrHandle',
          missingHandleType: missingCornerHandleType,
          startPointer: point,
        };
      } else {
        dragTarget = {
          pathId: anchorHit.pathId,
          pointIndex: anchorHit.pointIndex,
          type: 'anchor',
        };
      }

      this.$editSession.set({
        ...session,
        activePathId: anchorHit.pathId,
        activePointIndex: anchorHit.pointIndex,
        selectedPoints,
        activeHandle: null,
        dragTarget,
      });
      this.render();
      return;
    }

    if (e.evt.shiftKey && activePath) {
      const segmentHit = findNearestBezierPathSegment(
        activePath.points,
        activePath.isClosed,
        point,
        pathHitSamplesPerSegment
      );
      if (segmentHit && segmentHit.distance <= pathHitTolerance) {
        const insertedPointIndex = this.insertPoint(activePath.id, segmentHit.segmentIndex, segmentHit.t);
        if (insertedPointIndex !== null) {
          this.$editSession.set({
            ...session,
            activePathId: activePath.id,
            activePointIndex: insertedPointIndex,
            selectedPoints: [{ pathId: activePath.id, pointIndex: insertedPointIndex }],
            activeHandle: null,
            dragTarget: {
              pathId: activePath.id,
              pointIndex: insertedPointIndex,
              type: 'anchor',
            },
          });
          this.render();
        }
        return;
      }
    }

    const pathHit = this.findPathHit(activeEntity.state.paths, point, pathHitTolerance, pathHitSamplesPerSegment);
    if (pathHit) {
      this.$editSession.set({
        ...session,
        activePathId: pathHit.pathId,
        activePointIndex: null,
        selectedPoints: [],
        activeHandle: null,
        dragTarget: null,
      });
      this.render();
      return;
    }

    if (!activePath) {
      this.$editSession.set({
        ...session,
        activePointIndex: null,
        selectedPoints: [],
        activeHandle: null,
        dragTarget: null,
      });
      this.render();
      return;
    }

    const selectionMode: PointSelectionMode = e.evt.ctrlKey || e.evt.metaKey ? 'add' : 'replace';
    const initialSelectedPoints = session.selectedPoints;
    this.$editSession.set({
      ...session,
      activePathId: activePath.id,
      activePointIndex: selectionMode === 'replace' ? null : session.activePointIndex,
      selectedPoints: selectionMode === 'replace' ? [] : initialSelectedPoints,
      activeHandle: null,
      dragTarget: {
        type: 'selectionRect',
        start: point,
        end: point,
        mode: selectionMode,
        activePathId: activePath.id,
        initialSelectedPoints,
      },
    });
    this.render();
  };

  private onEditPointerMove = (_evt: PointerEvent) => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!session || !activeEntity || !session.dragTarget || !cursorPos) {
      return;
    }

    const dragTarget = session.dragTarget;
    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
    const nextPaths = deepClone(activeEntity.state.paths);
    if (dragTarget.type === 'selectionRect') {
      const hitPoints = this.getPointsInSelectionRect(
        nextPaths,
        dragTarget.start,
        point,
        dragTarget.activePathId,
        this.manager.stage.unscale(this.config.ANCHOR_RADIUS_PX + 4)
      );
      const selectedPoints = this.applyPointSelection(dragTarget.initialSelectedPoints, hitPoints, dragTarget.mode);
      const activePointRef = selectedPoints.at(-1) ?? null;

      this.$editSession.set({
        ...session,
        activePathId: activePointRef?.pathId ?? session.activePathId,
        activePointIndex: activePointRef?.pointIndex ?? null,
        selectedPoints,
        activeHandle: null,
        dragTarget: { ...dragTarget, end: point },
      });
      this.render();
      return;
    }

    const path = nextPaths.find((candidate) => candidate.id === dragTarget.pathId);
    const bezierPoint = path?.points[dragTarget.pointIndex];
    if (!path || !bezierPoint) {
      return;
    }
    let nextSession: CanvasPathEditSession | null = null;
    const moveAnchorPoints = (pointRefs: CanvasPathPointRef[]) => {
      const dx = point.x - bezierPoint.anchor.x;
      const dy = point.y - bezierPoint.anchor.y;
      for (const pointRef of pointRefs) {
        const selectedPath = nextPaths.find((candidate) => candidate.id === pointRef.pathId);
        const selectedPoint = selectedPath?.points[pointRef.pointIndex];
        if (!selectedPoint) {
          continue;
        }
        selectedPoint.anchor = { x: selectedPoint.anchor.x + dx, y: selectedPoint.anchor.y + dy };
        if (selectedPoint.inHandle) {
          selectedPoint.inHandle = { x: selectedPoint.inHandle.x + dx, y: selectedPoint.inHandle.y + dy };
        }
        if (selectedPoint.outHandle) {
          selectedPoint.outHandle = { x: selectedPoint.outHandle.x + dx, y: selectedPoint.outHandle.y + dy };
        }
      }
    };

    if (dragTarget.type === 'anchor') {
      const dragPointRef = { pathId: dragTarget.pathId, pointIndex: dragTarget.pointIndex };
      const pointRefs = session.selectedPoints.some((pointRef) => isSamePointRef(pointRef, dragPointRef))
        ? session.selectedPoints
        : [dragPointRef];
      moveAnchorPoints(pointRefs);
    } else if (dragTarget.type === 'anchorOrHandle') {
      const dragDistance = getDistance(dragTarget.startPointer, point);
      if (dragDistance < this.manager.stage.unscale(this.config.HANDLE_PULL_INTENT_THRESHOLD_PX)) {
        return;
      }

      if (this.getShouldPullMissingCornerHandle(bezierPoint, dragTarget.missingHandleType, point)) {
        setBezierPointHandle(bezierPoint, dragTarget.missingHandleType, point);
        nextSession = {
          ...session,
          activeHandle: dragTarget.missingHandleType,
          dragTarget: {
            pathId: dragTarget.pathId,
            pointIndex: dragTarget.pointIndex,
            type: dragTarget.missingHandleType,
          },
        };
      } else {
        moveAnchorPoints([{ pathId: dragTarget.pathId, pointIndex: dragTarget.pointIndex }]);
        nextSession = {
          ...session,
          activeHandle: null,
          dragTarget: {
            pathId: dragTarget.pathId,
            pointIndex: dragTarget.pointIndex,
            type: 'anchor',
          },
        };
      }
    } else if (dragTarget.type === 'pullHandles') {
      const handleType =
        dragTarget.handleType ?? getBezierPointPullHandleType(path.points, path.isClosed, dragTarget.pointIndex, point);
      setBezierPointHandle(bezierPoint, handleType, point);
      if (!dragTarget.handleType) {
        nextSession = {
          ...session,
          activeHandle: handleType,
          dragTarget: {
            ...dragTarget,
            handleType,
          },
        };
      }
    } else {
      setBezierPointHandle(bezierPoint, dragTarget.type, point);
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });
    if (nextSession) {
      this.$editSession.set(nextSession);
    }
    this.render();
  };

  private clearEditDragTarget = () => {
    const session = this.$editSession.get();
    if (!session || !session.dragTarget) {
      return;
    }

    this.$editSession.set({ ...session, dragTarget: null });
    this.recordEditHistory();
    this.render();
  };

  private getPointsInSelectionRect = (
    paths: CanvasBezierPathState[],
    start: Coordinate,
    end: Coordinate,
    activePathId: string | null,
    priorityRadius: number
  ): CanvasPathPointRef[] => {
    const minX = Math.min(start.x, end.x);
    const maxX = Math.max(start.x, end.x);
    const minY = Math.min(start.y, end.y);
    const maxY = Math.max(start.y, end.y);

    const activePointRefs: CanvasPathPointRef[] = [];
    const otherPointRefs: CanvasPathPointRef[] = [];
    const activeAnchors: Coordinate[] = [];
    for (const path of paths) {
      path.points.forEach((point, pointIndex) => {
        if (point.anchor.x >= minX && point.anchor.x <= maxX && point.anchor.y >= minY && point.anchor.y <= maxY) {
          const pointRef = { pathId: path.id, pointIndex };
          if (path.id === activePathId) {
            activePointRefs.push(pointRef);
            activeAnchors.push(point.anchor);
          } else {
            otherPointRefs.push(pointRef);
          }
        }
      });
    }

    const nonOverlappingPointRefs = otherPointRefs.filter((pointRef) => {
      const path = paths.find((candidate) => candidate.id === pointRef.pathId);
      const anchor = path?.points[pointRef.pointIndex]?.anchor;
      return anchor && !activeAnchors.some((activeAnchor) => getDistance(anchor, activeAnchor) <= priorityRadius);
    });

    return [...nonOverlappingPointRefs, ...activePointRefs];
  };

  private applyPointSelection = (
    initialPoints: CanvasPathPointRef[],
    hitPoints: CanvasPathPointRef[],
    mode: PointSelectionMode
  ): CanvasPathPointRef[] => {
    if (mode === 'replace') {
      return hitPoints;
    }

    const selectedPoints = new Map(initialPoints.map((pointRef) => [getPointRefKey(pointRef), pointRef]));
    for (const pointRef of hitPoints) {
      selectedPoints.set(getPointRefKey(pointRef), pointRef);
    }
    return [...selectedPoints.values()];
  };

  private findAnchorHit = (
    paths: CanvasBezierPathState[],
    point: Coordinate,
    hitRadius: number,
    activePathId: string | null
  ): { pathId: string; pointIndex: number } | null => {
    type AnchorHit = {
      pathId: string;
      pointIndex: number;
      anchor: Coordinate;
      distance: number;
      pathOrder: number;
    };

    let bestHit: AnchorHit | null = null;
    let bestActiveHit: AnchorHit | null = null;
    for (let pathOrder = 0; pathOrder < paths.length; pathOrder += 1) {
      const path = paths[pathOrder];
      if (!path) {
        continue;
      }

      for (let pointIndex = 0; pointIndex < path.points.length; pointIndex += 1) {
        const candidatePoint = path.points[pointIndex];
        if (!candidatePoint) {
          continue;
        }
        const distance = getDistance(candidatePoint.anchor, point);
        if (distance > hitRadius) {
          continue;
        }
        if (
          !bestHit ||
          distance < bestHit.distance ||
          (distance === bestHit.distance && pathOrder > bestHit.pathOrder)
        ) {
          bestHit = { pathId: path.id, pointIndex, anchor: candidatePoint.anchor, distance, pathOrder };
        }
        if (path.id === activePathId && (!bestActiveHit || distance < bestActiveHit.distance)) {
          bestActiveHit = { pathId: path.id, pointIndex, anchor: candidatePoint.anchor, distance, pathOrder };
        }
      }
    }

    if (!bestHit) {
      return null;
    }

    const hit =
      bestActiveHit && getDistance(bestActiveHit.anchor, bestHit.anchor) <= hitRadius ? bestActiveHit : bestHit;
    return { pathId: hit.pathId, pointIndex: hit.pointIndex };
  };

  private getMissingCornerHandleType = (path: CanvasBezierPathState, pointIndex: number): PathHandleType | null => {
    const point = path.points[pointIndex];
    if (!point || point.type !== 'corner') {
      return null;
    }

    const editableHandleTypes: PathHandleType[] = [];
    if (path.isClosed || pointIndex > 0) {
      editableHandleTypes.push('inHandle');
    }
    if (path.isClosed || pointIndex < path.points.length - 1) {
      editableHandleTypes.push('outHandle');
    }

    const existingHandleTypes = editableHandleTypes.filter((handleType) => point[handleType] !== null);
    const missingHandleTypes = editableHandleTypes.filter((handleType) => point[handleType] === null);
    if (existingHandleTypes.length !== 1 || missingHandleTypes.length !== 1) {
      return null;
    }
    return missingHandleTypes[0] ?? null;
  };

  private getShouldPullMissingCornerHandle = (
    point: CanvasBezierPathState['points'][number],
    missingHandleType: PathHandleType,
    pointer: Coordinate
  ): boolean => {
    const existingHandle = point[missingHandleType === 'inHandle' ? 'outHandle' : 'inHandle'];
    if (!existingHandle) {
      return true;
    }

    const existingVector = {
      x: existingHandle.x - point.anchor.x,
      y: existingHandle.y - point.anchor.y,
    };
    const dragVector = {
      x: pointer.x - point.anchor.x,
      y: pointer.y - point.anchor.y,
    };
    const existingLength = Math.hypot(existingVector.x, existingVector.y);
    const dragLength = Math.hypot(dragVector.x, dragVector.y);
    if (existingLength === 0 || dragLength === 0) {
      return true;
    }

    const directionDotProduct =
      (existingVector.x * dragVector.x + existingVector.y * dragVector.y) / (existingLength * dragLength);
    return directionDotProduct < -0.25;
  };

  private findHandleHit = (
    path: CanvasBezierPathState,
    pointIndex: number,
    point: Coordinate,
    hitRadius: number
  ): PathHandleType | null => {
    const bezierPoint = path.points[pointIndex];
    if (!bezierPoint) {
      return null;
    }

    const handles: Array<{ type: PathHandleType; point: Coordinate | null }> = [
      { type: 'inHandle', point: bezierPoint.inHandle },
      { type: 'outHandle', point: bezierPoint.outHandle },
    ];

    let bestHit: { type: PathHandleType; distance: number } | null = null;
    for (const handle of handles) {
      if (!handle.point) {
        continue;
      }
      const distance = getDistance(handle.point, point);
      if (distance > hitRadius) {
        continue;
      }
      if (!bestHit || distance < bestHit.distance) {
        bestHit = { type: handle.type, distance };
      }
    }

    return bestHit?.type ?? null;
  };

  private findPathHit = (
    paths: CanvasBezierPathState[],
    point: Coordinate,
    hitTolerance: number,
    samplesPerSegment: number
  ): { pathId: string } | null => {
    let bestHit: { pathId: string; distance: number; pathOrder: number } | null = null;
    for (let pathOrder = 0; pathOrder < paths.length; pathOrder += 1) {
      const path = paths[pathOrder];
      if (!path) {
        continue;
      }

      const hit = findNearestBezierPathSegment(path.points, path.isClosed, point, samplesPerSegment);
      if (!hit || hit.distance > hitTolerance) {
        continue;
      }
      if (
        !bestHit ||
        hit.distance < bestHit.distance ||
        (hit.distance === bestHit.distance && pathOrder > bestHit.pathOrder)
      ) {
        bestHit = { pathId: path.id, distance: hit.distance, pathOrder };
      }
    }

    return bestHit ? { pathId: bestHit.pathId } : null;
  };

  deleteSelectedPointsOrActivePath = () => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return;
    }
    if (session.selectedPoints.length === 0) {
      this.deleteActivePath();
      return;
    }

    const activePathIndex = activeEntity.state.paths.findIndex((path) => path.id === session.activePathId);
    const { paths: nextPaths, didDelete } = deleteVectorPathPoints(activeEntity.state.paths, session.selectedPoints);

    if (!didDelete) {
      return;
    }
    const nextActivePath =
      nextPaths.find((path) => path.id === session.activePathId) ??
      nextPaths[Math.min(Math.max(activePathIndex, 0), nextPaths.length - 1)] ??
      null;
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });

    this.$editSession.set({
      ...session,
      activePathId: nextActivePath?.id ?? null,
      activePointIndex: null,
      selectedPoints: [],
      activeHandle: null,
      dragTarget: null,
    });
    this.recordEditHistory();
    this.render();
  };

  private insertPoint = (pathId: string, segmentIndex: number, t: number): number | null => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return null;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const path = nextPaths.find((candidate) => candidate.id === pathId);
    if (!path || path.points.length < 2) {
      return null;
    }

    const from = path.points[segmentIndex];
    const to = path.points[(segmentIndex + 1) % path.points.length];
    if (!from || !to) {
      return null;
    }

    const split = splitBezierSegmentAt(from, to, t);
    from.outHandle = split.fromOutHandle;
    to.inHandle = split.toInHandle;

    const insertIndex = segmentIndex + 1;
    path.points.splice(insertIndex, 0, split.insertPoint);

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
      undoGroup: session.id,
    });

    return insertIndex;
  };
}
