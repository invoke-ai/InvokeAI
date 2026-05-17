import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { addCoords, floorCoord, getPrefixedId, offsetCoord } from 'features/controlLayers/konva/util';
import type { CanvasBezierPathState, CanvasEntityIdentifier, Coordinate } from 'features/controlLayers/store/types';
import { getBezierPathState } from 'features/controlLayers/store/util';
import {
  anchorsToBezierPoints,
  buildBezierPathData,
  findNearestBezierPathSegment,
  splitBezierSegmentAt,
} from 'features/controlLayers/util/bezierPath';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasPathToolModuleConfig = {
  ANCHOR_RADIUS_PX: number;
  ANCHOR_STROKE_WIDTH_PX: number;
  HANDLE_RADIUS_PX: number;
  HANDLE_STROKE_WIDTH_PX: number;
  HANDLE_LINE_WIDTH_PX: number;
  PATH_HIT_TOLERANCE_PX: number;
  PREVIEW_STROKE_COLOR: string;
  PREVIEW_STROKE_WIDTH_PX: number;
  PREVIEW_POINT_FILL: string;
  PREVIEW_POINT_STROKE: string;
  EDIT_PATH_STROKE_COLOR: string;
  EDIT_ACTIVE_POINT_FILL: string;
  EDIT_ACTIVE_POINT_STROKE: string;
  EDIT_INACTIVE_POINT_FILL: string;
  EDIT_INACTIVE_POINT_STROKE: string;
  EDIT_HANDLE_FILL: string;
  EDIT_HANDLE_STROKE: string;
  EDIT_HANDLE_LINE: string;
};

type PathHandleType = 'inHandle' | 'outHandle';

type CanvasPathEditDragTarget =
  | { pathId: string; pointIndex: number; type: 'anchor' }
  | { pathId: string; pointIndex: number; type: 'pullHandles' }
  | { pathId: string; pointIndex: number; type: PathHandleType };

type CanvasPathEditSession = {
  entityIdentifier: CanvasEntityIdentifier<'vector_layer'>;
  snapshotPaths: CanvasBezierPathState[];
  activePathId: string | null;
  activePointIndex: number | null;
  activeHandle: PathHandleType | null;
  dragTarget: CanvasPathEditDragTarget | null;
};

const DEFAULT_CONFIG: CanvasPathToolModuleConfig = {
  ANCHOR_RADIUS_PX: 4,
  ANCHOR_STROKE_WIDTH_PX: 2,
  HANDLE_RADIUS_PX: 3.5,
  HANDLE_STROKE_WIDTH_PX: 1.5,
  HANDLE_LINE_WIDTH_PX: 1,
  PATH_HIT_TOLERANCE_PX: 10,
  PREVIEW_STROKE_COLOR: 'rgba(90, 175, 255, 1)',
  PREVIEW_STROKE_WIDTH_PX: 1.5,
  PREVIEW_POINT_FILL: 'rgba(255, 255, 255, 1)',
  PREVIEW_POINT_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_PATH_STROKE_COLOR: 'rgba(90, 175, 255, 1)',
  EDIT_ACTIVE_POINT_FILL: 'rgba(90, 175, 255, 1)',
  EDIT_ACTIVE_POINT_STROKE: 'rgba(255, 255, 255, 1)',
  EDIT_INACTIVE_POINT_FILL: 'rgba(255, 255, 255, 0.95)',
  EDIT_INACTIVE_POINT_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_HANDLE_FILL: 'rgba(255, 255, 255, 1)',
  EDIT_HANDLE_STROKE: 'rgba(90, 175, 255, 1)',
  EDIT_HANDLE_LINE: 'rgba(90, 175, 255, 0.75)',
};

const getDistance = (a: Coordinate, b: Coordinate) => Math.hypot(a.x - b.x, a.y - b.y);
const normalizeHandle = (anchor: Coordinate, handle: Coordinate): Coordinate | null =>
  anchor.x === handle.x && anchor.y === handle.y ? null : handle;
const mirrorHandle = (anchor: Coordinate, handle: Coordinate): Coordinate => ({
  x: anchor.x + (anchor.x - handle.x),
  y: anchor.y + (anchor.y - handle.y),
});

export class CanvasPathToolModule extends CanvasModuleBase {
  readonly type = 'path_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasPathToolModuleConfig = DEFAULT_CONFIG;

  $editSession = atom<CanvasPathEditSession | null>(null);

  private activeEntityIdentifier: CanvasEntityIdentifier<'vector_layer'> | null = null;
  private points: Coordinate[] = [];
  private previewPoint: Coordinate | null = null;

  konva: {
    group: Konva.Group;
    previewPath: Konva.Path;
    previewAnchorsGroup: Konva.Group;
    editPath: Konva.Path;
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

  hasActiveEditDragSession = (): boolean => {
    return Boolean(this.$editSession.get()?.dragTarget);
  };

  hasActiveSession = (): boolean => {
    return this.hasActiveCreateSession() || this.hasActiveEditSession();
  };

  startEdit = (entityIdentifier: CanvasEntityIdentifier<'vector_layer'>) => {
    const adapter = this.manager.getAdapter(entityIdentifier);
    if (!adapter || adapter.state.type !== 'vector_layer' || adapter.state.paths.length === 0) {
      return;
    }

    const existingSession = this.$editSession.get();
    if (
      existingSession &&
      existingSession.entityIdentifier.id === entityIdentifier.id &&
      existingSession.entityIdentifier.type === entityIdentifier.type
    ) {
      this.parent.$tool.set('path');
      this.render();
      return;
    }

    if (existingSession) {
      this.acceptEditSession();
    }

    this.resetCreateState();
    this.$editSession.set({
      entityIdentifier,
      snapshotPaths: deepClone(adapter.state.paths),
      activePathId: adapter.state.paths[0]?.id ?? null,
      activePointIndex: null,
      activeHandle: null,
      dragTarget: null,
    });
    this.parent.$toolBuffer.set(null);
    this.parent.$tool.set('path');
    this.render();
  };

  acceptEditSession = () => {
    this.$editSession.set(null);
    this.render();
  };

  onToolChanged = () => {
    const tool = this.parent.$tool.get();
    if (tool !== 'path' && !this.isTemporaryToolSwitch(tool, this.parent.$toolBuffer.get())) {
      if (this.hasActiveEditSession()) {
        this.acceptEditSession();
      }
      this.resetCreateState();
    }
  };

  syncCursorStyle = () => {
    this.manager.stage.setCursor(this.parent.getCanDraw() ? 'crosshair' : 'not-allowed');
  };

  render = () => {
    const tool = this.parent.$tool.get();
    const isTemporaryToolSwitch = this.isTemporaryToolSwitch(tool, this.parent.$toolBuffer.get());

    if (tool !== 'path' && !isTemporaryToolSwitch) {
      this.hideCreatePreview();
      this.hideEditOverlay();
      return;
    }

    if (tool === 'path') {
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
      this.cancelEditSession();
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
        isClosed: false,
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
    this.previewPoint = this.getPathPoint(point, e.evt.shiftKey);
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
    if (!session) {
      return null;
    }

    const adapter = this.manager.getAdapter(session.entityIdentifier);
    if (!adapter || adapter.state.type !== 'vector_layer') {
      return null;
    }

    return adapter;
  };

  private cancelEditSession = () => {
    const session = this.$editSession.get();
    if (!session) {
      return;
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: deepClone(session.snapshotPaths),
    });
    this.$editSession.set(null);
    this.render();
  };

  private getEntityRelativePoint = (point: Coordinate, position: Coordinate): Coordinate => {
    return floorCoord(offsetCoord(point, position));
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

  private syncCreatePreview = () => {
    const activeEntity = this.getCreateEntityAdapter();
    if (!activeEntity || this.points.length === 0) {
      this.hideCreatePreview();
      return;
    }

    const entityPosition = activeEntity.state.position;
    const previewAnchors = this.previewPoint ? [...this.points, this.previewPoint] : [...this.points];
    const previewPoints = previewAnchors.map((point) => addCoords(point, entityPosition));
    const data = buildBezierPathData(anchorsToBezierPoints(previewPoints), false);
    const anchorRadius = this.manager.stage.unscale(this.config.ANCHOR_RADIUS_PX);
    const anchorStrokeWidth = this.manager.stage.unscale(this.config.ANCHOR_STROKE_WIDTH_PX);

    this.konva.previewPath.setAttrs({
      data,
      visible: Boolean(data),
      strokeWidth: this.manager.stage.unscale(this.config.PREVIEW_STROKE_WIDTH_PX),
    });

    this.konva.previewAnchorsGroup.destroyChildren();
    for (const point of this.points.map((anchor) => addCoords(anchor, entityPosition))) {
      this.konva.previewAnchorsGroup.add(
        new Konva.Circle({
          x: point.x,
          y: point.y,
          radius: anchorRadius,
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

    this.konva.editAnchorsGroup.destroyChildren();
    activePath.points.forEach((point, pointIndex) => {
      const isActivePoint = pointIndex === session.activePointIndex;
      const stagePoint = addCoords(point.anchor, entityPosition);
      this.konva.editAnchorsGroup.add(
        new Konva.Circle({
          x: stagePoint.x,
          y: stagePoint.y,
          radius: anchorRadius,
          fill: isActivePoint ? this.config.EDIT_ACTIVE_POINT_FILL : this.config.EDIT_INACTIVE_POINT_FILL,
          stroke: isActivePoint ? this.config.EDIT_ACTIVE_POINT_STROKE : this.config.EDIT_INACTIVE_POINT_STROKE,
          strokeWidth: anchorStrokeWidth,
          listening: false,
          perfectDrawEnabled: false,
        })
      );
    });
    this.konva.editAnchorsGroup.visible(activePath.points.length > 0);

    this.konva.editHandlesGroup.destroyChildren();
    const activePoint = session.activePointIndex !== null ? activePath.points[session.activePointIndex] : null;
    if (activePoint) {
      const anchor = addCoords(activePoint.anchor, entityPosition);
      const handles: Array<{ type: PathHandleType; point: Coordinate | null }> = [
        { type: 'inHandle', point: activePoint.inHandle ? addCoords(activePoint.inHandle, entityPosition) : null },
        { type: 'outHandle', point: activePoint.outHandle ? addCoords(activePoint.outHandle, entityPosition) : null },
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
    }
    this.konva.editHandlesGroup.visible(Boolean(activePoint));
  };

  private hideCreatePreview = () => {
    this.konva.previewPath.visible(false);
    this.konva.previewAnchorsGroup.visible(false);
    this.konva.previewAnchorsGroup.destroyChildren();
  };

  private hideEditOverlay = () => {
    this.konva.editPath.visible(false);
    this.konva.editAnchorsGroup.visible(false);
    this.konva.editAnchorsGroup.destroyChildren();
    this.konva.editHandlesGroup.visible(false);
    this.konva.editHandlesGroup.destroyChildren();
  };

  private isTemporaryToolSwitch = (tool: string, toolBuffer: string | null) => {
    return toolBuffer === 'path' && (tool === 'view' || tool === 'colorPicker' || tool === 'path');
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
    const activePath =
      activeEntity.state.paths.find((path) => path.id === session.activePathId) ?? activeEntity.state.paths[0] ?? null;

    if (activePath && session.activePointIndex !== null) {
      const handleHit = this.findHandleHit(activePath, session.activePointIndex, point, handleHitRadius);
      if (handleHit) {
        this.$editSession.set({
          ...session,
          activePathId: activePath.id,
          activePointIndex: session.activePointIndex,
          activeHandle: handleHit,
          dragTarget: {
            pathId: activePath.id,
            pointIndex: session.activePointIndex,
            type: handleHit,
          },
        });
        this.render();
        return;
      }
    }

    const anchorHit = this.findAnchorHit(activeEntity.state.paths, point, anchorHitRadius);
    if (anchorHit) {
      if (e.evt.ctrlKey) {
        this.deletePoint(anchorHit.pathId, anchorHit.pointIndex);
        return;
      }

      const clickedPath = activeEntity.state.paths.find((path) => path.id === anchorHit.pathId);
      const clickedPoint = clickedPath?.points[anchorHit.pointIndex];
      const shouldPullHandles =
        Boolean(clickedPoint) &&
        session.activePathId === anchorHit.pathId &&
        session.activePointIndex === anchorHit.pointIndex &&
        !clickedPoint?.inHandle &&
        !clickedPoint?.outHandle;

      this.$editSession.set({
        ...session,
        activePathId: anchorHit.pathId,
        activePointIndex: anchorHit.pointIndex,
        activeHandle: null,
        dragTarget: {
          pathId: anchorHit.pathId,
          pointIndex: anchorHit.pointIndex,
          type: shouldPullHandles ? 'pullHandles' : 'anchor',
        },
      });
      this.render();
      return;
    }

    if (e.evt.shiftKey && activePath) {
      const segmentHit = findNearestBezierPathSegment(activePath.points, activePath.isClosed, point);
      if (segmentHit && segmentHit.distance <= pathHitTolerance) {
        const insertedPointIndex = this.insertPoint(activePath.id, segmentHit.segmentIndex, segmentHit.t);
        if (insertedPointIndex !== null) {
          this.$editSession.set({
            ...session,
            activePathId: activePath.id,
            activePointIndex: insertedPointIndex,
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

    const pathHit = this.findPathHit(activeEntity.state.paths, point, pathHitTolerance);
    if (pathHit) {
      this.$editSession.set({
        ...session,
        activePathId: pathHit.pathId,
        activePointIndex: null,
        activeHandle: null,
        dragTarget: null,
      });
      this.render();
      return;
    }

    this.$editSession.set({
      ...session,
      activePointIndex: null,
      activeHandle: null,
      dragTarget: null,
    });
    this.render();
  };

  private onEditPointerMove = (evt: PointerEvent) => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!session || !activeEntity || !session.dragTarget || !cursorPos) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
    const nextPaths = deepClone(activeEntity.state.paths);
    const path = nextPaths.find((candidate) => candidate.id === session.dragTarget?.pathId);
    const bezierPoint = path?.points[session.dragTarget.pointIndex];
    if (!path || !bezierPoint) {
      return;
    }

    if (session.dragTarget.type === 'anchor') {
      const dx = point.x - bezierPoint.anchor.x;
      const dy = point.y - bezierPoint.anchor.y;
      bezierPoint.anchor = point;
      if (bezierPoint.inHandle) {
        bezierPoint.inHandle = { x: bezierPoint.inHandle.x + dx, y: bezierPoint.inHandle.y + dy };
      }
      if (bezierPoint.outHandle) {
        bezierPoint.outHandle = { x: bezierPoint.outHandle.x + dx, y: bezierPoint.outHandle.y + dy };
      }
    } else if (session.dragTarget.type === 'pullHandles') {
      const outHandle = normalizeHandle(bezierPoint.anchor, point);
      bezierPoint.type = 'smooth';
      bezierPoint.outHandle = outHandle;
      bezierPoint.inHandle = outHandle
        ? normalizeHandle(bezierPoint.anchor, mirrorHandle(bezierPoint.anchor, outHandle))
        : null;
    } else {
      bezierPoint[session.dragTarget.type] = normalizeHandle(bezierPoint.anchor, point);
      if (bezierPoint.type === 'smooth') {
        const oppositeHandleType = session.dragTarget.type === 'inHandle' ? 'outHandle' : 'inHandle';
        const handle = bezierPoint[session.dragTarget.type];
        bezierPoint[oppositeHandleType] = handle
          ? normalizeHandle(bezierPoint.anchor, mirrorHandle(bezierPoint.anchor, handle))
          : null;
      }
    }

    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
    });

    if (evt.shiftKey && session.dragTarget.type === 'anchor') {
      this.render();
    }
  };

  private clearEditDragTarget = () => {
    const session = this.$editSession.get();
    if (!session || !session.dragTarget) {
      return;
    }

    this.$editSession.set({ ...session, dragTarget: null });
    this.render();
  };

  private findAnchorHit = (
    paths: CanvasBezierPathState[],
    point: Coordinate,
    hitRadius: number
  ): { pathId: string; pointIndex: number } | null => {
    let bestHit: { pathId: string; pointIndex: number; distance: number; pathOrder: number } | null = null;
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
          bestHit = { pathId: path.id, pointIndex, distance, pathOrder };
        }
      }
    }

    if (!bestHit) {
      return null;
    }

    const hit = bestHit;
    return { pathId: hit.pathId, pointIndex: hit.pointIndex };
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
    hitTolerance: number
  ): { pathId: string } | null => {
    let bestHit: { pathId: string; distance: number; pathOrder: number } | null = null;
    for (let pathOrder = 0; pathOrder < paths.length; pathOrder += 1) {
      const path = paths[pathOrder];
      if (!path) {
        continue;
      }

      const hit = findNearestBezierPathSegment(path.points, path.isClosed, point);
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

  private deletePoint = (pathId: string, pointIndex: number) => {
    const session = this.$editSession.get();
    const activeEntity = this.getEditSessionAdapter();
    if (!session || !activeEntity) {
      return;
    }

    const nextPaths = deepClone(activeEntity.state.paths);
    const path = nextPaths.find((candidate) => candidate.id === pathId);
    if (!path) {
      return;
    }

    const minPointCount = path.isClosed ? 3 : 2;
    if (path.points.length <= minPointCount) {
      this.$editSession.set({
        ...session,
        activePathId: pathId,
        activePointIndex: pointIndex,
        activeHandle: null,
        dragTarget: null,
      });
      this.render();
      return;
    }

    path.points.splice(pointIndex, 1);
    this.manager.stateApi.replaceVectorPaths({
      entityIdentifier: session.entityIdentifier,
      paths: nextPaths,
    });

    const nextActivePointIndex = path.points.length === 0 ? null : Math.min(pointIndex, path.points.length - 1);
    this.$editSession.set({
      ...session,
      activePathId: pathId,
      activePointIndex: nextActivePointIndex,
      activeHandle: null,
      dragTarget: null,
    });
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
    });

    return insertIndex;
  };
}
