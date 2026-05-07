import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { shouldPreserveSuspendableShapesSession } from 'features/controlLayers/konva/CanvasTool/toolHotkeys';
import {
  addCoords,
  floorCoord,
  getPrefixedId,
  isDistanceMoreThanMin,
  offsetCoord,
} from 'features/controlLayers/konva/util';
import { selectShapeType } from 'features/controlLayers/store/canvasSettingsSlice';
import type {
  CanvasEntityIdentifier,
  CanvasPolygonState,
  CanvasRectState,
  Coordinate,
} from 'features/controlLayers/store/types';
import { simplifyFlatNumbersArray } from 'features/controlLayers/util/simplify';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Logger } from 'roarr';

type CanvasShapeToolModuleConfig = {
  START_POINT_RADIUS_PX: number;
  START_POINT_STROKE_WIDTH_PX: number;
  START_POINT_HOVER_RADIUS_DELTA_PX: number;
  POLYGON_CLOSE_RADIUS_PX: number;
  MIN_FREEHAND_POINT_DISTANCE_PX: number;
  MAX_FREEHAND_SEGMENT_LENGTH_PX: number;
  FREEHAND_SIMPLIFY_MIN_POINTS: number;
  FREEHAND_SIMPLIFY_TOLERANCE: number;
  PREVIEW_STROKE_COLOR: string;
};

const DEFAULT_CONFIG: CanvasShapeToolModuleConfig = {
  START_POINT_RADIUS_PX: 4,
  START_POINT_STROKE_WIDTH_PX: 2,
  START_POINT_HOVER_RADIUS_DELTA_PX: 2,
  POLYGON_CLOSE_RADIUS_PX: 10,
  MIN_FREEHAND_POINT_DISTANCE_PX: 1,
  MAX_FREEHAND_SEGMENT_LENGTH_PX: 2,
  FREEHAND_SIMPLIFY_MIN_POINTS: 200,
  FREEHAND_SIMPLIFY_TOLERANCE: 0.6,
  PREVIEW_STROKE_COLOR: rgbaColorToString({ r: 90, g: 175, b: 255, a: 1 }),
};

const SUBTRACT_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(
  `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>
    <g fill='none' stroke-linecap='round'>
      <path d='M12 3v5M12 16v5M3 12h5M16 12h5' stroke='white' stroke-width='3'/>
      <path d='M12 3v5M12 16v5M3 12h5M16 12h5' stroke='black' stroke-width='1.5'/>
      <circle cx='6' cy='18' r='4.5' fill='white' stroke='black' stroke-width='1'/>
      <path d='M4 18h4' stroke='black' stroke-width='1.5'/>
    </g>
  </svg>`
)}") 12 12, crosshair`;

const getAxisSign = (value: number, fallback: number): number => {
  if (value === 0) {
    return fallback === 0 ? 1 : Math.sign(fallback);
  }
  return Math.sign(value);
};

export class CanvasShapeToolModule extends CanvasModuleBase {
  readonly type = 'shape_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasShapeToolModuleConfig = DEFAULT_CONFIG;
  subscriptions: Set<() => void> = new Set();

  private activeEntityIdentifier: CanvasEntityIdentifier | null = null;
  private shapeId: string | null = null;
  private dragStartPoint: Coordinate | null = null;
  private dragCurrentPoint: Coordinate | null = null;
  private translatePreviousPointerPoint: Coordinate | null = null;
  private freehandPoints: Coordinate[] = [];
  private isDrawingFreehand = false;
  private polygonPoints: Coordinate[] = [];
  private polygonPointer: Coordinate | null = null;

  konva: {
    group: Konva.Group;
    startPointIndicator: Konva.Circle;
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
      startPointIndicator: new Konva.Circle({
        name: `${this.type}:start_point_indicator`,
        listening: false,
        fillEnabled: false,
        stroke: this.config.PREVIEW_STROKE_COLOR,
        visible: false,
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.startPointIndicator);

    this.subscriptions.add(this.manager.stateApi.$altKey.listen(this.onModifierChanged));
    this.subscriptions.add(this.manager.stateApi.$ctrlKey.listen(this.onModifierChanged));
    this.subscriptions.add(this.manager.stateApi.$metaKey.listen(this.onModifierChanged));
    this.subscriptions.add(this.manager.stateApi.$shiftKey.listen(this.onModifierChanged));
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectShapeType, () => {
        if (this.hasActiveSession()) {
          this.cancel();
        }
        this.render();
      })
    );
  }

  hasActiveSession = (): boolean => {
    return Boolean(
      this.dragStartPoint || this.isDrawingFreehand || this.freehandPoints.length || this.polygonPoints.length
    );
  };

  hasSuspendableSession = (): boolean => {
    return Boolean(this.isDrawingFreehand || this.freehandPoints.length || this.polygonPoints.length);
  };

  hasActiveDragSession = (): boolean => {
    return Boolean(this.dragStartPoint || this.isDrawingFreehand);
  };

  hasActiveRectOvalDragSession = (): boolean => {
    const shapeType = this.manager.stateApi.getSettings().shapeType;
    return Boolean(this.dragStartPoint && this.dragCurrentPoint && (shapeType === 'rect' || shapeType === 'oval'));
  };

  hasActivePolygonSession = (): boolean => {
    return this.polygonPoints.length > 0;
  };

  isTranslatingDragSession = (): boolean => {
    return this.translatePreviousPointerPoint !== null;
  };

  freezePolygonPreview = async () => {
    if (!this.hasActivePolygonSession()) {
      return;
    }

    const activeEntity = this.getActiveEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();
    if (!activeEntity || !cursorPos) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
    this.polygonPointer = point;
    await this.updatePolygonBuffer();
    this.render();
  };

  onToolChanged = () => {
    const tool = this.parent.$tool.get();
    const isTemporaryToolSwitch = shouldPreserveSuspendableShapesSession(
      tool,
      this.parent.$toolBuffer.get(),
      this.hasSuspendableSession()
    );
    if (tool !== 'rect' && !isTemporaryToolSwitch) {
      this.cancel();
    }
  };

  syncCursorStyle = () => {
    this.manager.stage.setCursor(this.getCompositeOperation() === 'destination-out' ? SUBTRACT_CURSOR : 'crosshair');
  };

  render = () => {
    const tool = this.parent.$tool.get();
    const isTemporaryToolSwitch = shouldPreserveSuspendableShapesSession(
      tool,
      this.parent.$toolBuffer.get(),
      this.hasSuspendableSession()
    );
    if (tool !== 'rect' && !isTemporaryToolSwitch) {
      this.konva.startPointIndicator.visible(false);
      return;
    }

    if (tool === 'rect') {
      this.syncCursorStyle();
    }

    this.syncStartPointIndicator();
  };

  cancel = () => {
    this.clearActiveBuffer();
    this.resetState();
    this.render();
  };

  startDragTranslation = () => {
    const activeEntity = this.getActiveEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();
    if (!activeEntity || !cursorPos || !this.hasActiveRectOvalDragSession()) {
      return;
    }

    this.translatePreviousPointerPoint = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);
  };

  stopDragTranslation = () => {
    this.translatePreviousPointerPoint = null;
  };

  onStagePointerDown = async (e: KonvaEventObject<PointerEvent>) => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();
    if (!selectedEntity || !cursorPos) {
      return;
    }

    if (e.evt.button !== 0) {
      return;
    }

    const shapeType = this.manager.stateApi.getSettings().shapeType;
    const point = this.getEntityRelativePoint(cursorPos.relative, selectedEntity.state.position);

    if (shapeType === 'polygon') {
      await this.onPolygonPointerDown(point, selectedEntity.entityIdentifier, e.evt.shiftKey);
      return;
    }

    if (shapeType === 'freehand') {
      if (!this.parent.$isPrimaryPointerDown.get()) {
        return;
      }

      await this.startFreehandSession(point, selectedEntity.entityIdentifier);
      return;
    }

    if (!this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    this.clearActiveBuffer();
    this.resetState();
    this.activeEntityIdentifier = selectedEntity.entityIdentifier;
    this.shapeId = getPrefixedId(shapeType);
    this.dragStartPoint = point;
    this.dragCurrentPoint = point;
    await this.updateDragBuffer();
  };

  onStagePointerMove = async (e: KonvaEventObject<PointerEvent>) => {
    const shapeType = this.manager.stateApi.getSettings().shapeType;
    const activeEntity = this.getActiveEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!activeEntity || !cursorPos) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);

    if (shapeType === 'polygon') {
      if (!this.hasActivePolygonSession()) {
        return;
      }
      this.polygonPointer = this.getPolygonPoint(point, e.evt.shiftKey);
      await this.updatePolygonBuffer();
      this.render();
      return;
    }

    if (shapeType === 'freehand') {
      await this.handleFreehandPointerMove(point);
      return;
    }

    if (!this.parent.$isPrimaryPointerDown.get() || !this.dragStartPoint) {
      return;
    }

    if (this.isTranslatingDragSession()) {
      await this.translateDragShape(point);
      return;
    }

    this.dragCurrentPoint = point;
    await this.updateDragBuffer();
  };

  onWindowPointerMove = async () => {
    const shapeType = this.manager.stateApi.getSettings().shapeType;
    const activeEntity = this.getActiveEntityAdapter();
    const cursorPos = this.parent.$cursorPos.get();

    if (!activeEntity || !cursorPos || !this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    const point = this.getEntityRelativePoint(cursorPos.relative, activeEntity.state.position);

    if (shapeType === 'freehand') {
      await this.handleFreehandPointerMove(point);
      return;
    }

    if ((shapeType !== 'rect' && shapeType !== 'oval') || !this.dragStartPoint) {
      return;
    }

    if (this.isTranslatingDragSession()) {
      await this.translateDragShape(point);
      return;
    }

    this.dragCurrentPoint = point;
    await this.updateDragBuffer();
  };

  onStagePointerUp = async (_e: KonvaEventObject<PointerEvent>) => {
    const shapeType = this.manager.stateApi.getSettings().shapeType;

    if (shapeType === 'polygon') {
      this.render();
      return;
    }

    if (shapeType === 'freehand') {
      await this.commitFreehand();
      return;
    }

    this.finishDragShapeSession();
  };

  onWindowPointerUp = async () => {
    if (this.isDrawingFreehand) {
      await this.commitFreehand();
      return;
    }

    if (!this.dragStartPoint) {
      return;
    }

    this.finishDragShapeSession();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      activeEntityIdentifier: this.activeEntityIdentifier,
      shapeId: this.shapeId,
      dragStartPoint: this.dragStartPoint,
      dragCurrentPoint: this.dragCurrentPoint,
      translatePreviousPointerPoint: this.translatePreviousPointerPoint,
      freehandPoints: this.freehandPoints,
      isDrawingFreehand: this.isDrawingFreehand,
      polygonPoints: this.polygonPoints,
      polygonPointer: this.polygonPointer,
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };

  private onModifierChanged = () => {
    const tool = this.parent.$tool.get();
    const isTemporaryToolSwitch = shouldPreserveSuspendableShapesSession(
      tool,
      this.parent.$toolBuffer.get(),
      this.hasSuspendableSession()
    );
    if (tool !== 'rect' && !isTemporaryToolSwitch) {
      return;
    }

    if (tool === 'rect') {
      this.syncCursorStyle();
    }
    void this.updateActivePreview();
    this.render();
  };

  private updateActivePreview = async () => {
    if (this.dragStartPoint) {
      await this.updateDragBuffer();
      return;
    }

    if (this.isDrawingFreehand || this.freehandPoints.length > 0) {
      await this.updateFreehandBuffer();
      return;
    }

    if (this.hasActivePolygonSession()) {
      await this.updatePolygonBuffer();
    }
  };

  private startFreehandSession = async (point: Coordinate, entityIdentifier: CanvasEntityIdentifier) => {
    this.clearActiveBuffer();
    this.resetState();
    this.activeEntityIdentifier = entityIdentifier;
    this.shapeId = getPrefixedId('polygon');
    this.freehandPoints = [point];
    this.isDrawingFreehand = true;
    await this.updateFreehandBuffer();
  };

  private handleFreehandPointerMove = async (point: Coordinate) => {
    if (!this.isDrawingFreehand || !this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    const minDistance = this.manager.stage.unscale(this.config.MIN_FREEHAND_POINT_DISTANCE_PX);
    const lastPoint = this.freehandPoints.at(-1) ?? null;
    if (!isDistanceMoreThanMin(point, lastPoint, minDistance)) {
      return;
    }

    this.appendFreehandPoint(point);
    await this.updateFreehandBuffer();
  };

  private translateDragShape = async (point: Coordinate) => {
    if (!this.translatePreviousPointerPoint || !this.dragStartPoint || !this.dragCurrentPoint) {
      return;
    }

    const dx = point.x - this.translatePreviousPointerPoint.x;
    const dy = point.y - this.translatePreviousPointerPoint.y;

    if (dx === 0 && dy === 0) {
      return;
    }

    this.dragStartPoint = {
      x: this.dragStartPoint.x + dx,
      y: this.dragStartPoint.y + dy,
    };
    this.dragCurrentPoint = {
      x: this.dragCurrentPoint.x + dx,
      y: this.dragCurrentPoint.y + dy,
    };
    this.translatePreviousPointerPoint = point;

    await this.updateDragBuffer();
  };

  private onPolygonPointerDown = async (
    point: Coordinate,
    entityIdentifier: CanvasEntityIdentifier,
    shouldSnap: boolean
  ) => {
    if (
      this.activeEntityIdentifier &&
      (this.activeEntityIdentifier.id !== entityIdentifier.id ||
        this.activeEntityIdentifier.type !== entityIdentifier.type)
    ) {
      this.cancel();
    }

    this.activeEntityIdentifier = entityIdentifier;
    this.dragStartPoint = null;
    this.dragCurrentPoint = null;

    if (this.polygonPoints.length === 0) {
      this.shapeId = getPrefixedId('polygon');
      this.polygonPoints = [point];
      this.polygonPointer = point;
      await this.updatePolygonBuffer();
      this.render();
      return;
    }

    const startPoint = this.polygonPoints[0];
    if (!startPoint) {
      return;
    }

    if (this.polygonPoints.length >= 3 && this.isPointNearStart(point)) {
      await this.commitPolygon();
      return;
    }

    const polygonPoint = this.getPolygonPoint(point, shouldSnap);
    this.polygonPoints = [...this.polygonPoints, polygonPoint];
    this.polygonPointer = polygonPoint;
    await this.updatePolygonBuffer();
    this.render();
  };

  private commitPolygon = async () => {
    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity || !this.shapeId || this.polygonPoints.length < 3) {
      this.cancel();
      return;
    }

    const polygonState: CanvasPolygonState = {
      id: this.shapeId,
      type: 'polygon',
      points: this.polygonPoints.flatMap((point) => [point.x, point.y]),
      color: this.manager.stateApi.getCurrentColor(),
      compositeOperation: this.getCompositeOperation(),
    };

    await activeEntity.bufferRenderer.setBuffer(polygonState);
    activeEntity.bufferRenderer.commitBuffer();
    this.resetState();
    this.render();
  };

  private commitFreehand = async () => {
    if (!this.isDrawingFreehand) {
      return;
    }

    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity || !this.shapeId) {
      this.cancel();
      return;
    }

    const simplifiedPoints = this.simplifyFreehandContour(this.freehandPoints);
    if (simplifiedPoints.length < 3) {
      activeEntity.bufferRenderer.clearBuffer();
      this.resetState();
      this.render();
      return;
    }

    const polygonState: CanvasPolygonState = {
      id: this.shapeId,
      type: 'polygon',
      points: simplifiedPoints.flatMap((point) => [point.x, point.y]),
      color: this.manager.stateApi.getCurrentColor(),
      compositeOperation: this.getCompositeOperation(),
    };

    await activeEntity.bufferRenderer.setBuffer(polygonState);
    activeEntity.bufferRenderer.commitBuffer();
    this.resetState();
    this.render();
  };

  private updateDragBuffer = async () => {
    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity || !this.dragStartPoint || !this.dragCurrentPoint || !this.shapeId) {
      return;
    }

    const shapeType = this.manager.stateApi.getSettings().shapeType;
    if (shapeType !== 'rect' && shapeType !== 'oval') {
      return;
    }

    const rect = this.getDragRect(this.dragStartPoint, this.dragCurrentPoint, {
      fromCenter: this.manager.stateApi.$altKey.get(),
      constrainSquare: this.manager.stateApi.$shiftKey.get(),
    });

    await activeEntity.bufferRenderer.setBuffer({
      id: this.shapeId,
      type: shapeType,
      rect,
      color: this.manager.stateApi.getCurrentColor(),
      compositeOperation: this.getCompositeOperation(),
    });
  };

  private updatePolygonBuffer = async () => {
    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity || !this.shapeId || this.polygonPoints.length === 0) {
      return;
    }

    await activeEntity.bufferRenderer.setBuffer({
      id: this.shapeId,
      type: 'polygon',
      points: this.polygonPoints.flatMap((point) => [point.x, point.y]),
      previewPoint: this.polygonPointer ?? this.polygonPoints.at(-1),
      color: this.manager.stateApi.getCurrentColor(),
      compositeOperation: this.getCompositeOperation(),
    });
  };

  private updateFreehandBuffer = async () => {
    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity || !this.shapeId || this.freehandPoints.length === 0) {
      return;
    }

    await activeEntity.bufferRenderer.setBuffer({
      id: this.shapeId,
      type: 'polygon',
      points: this.freehandPoints.flatMap((point) => [point.x, point.y]),
      color: this.manager.stateApi.getCurrentColor(),
      compositeOperation: this.getCompositeOperation(),
    });
  };

  private syncStartPointIndicator = () => {
    const activeEntity = this.getActiveEntityAdapter();
    const startPoint = this.polygonPoints[0];
    if (!activeEntity || !startPoint || this.manager.stateApi.getSettings().shapeType !== 'polygon') {
      this.konva.startPointIndicator.visible(false);
      return;
    }

    const isHoveringStartPoint = this.getIsHoveringStartPoint(startPoint, activeEntity.state.position);
    const baseRadius = this.manager.stage.unscale(this.config.START_POINT_RADIUS_PX);
    const stagePoint = addCoords(startPoint, activeEntity.state.position);

    this.konva.startPointIndicator.setAttrs({
      x: stagePoint.x,
      y: stagePoint.y,
      radius:
        baseRadius +
        (isHoveringStartPoint ? this.manager.stage.unscale(this.config.START_POINT_HOVER_RADIUS_DELTA_PX) : 0),
      strokeWidth: this.manager.stage.unscale(this.config.START_POINT_STROKE_WIDTH_PX),
      visible: true,
    });
  };

  private getEntityRelativePoint = (point: Coordinate, position: Coordinate): Coordinate => {
    return floorCoord(offsetCoord(point, position));
  };

  private getCompositeOperation = (): CanvasRectState['compositeOperation'] => {
    return this.manager.stateApi.$ctrlKey.get() || this.manager.stateApi.$metaKey.get()
      ? 'destination-out'
      : 'source-over';
  };

  private getPolygonPoint = (point: Coordinate, shouldSnap: boolean): Coordinate => {
    if (!shouldSnap) {
      return point;
    }

    const lastPoint = this.polygonPoints.at(-1);
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

    const snappedPoint = {
      x: lastPoint.x + Math.cos(snappedAngle) * distance,
      y: lastPoint.y + Math.sin(snappedAngle) * distance,
    };

    return this.alignPointToStart(snappedPoint);
  };

  private isPointNearStart = (point: Coordinate): boolean => {
    const startPoint = this.polygonPoints[0];
    if (!startPoint) {
      return false;
    }
    return Math.hypot(point.x - startPoint.x, point.y - startPoint.y) <= this.getPolygonCloseRadius();
  };

  private getPolygonCloseRadius = (): number => {
    return this.manager.stage.unscale(this.config.POLYGON_CLOSE_RADIUS_PX);
  };

  private getIsHoveringStartPoint = (startPoint: Coordinate, entityPosition: Coordinate): boolean => {
    if (this.polygonPoints.length < 3) {
      return false;
    }

    const pointerPoint = this.parent.$cursorPos.get()?.relative;
    if (!pointerPoint) {
      return false;
    }

    const entityRelativePointerPoint = this.getEntityRelativePoint(pointerPoint, entityPosition);
    return (
      Math.hypot(entityRelativePointerPoint.x - startPoint.x, entityRelativePointerPoint.y - startPoint.y) <=
      this.getPolygonCloseRadius()
    );
  };

  private alignPointToStart = (point: Coordinate): Coordinate => {
    if (this.polygonPoints.length < 2) {
      return point;
    }

    const startPoint = this.polygonPoints[0];
    if (!startPoint) {
      return point;
    }

    const alignThreshold = this.getPolygonCloseRadius();
    const deltaX = Math.abs(point.x - startPoint.x);
    const deltaY = Math.abs(point.y - startPoint.y);
    const canAlignX = deltaX <= alignThreshold;
    const canAlignY = deltaY <= alignThreshold;

    if (!canAlignX && !canAlignY) {
      return point;
    }

    if (canAlignX && canAlignY) {
      if (deltaX <= deltaY) {
        return { x: startPoint.x, y: point.y };
      }
      return { x: point.x, y: startPoint.y };
    }

    if (canAlignX) {
      return { x: startPoint.x, y: point.y };
    }

    return { x: point.x, y: startPoint.y };
  };

  private appendFreehandPoint = (point: Coordinate) => {
    const lastPoint = this.freehandPoints.at(-1) ?? null;
    if (!lastPoint) {
      this.freehandPoints.push(point);
      return;
    }

    const maxSegmentLength = this.manager.stage.unscale(this.config.MAX_FREEHAND_SEGMENT_LENGTH_PX);
    const dx = point.x - lastPoint.x;
    const dy = point.y - lastPoint.y;
    const distance = Math.hypot(dx, dy);

    if (distance <= maxSegmentLength) {
      this.freehandPoints.push(point);
      return;
    }

    const steps = Math.ceil(distance / maxSegmentLength);
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      this.freehandPoints.push({
        x: lastPoint.x + dx * t,
        y: lastPoint.y + dy * t,
      });
    }
  };

  private simplifyFreehandContour = (points: Coordinate[]): Coordinate[] => {
    if (points.length < this.config.FREEHAND_SIMPLIFY_MIN_POINTS) {
      return points;
    }

    const simplifiedFlatPoints = simplifyFlatNumbersArray(
      points.flatMap((point) => [point.x, point.y]),
      {
        tolerance: this.config.FREEHAND_SIMPLIFY_TOLERANCE,
        highestQuality: true,
      }
    );

    if (simplifiedFlatPoints.length < 6) {
      return points;
    }

    const simplifiedPoints = this.flatNumbersToCoords(simplifiedFlatPoints);
    if (simplifiedPoints.length < 3) {
      return points;
    }

    return simplifiedPoints;
  };

  private flatNumbersToCoords = (points: number[]): Coordinate[] => {
    const coords: Coordinate[] = [];
    for (let i = 0; i < points.length; i += 2) {
      const x = points[i];
      const y = points[i + 1];
      if (x === undefined || y === undefined) {
        continue;
      }
      coords.push({ x, y });
    }
    return coords;
  };

  private getDragRect = (
    start: Coordinate,
    end: Coordinate,
    options: { fromCenter: boolean; constrainSquare: boolean }
  ): CanvasRectState['rect'] => {
    let dx = end.x - start.x;
    let dy = end.y - start.y;

    if (options.constrainSquare) {
      const size = Math.max(Math.abs(dx), Math.abs(dy));
      const dxSign = getAxisSign(dx, dy);
      const dySign = getAxisSign(dy, dx);
      dx = dxSign * size;
      dy = dySign * size;
    }

    const x1 = options.fromCenter ? start.x - dx : start.x;
    const y1 = options.fromCenter ? start.y - dy : start.y;
    const x2 = options.fromCenter ? start.x + dx : start.x + dx;
    const y2 = options.fromCenter ? start.y + dy : start.y + dy;

    return {
      x: Math.min(x1, x2),
      y: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
    };
  };

  private getActiveEntityAdapter = () => {
    if (!this.activeEntityIdentifier) {
      return null;
    }
    return this.manager.getAdapter(this.activeEntityIdentifier);
  };

  private finishDragShapeSession = () => {
    const activeEntity = this.getActiveEntityAdapter();
    if (!activeEntity) {
      this.resetState();
      this.render();
      return;
    }

    const bufferState = activeEntity.bufferRenderer.state;
    if (
      bufferState &&
      (bufferState.type === 'rect' || bufferState.type === 'oval') &&
      activeEntity.bufferRenderer.hasBuffer() &&
      bufferState.rect.width > 0 &&
      bufferState.rect.height > 0
    ) {
      activeEntity.bufferRenderer.commitBuffer();
    } else {
      activeEntity.bufferRenderer.clearBuffer();
    }

    this.resetState();
    this.render();
  };

  private clearActiveBuffer = () => {
    this.getActiveEntityAdapter()?.bufferRenderer.clearBuffer();
  };

  private resetState = () => {
    this.activeEntityIdentifier = null;
    this.shapeId = null;
    this.dragStartPoint = null;
    this.dragCurrentPoint = null;
    this.translatePreviousPointerPoint = null;
    this.freehandPoints = [];
    this.isDrawingFreehand = false;
    this.polygonPoints = [];
    this.polygonPointer = null;
    this.konva.startPointIndicator.visible(false);
  };
}
