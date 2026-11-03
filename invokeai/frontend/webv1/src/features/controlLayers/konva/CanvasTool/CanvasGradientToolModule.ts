import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { buildGradientBufferState } from 'features/controlLayers/konva/CanvasTool/gradientBufferState';
import { getTransparencyLockedCompositeOperation } from 'features/controlLayers/konva/CanvasTool/transparencyLocking';
import { getPrefixedId, offsetCoord } from 'features/controlLayers/konva/util';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Logger } from 'roarr';

export class CanvasGradientToolModule extends CanvasModuleBase {
  readonly type = 'gradient_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  private startPoint: { x: number; y: number } | null = null;
  private lastPoint: { x: number; y: number } | null = null;
  private gradientId: string | null = null;

  constructor(parent: CanvasToolModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');
  }

  syncCursorStyle = () => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity || selectedEntity.state.type !== 'raster_layer') {
      this.manager.stage.setCursor('not-allowed');
      return;
    }
    this.manager.stage.setCursor('crosshair');
  };

  onStagePointerDown = async (_e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();
    const isPrimaryPointerDown = this.parent.$isPrimaryPointerDown.get();
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!cursorPos || !isPrimaryPointerDown || !selectedEntity || selectedEntity.state.type !== 'raster_layer') {
      return;
    }

    if (selectedEntity.bufferRenderer.hasBuffer()) {
      selectedEntity.bufferRenderer.commitBuffer();
    }

    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
    this.startPoint = normalizedPoint;
    this.lastPoint = normalizedPoint;
    this.gradientId = getPrefixedId('gradient');

    await this.updateGradientBuffer(normalizedPoint, normalizedPoint);
  };

  onStagePointerMove = async (_e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();

    if (!cursorPos || !this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity || selectedEntity.state.type !== 'raster_layer') {
      return;
    }

    const bufferState = selectedEntity.bufferRenderer.state;
    if (!bufferState || bufferState.type !== 'gradient' || !this.startPoint) {
      return;
    }

    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
    this.lastPoint = normalizedPoint;
    await this.updateGradientBuffer(this.startPoint, normalizedPoint);
  };

  onStagePointerUp = (_e: KonvaEventObject<PointerEvent>) => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity) {
      this.resetState();
      return;
    }

    const shouldCommit = this.startPoint && this.lastPoint && this.getRadius(this.startPoint, this.lastPoint) > 1;
    if (selectedEntity.bufferRenderer.state?.type === 'gradient' && selectedEntity.bufferRenderer.hasBuffer()) {
      if (shouldCommit) {
        selectedEntity.bufferRenderer.commitBuffer();
      } else {
        selectedEntity.bufferRenderer.clearBuffer();
      }
    }

    this.resetState();
  };

  private resetState = () => {
    this.startPoint = null;
    this.lastPoint = null;
    this.gradientId = null;
  };

  private getRadius = (start: { x: number; y: number }, end: { x: number; y: number }) => {
    return Math.hypot(end.x - start.x, end.y - start.y);
  };

  private updateGradientBuffer = async (start: { x: number; y: number }, end: { x: number; y: number }) => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity || selectedEntity.state.type !== 'raster_layer') {
      return;
    }

    const settings = this.manager.stateApi.getSettings();
    const { rect: bboxRect } = this.manager.stateApi.getBbox();
    const bboxInLayer = {
      x: bboxRect.x - selectedEntity.state.position.x,
      y: bboxRect.y - selectedEntity.state.position.y,
      width: bboxRect.width,
      height: bboxRect.height,
    };
    const radius = this.getRadius(start, end);
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    let rect = {
      x: start.x - radius,
      y: start.y - radius,
      width: radius * 2,
      height: radius * 2,
    };

    if (settings.gradientType === 'linear') {
      // Always render linear gradients on a rect that fully covers the bbox.
      // Angle-dependent rect sizing can undershoot on non-square bboxes (e.g. 90deg on tall bboxes),
      // leaving uncovered bands when the result is clipped back to bbox.
      rect = {
        x: bboxInLayer.x,
        y: bboxInLayer.y,
        width: Math.max(bboxInLayer.width, 1),
        height: Math.max(bboxInLayer.height, 1),
      };
    }

    const startInRect = offsetCoord(start, { x: rect.x, y: rect.y });
    const endInRect = offsetCoord(end, { x: rect.x, y: rect.y });

    const id = this.gradientId ?? getPrefixedId('gradient');
    if (!this.gradientId) {
      this.gradientId = id;
    }

    const activeColor = settings.activeColor === 'bgColor' ? settings.bgColor : settings.fgColor;
    const inactiveColor = settings.activeColor === 'bgColor' ? settings.fgColor : settings.bgColor;
    const clipEnabled = settings.gradientClipEnabled;
    const globalCompositeOperation = getTransparencyLockedCompositeOperation(selectedEntity.state);

    if (settings.gradientType === 'radial') {
      let radialRect = rect;
      let radialCenter = startInRect;
      let radialRadius = Math.max(1, radius);
      if (!clipEnabled) {
        radialRect = bboxInLayer;
        radialCenter = offsetCoord(start, { x: radialRect.x, y: radialRect.y });
      }
      await selectedEntity.bufferRenderer.setBuffer(
        buildGradientBufferState({
          id,
          gradientType: 'radial',
          rect: radialRect,
          center: radialCenter,
          radius: radialRadius,
          clipCenter: start,
          clipRadius: radius,
          clipEnabled,
          bboxRect: bboxInLayer,
          fgColor: activeColor,
          bgColor: inactiveColor,
          globalCompositeOperation,
        })
      );
    } else {
      await selectedEntity.bufferRenderer.setBuffer(
        buildGradientBufferState({
          id,
          gradientType: 'linear',
          rect,
          start: startInRect,
          end: endInRect,
          clipCenter: start,
          clipRadius: radius,
          clipAngle: angle,
          clipEnabled,
          bboxRect: bboxInLayer,
          fgColor: activeColor,
          bgColor: inactiveColor,
          globalCompositeOperation,
        })
      );
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      startPoint: this.startPoint,
      gradientId: this.gradientId,
    };
  };
}
