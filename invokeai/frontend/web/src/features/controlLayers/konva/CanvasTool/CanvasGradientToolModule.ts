import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
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
      const bboxCenter = {
        x: bboxInLayer.x + bboxInLayer.width / 2,
        y: bboxInLayer.y + bboxInLayer.height / 2,
      };
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const halfWidth = (Math.abs(bboxInLayer.width * cos) + Math.abs(bboxInLayer.height * sin)) / 2;
      const halfHeight = (Math.abs(bboxInLayer.width * sin) + Math.abs(bboxInLayer.height * cos)) / 2;
      rect = {
        x: bboxCenter.x - halfWidth,
        y: bboxCenter.y - halfHeight,
        width: Math.max(halfWidth * 2, 1),
        height: Math.max(halfHeight * 2, 1),
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

    if (settings.gradientType === 'radial') {
      let radialRect = rect;
      let radialCenter = startInRect;
      let radialRadius = Math.max(1, radius);
      if (!clipEnabled) {
        radialRect = bboxInLayer;
        radialCenter = offsetCoord(start, { x: radialRect.x, y: radialRect.y });
      }
      await selectedEntity.bufferRenderer.setBuffer({
        id,
        type: 'gradient',
        gradientType: 'radial',
        rect: radialRect,
        center: radialCenter,
        radius: radialRadius,
        clipCenter: start,
        clipRadius: Math.max(1, radius),
        clipEnabled,
        bboxRect: bboxInLayer,
        fgColor: activeColor,
        bgColor: inactiveColor,
      });
    } else {
      const endPoint = {
        x: endInRect.x === startInRect.x && endInRect.y === startInRect.y ? endInRect.x + 1 : endInRect.x,
        y: endInRect.x === startInRect.x && endInRect.y === startInRect.y ? endInRect.y : endInRect.y,
      };
      await selectedEntity.bufferRenderer.setBuffer({
        id,
        type: 'gradient',
        gradientType: 'linear',
        rect,
        start: startInRect,
        end: endPoint,
        clipCenter: start,
        clipRadius: Math.max(1, radius),
        clipAngle: angle,
        clipEnabled,
        bboxRect: bboxInLayer,
        fgColor: activeColor,
        bgColor: inactiveColor,
      });
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
