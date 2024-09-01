import { CanvasBrushToolPreview } from 'features/controlLayers/konva/CanvasBrushToolPreview';
import { CanvasColorPickerToolPreview } from 'features/controlLayers/konva/CanvasColorPickerToolPreview';
import { CanvasEraserToolPreview } from 'features/controlLayers/konva/CanvasEraserToolPreview';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  alignCoordForTool,
  calculateNewBrushSizeFromWheelDelta,
  getIsPrimaryMouseDown,
  getLastPointOfLine,
  getPrefixedId,
  getScaledCursorPosition,
  offsetCoord,
  validateCandidatePoint,
} from 'features/controlLayers/konva/util';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  Coordinate,
  RgbColor,
  Tool,
} from 'features/controlLayers/store/types';
import { isDrawableEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Logger } from 'roarr';

type CanvasToolModuleConfig = {
  BRUSH_SPACING_TARGET_SCALE: number;
};

const DEFAULT_CONFIG: CanvasToolModuleConfig = {
  BRUSH_SPACING_TARGET_SCALE: 0.1,
};

export class CanvasToolModule extends CanvasModuleBase {
  readonly type = 'tool';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;
  subscriptions: Set<() => void> = new Set();

  config: CanvasToolModuleConfig = DEFAULT_CONFIG;

  brushToolPreview: CanvasBrushToolPreview;
  eraserToolPreview: CanvasEraserToolPreview;
  colorPickerToolPreview: CanvasColorPickerToolPreview;

  konva: {
    stage: Konva.Stage;
    group: Konva.Group;
  };

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating tool module');

    this.brushToolPreview = new CanvasBrushToolPreview(this);
    this.eraserToolPreview = new CanvasEraserToolPreview(this);
    this.colorPickerToolPreview = new CanvasColorPickerToolPreview(this);

    this.konva = {
      stage: this.manager.stage.konva.stage,
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
    };

    this.konva.group.add(this.brushToolPreview.konva.group);
    this.konva.group.add(this.eraserToolPreview.konva.group);
    this.konva.group.add(this.colorPickerToolPreview.konva.group);

    this.subscriptions.add(this.manager.stateApi.$stageAttrs.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.$toolState.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.$tool.listen(this.render));

    const cleanupListeners = this.setEventListeners();

    this.subscriptions.add(cleanupListeners);
  }

  setToolVisibility = (tool: Tool, isDrawable: boolean) => {
    this.brushToolPreview.setVisibility(isDrawable && tool === 'brush');
    this.eraserToolPreview.setVisibility(isDrawable && tool === 'eraser');
    this.colorPickerToolPreview.setVisibility(tool === 'colorPicker');
  };

  syncCursorStyle = () => {
    const stage = this.manager.stage;
    const renderedEntityCount = this.manager.stateApi.getRenderedEntityCount();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const isMouseDown = this.manager.stateApi.$isMouseDown.get();
    const tool = this.manager.stateApi.$tool.get();

    const isDrawable =
      !!selectedEntity &&
      selectedEntity.state.isEnabled &&
      !selectedEntity.state.isLocked &&
      isDrawableEntity(selectedEntity.state);

    // Update the stage's pointer style
    if (Boolean(this.manager.stateApi.$transformingEntity.get()) || renderedEntityCount === 0) {
      // We are transforming and/or have no layers, so we should not render any tool
      stage.container.style.cursor = 'default';
    } else if (tool === 'view') {
      // view tool gets a hand
      stage.container.style.cursor = isMouseDown ? 'grabbing' : 'grab';
      // Bbox tool gets default
    } else if (tool === 'bbox') {
      stage.container.style.cursor = 'default';
    } else if (tool === 'colorPicker') {
      // Color picker gets none
      stage.container.style.cursor = 'none';
    } else if (isDrawable) {
      if (tool === 'move') {
        // Move gets default arrow
        stage.container.style.cursor = 'default';
      } else if (tool === 'rect') {
        // Rect gets a crosshair
        stage.container.style.cursor = 'crosshair';
      } else if (tool === 'brush' || tool === 'eraser') {
        // Hide the native cursor and use the konva-rendered brush preview
        stage.container.style.cursor = 'none';
      }
    } else {
      // isDrawable === 'false'
      // Non-drawable layers don't have tools
      stage.container.style.cursor = 'not-allowed';
    }
  };

  render = () => {
    const stage = this.manager.stage;
    const renderedEntityCount = this.manager.stateApi.getRenderedEntityCount();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const cursorPos = this.manager.stateApi.$lastCursorPos.get();
    const tool = this.manager.stateApi.$tool.get();

    const isDrawable =
      !!selectedEntity &&
      selectedEntity.state.isEnabled &&
      !selectedEntity.state.isLocked &&
      isDrawableEntity(selectedEntity.state);

    this.syncCursorStyle();

    stage.setIsDraggable(tool === 'view');

    if (!cursorPos || renderedEntityCount === 0) {
      // We can bail early if the mouse isn't over the stage or there are no layers
      this.konva.group.visible(false);
    } else {
      this.konva.group.visible(true);

      // No need to render the brush preview if the cursor position or color is missing
      if (cursorPos && tool === 'brush') {
        this.brushToolPreview.render();
      } else if (cursorPos && tool === 'eraser') {
        this.eraserToolPreview.render();
      } else if (cursorPos && tool === 'colorPicker') {
        this.colorPickerToolPreview.render();
      }

      this.setToolVisibility(tool, isDrawable);
    }
  };

  syncLastCursorPos = (): Coordinate | null => {
    const pos = getScaledCursorPosition(this.konva.stage);
    this.manager.stateApi.$lastCursorPos.set(pos);
    return pos;
  };

  getColorUnderCursor = (): RgbColor | null => {
    const pos = this.konva.stage.getPointerPosition();
    if (!pos) {
      return null;
    }
    const ctx = this.konva.stage
      .toCanvas({ x: pos.x, y: pos.y, width: 1, height: 1, imageSmoothingEnabled: false })
      .getContext('2d');

    if (!ctx) {
      return null;
    }

    const [r, g, b, _a] = ctx.getImageData(0, 0, 1, 1).data;

    if (r === undefined || g === undefined || b === undefined) {
      return null;
    }

    return { r, g, b };
  };

  getClip = (
    entity: CanvasRegionalGuidanceState | CanvasControlLayerState | CanvasRasterLayerState | CanvasInpaintMaskState
  ) => {
    const settings = this.manager.stateApi.getSettings();

    if (settings.clipToBbox) {
      const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
      return {
        x: x - entity.position.x,
        y: y - entity.position.y,
        width,
        height,
      };
    } else {
      const { x, y } = this.manager.stage.getPosition();
      const scale = this.manager.stage.getScale();
      const { width, height } = this.manager.stage.getSize();
      return {
        x: -x / scale - entity.position.x,
        y: -y / scale - entity.position.y,
        width: width / scale,
        height: height / scale,
      };
    }
  };

  setEventListeners = (): (() => void) => {
    this.konva.stage.on('mouseenter', this.onStageMouseEnter);
    this.konva.stage.on('mousedown', this.onStageMouseDown);
    this.konva.stage.on('mouseup', this.onStageMouseUp);
    this.konva.stage.on('mousemove', this.onStageMouseMove);
    this.konva.stage.on('mouseleave', this.onStageMouseLeave);
    this.konva.stage.on('wheel', this.onStageMouseWheel);

    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);

    return () => {
      this.konva.stage.off('mouseenter', this.onStageMouseEnter);
      this.konva.stage.off('mousedown', this.onStageMouseDown);
      this.konva.stage.off('mouseup', this.onStageMouseUp);
      this.konva.stage.off('mousemove', this.onStageMouseMove);
      this.konva.stage.off('mouseleave', this.onStageMouseLeave);

      this.konva.stage.off('wheel', this.onStageMouseWheel);
      window.removeEventListener('keydown', this.onKeyDown);
      window.removeEventListener('keyup', this.onKeyUp);
    };
  };

  onStageMouseEnter = (_: KonvaEventObject<MouseEvent>) => {
    this.render();
  };

  onStageMouseDown = async (e: KonvaEventObject<MouseEvent>) => {
    this.manager.stateApi.$isMouseDown.set(true);
    const toolState = this.manager.stateApi.getToolState();
    const tool = this.manager.stateApi.$tool.get();
    const pos = this.syncLastCursorPos();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();

    if (tool === 'colorPicker') {
      const color = this.getColorUnderCursor();
      if (color) {
        this.manager.stateApi.$colorUnderCursor.set(color);
      }
      if (color) {
        this.manager.stateApi.setFill({ ...toolState.fill, ...color });
      }
      this.render();
    } else {
      const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked;
      if (pos && isDrawable && !this.manager.stateApi.$spaceKey.get() && getIsPrimaryMouseDown(e)) {
        this.manager.stateApi.$lastMouseDownPos.set(pos);
        const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);

        if (tool === 'brush') {
          const lastLinePoint = selectedEntity.adapter.getLastPointOfLastLine('brush_line');
          const alignedPoint = alignCoordForTool(normalizedPoint, toolState.brush.width);
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }

            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('brush_line'),
              type: 'brush_line',
              points: [
                // The last point of the last line is already normalized to the entity's coordinates
                lastLinePoint.x,
                lastLinePoint.y,
                alignedPoint.x,
                alignedPoint.y,
              ],
              strokeWidth: toolState.brush.width,
              color: this.manager.stateApi.getCurrentFill(),
              clip: this.getClip(selectedEntity.state),
            });
          } else {
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }
            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('brush_line'),
              type: 'brush_line',
              points: [alignedPoint.x, alignedPoint.y],
              strokeWidth: toolState.brush.width,
              color: this.manager.stateApi.getCurrentFill(),
              clip: this.getClip(selectedEntity.state),
            });
          }
          this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
        }

        if (tool === 'eraser') {
          const lastLinePoint = selectedEntity.adapter.getLastPointOfLastLine('eraser_line');
          const alignedPoint = alignCoordForTool(normalizedPoint, toolState.eraser.width);
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }
            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('eraser_line'),
              type: 'eraser_line',
              points: [
                // The last point of the last line is already normalized to the entity's coordinates
                lastLinePoint.x,
                lastLinePoint.y,
                alignedPoint.x,
                alignedPoint.y,
              ],
              strokeWidth: toolState.eraser.width,
              clip: this.getClip(selectedEntity.state),
            });
          } else {
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }
            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('eraser_line'),
              type: 'eraser_line',
              points: [alignedPoint.x, alignedPoint.y],
              strokeWidth: toolState.eraser.width,
              clip: this.getClip(selectedEntity.state),
            });
          }
          this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
        }

        if (tool === 'rect') {
          if (selectedEntity.adapter.renderer.bufferState) {
            selectedEntity.adapter.renderer.commitBuffer();
          }
          await selectedEntity.adapter.renderer.setBuffer({
            id: getPrefixedId('rect'),
            type: 'rect',
            rect: { x: Math.round(normalizedPoint.x), y: Math.round(normalizedPoint.y), width: 0, height: 0 },
            color: this.manager.stateApi.getCurrentFill(),
          });
        }
      }
    }
  };

  onStageMouseUp = (_: KonvaEventObject<MouseEvent>) => {
    this.manager.stateApi.$isMouseDown.set(false);
    const pos = this.manager.stateApi.$lastCursorPos.get();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked;
    const tool = this.manager.stateApi.$tool.get();

    if (pos && isDrawable && !this.manager.stateApi.$spaceKey.get()) {
      if (tool === 'brush') {
        const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
        if (drawingBuffer?.type === 'brush_line') {
          selectedEntity.adapter.renderer.commitBuffer();
        } else {
          selectedEntity.adapter.renderer.clearBuffer();
        }
      }

      if (tool === 'eraser') {
        const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
        if (drawingBuffer?.type === 'eraser_line') {
          selectedEntity.adapter.renderer.commitBuffer();
        } else {
          selectedEntity.adapter.renderer.clearBuffer();
        }
      }

      if (tool === 'rect') {
        const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
        if (drawingBuffer?.type === 'rect') {
          selectedEntity.adapter.renderer.commitBuffer();
        } else {
          selectedEntity.adapter.renderer.clearBuffer();
        }
      }

      this.manager.stateApi.$lastMouseDownPos.set(null);
    }
    this.render();
  };

  onStageMouseMove = async (e: KonvaEventObject<MouseEvent>) => {
    const toolState = this.manager.stateApi.getToolState();
    const pos = this.syncLastCursorPos();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const tool = this.manager.stateApi.$tool.get();

    if (tool === 'colorPicker') {
      const color = this.getColorUnderCursor();
      if (color) {
        this.manager.stateApi.$colorUnderCursor.set(color);
      }
    } else {
      const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked;
      if (pos && isDrawable && !this.manager.stateApi.$spaceKey.get() && getIsPrimaryMouseDown(e)) {
        if (tool === 'brush') {
          const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
          if (drawingBuffer) {
            if (drawingBuffer.type === 'brush_line') {
              const lastPoint = getLastPointOfLine(drawingBuffer.points);
              const minDistance = toolState.brush.width * this.config.BRUSH_SPACING_TARGET_SCALE;
              if (lastPoint && validateCandidatePoint(pos, lastPoint, minDistance)) {
                const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
                const alignedPoint = alignCoordForTool(normalizedPoint, toolState.brush.width);
                // Do not add duplicate points
                if (lastPoint.x !== alignedPoint.x || lastPoint.y !== alignedPoint.y) {
                  drawingBuffer.points.push(alignedPoint.x, alignedPoint.y);
                  await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
                  this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
                }
              }
            } else {
              selectedEntity.adapter.renderer.clearBuffer();
            }
          } else {
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }
            const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
            const alignedPoint = alignCoordForTool(normalizedPoint, toolState.brush.width);
            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('brush_line'),
              type: 'brush_line',
              points: [alignedPoint.x, alignedPoint.y],
              strokeWidth: toolState.brush.width,
              color: this.manager.stateApi.getCurrentFill(),
              clip: this.getClip(selectedEntity.state),
            });
            this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
          }
        }

        if (tool === 'eraser') {
          const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
          if (drawingBuffer) {
            if (drawingBuffer.type === 'eraser_line') {
              const lastPoint = getLastPointOfLine(drawingBuffer.points);
              const minDistance = toolState.eraser.width * this.config.BRUSH_SPACING_TARGET_SCALE;
              if (lastPoint && validateCandidatePoint(pos, lastPoint, minDistance)) {
                const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
                const alignedPoint = alignCoordForTool(normalizedPoint, toolState.eraser.width);
                // Do not add duplicate points
                if (lastPoint.x !== alignedPoint.x || lastPoint.y !== alignedPoint.y) {
                  drawingBuffer.points.push(alignedPoint.x, alignedPoint.y);
                  await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
                  this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
                }
              }
            } else {
              selectedEntity.adapter.renderer.clearBuffer();
            }
          } else {
            if (selectedEntity.adapter.renderer.bufferState) {
              selectedEntity.adapter.renderer.commitBuffer();
            }
            const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
            const alignedPoint = alignCoordForTool(normalizedPoint, toolState.eraser.width);
            await selectedEntity.adapter.renderer.setBuffer({
              id: getPrefixedId('eraser_line'),
              type: 'eraser_line',
              points: [alignedPoint.x, alignedPoint.y],
              strokeWidth: toolState.eraser.width,
              clip: this.getClip(selectedEntity.state),
            });
            this.manager.stateApi.$lastAddedPoint.set(alignedPoint);
          }
        }

        if (tool === 'rect') {
          const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
          if (drawingBuffer) {
            if (drawingBuffer.type === 'rect') {
              const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
              drawingBuffer.rect.width = Math.round(normalizedPoint.x - drawingBuffer.rect.x);
              drawingBuffer.rect.height = Math.round(normalizedPoint.y - drawingBuffer.rect.y);
              await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
            } else {
              selectedEntity.adapter.renderer.clearBuffer();
            }
          }
        }
      }
    }

    this.render();
  };

  onStageMouseLeave = async (e: KonvaEventObject<MouseEvent>) => {
    const pos = this.syncLastCursorPos();
    this.manager.stateApi.$lastCursorPos.set(null);
    this.manager.stateApi.$lastMouseDownPos.set(null);
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const toolState = this.manager.stateApi.getToolState();
    const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked;
    const tool = this.manager.stateApi.$tool.get();

    if (pos && isDrawable && !this.manager.stateApi.$spaceKey.get() && getIsPrimaryMouseDown(e)) {
      const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
      const normalizedPoint = offsetCoord(pos, selectedEntity.state.position);
      if (tool === 'brush' && drawingBuffer?.type === 'brush_line') {
        const alignedPoint = alignCoordForTool(normalizedPoint, toolState.brush.width);
        drawingBuffer.points.push(alignedPoint.x, alignedPoint.y);
        await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
        selectedEntity.adapter.renderer.commitBuffer();
      } else if (tool === 'eraser' && drawingBuffer?.type === 'eraser_line') {
        const alignedPoint = alignCoordForTool(normalizedPoint, toolState.eraser.width);
        drawingBuffer.points.push(alignedPoint.x, alignedPoint.y);
        await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
        selectedEntity.adapter.renderer.commitBuffer();
      } else if (tool === 'rect' && drawingBuffer?.type === 'rect') {
        drawingBuffer.rect.width = Math.round(normalizedPoint.x - drawingBuffer.rect.x);
        drawingBuffer.rect.height = Math.round(normalizedPoint.y - drawingBuffer.rect.y);
        await selectedEntity.adapter.renderer.setBuffer(drawingBuffer);
        selectedEntity.adapter.renderer.commitBuffer();
      }
    }

    this.render();
  };

  onStageMouseWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();

    if (!e.evt.ctrlKey && !e.evt.metaKey) {
      return;
    }

    const toolState = this.manager.stateApi.getToolState();
    const tool = this.manager.stateApi.$tool.get();

    let delta = e.evt.deltaY;

    if (toolState.invertScroll) {
      delta = -delta;
    }

    // Holding ctrl or meta while scrolling changes the brush size
    if (tool === 'brush') {
      this.manager.stateApi.setBrushWidth(calculateNewBrushSizeFromWheelDelta(toolState.brush.width, delta));
    } else if (tool === 'eraser') {
      this.manager.stateApi.setEraserWidth(calculateNewBrushSizeFromWheelDelta(toolState.eraser.width, delta));
    }

    this.render();
  };

  onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (e.key === 'Escape') {
      // Cancel shape drawing on escape
      const selectedEntity = this.manager.stateApi.getSelectedEntity();
      if (selectedEntity) {
        selectedEntity.adapter.renderer.clearBuffer();
        this.manager.stateApi.$lastMouseDownPos.set(null);
      }
    } else if (e.key === ' ') {
      // Select the view tool on space key down
      this.manager.stateApi.$toolBuffer.set(this.manager.stateApi.$tool.get());
      this.manager.stateApi.$tool.set('view');
      this.manager.stateApi.$spaceKey.set(true);
      this.manager.stateApi.$lastCursorPos.set(null);
      this.manager.stateApi.$lastMouseDownPos.set(null);
    } else if (e.key === 'Alt') {
      // Select the color picker on alt key down
      this.manager.stateApi.$toolBuffer.set(this.manager.stateApi.$tool.get());
      this.manager.stateApi.$tool.set('colorPicker');
    }
  };

  onKeyUp = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (e.key === ' ') {
      // Revert the tool to the previous tool on space key up
      const toolBuffer = this.manager.stateApi.$toolBuffer.get();
      this.manager.stateApi.$tool.set(toolBuffer ?? 'move');
      this.manager.stateApi.$toolBuffer.set(null);
      this.manager.stateApi.$spaceKey.set(false);
    } else if (e.key === 'Alt') {
      // Revert the tool to the previous tool on alt key up
      const toolBuffer = this.manager.stateApi.$toolBuffer.get();
      this.manager.stateApi.$tool.set(toolBuffer ?? 'move');
      this.manager.stateApi.$toolBuffer.set(null);
    }
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };
}
