import { CanvasBrushToolPreview } from 'features/controlLayers/konva/CanvasBrushToolPreview';
import { CanvasColorPickerToolPreview } from 'features/controlLayers/konva/CanvasColorPickerToolPreview';
import { CanvasEraserToolPreview } from 'features/controlLayers/konva/CanvasEraserToolPreview';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  alignCoordForTool,
  calculateNewBrushSizeFromWheelDelta,
  floorCoord,
  getIsPrimaryMouseDown,
  getLastPointOfLastLine,
  getLastPointOfLine,
  getPrefixedId,
  getScaledCursorPosition,
  isDistanceMoreThanMin,
  offsetCoord,
} from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  Coordinate,
  RgbColor,
  Tool,
} from 'features/controlLayers/store/types';
import { isRenderableEntity, RGBA_BLACK } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasToolModuleConfig = {
  BRUSH_SPACING_TARGET_SCALE: number;
};

const DEFAULT_CONFIG: CanvasToolModuleConfig = {
  BRUSH_SPACING_TARGET_SCALE: 0.1,
};

export class CanvasToolModule extends CanvasModuleBase {
  readonly type = 'tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;
  subscriptions: Set<() => void> = new Set();

  config: CanvasToolModuleConfig = DEFAULT_CONFIG;

  brushToolPreview: CanvasBrushToolPreview;
  eraserToolPreview: CanvasEraserToolPreview;
  colorPickerToolPreview: CanvasColorPickerToolPreview;

  /**
   * The currently selected tool.
   */
  $tool = atom<Tool>('brush');
  /**
   * A buffer for the currently selected tool. This is used to temporarily store the tool while the user is using any
   * hold-to-activate tools, like the view or color picker tools.
   */
  $toolBuffer = atom<Tool | null>(null);
  /**
   * Whether the mouse is currently down.
   */
  $isMouseDown = atom<boolean>(false);
  /**
   * The last cursor position.
   */
  $cursorPos = atom<Coordinate | null>(null);
  /**
   * The color currently under the cursor. Only has a value when the color picker tool is active.
   */
  $colorUnderCursor = atom<RgbColor>(RGBA_BLACK);

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

    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSettingsSlice, this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.syncCursorStyle));
    this.subscriptions.add(
      this.$tool.listen(() => {
        // On tool switch, reset mouse state
        this.manager.tool.$isMouseDown.set(false);
        this.render();
      })
    );

    const cleanupListeners = this.setEventListeners();

    this.subscriptions.add(cleanupListeners);
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.render();
    this.syncCursorStyle();
  };

  setToolVisibility = (tool: Tool, isDrawable: boolean) => {
    this.brushToolPreview.setVisibility(isDrawable && tool === 'brush');
    this.eraserToolPreview.setVisibility(isDrawable && tool === 'eraser');
    this.colorPickerToolPreview.setVisibility(tool === 'colorPicker');
  };

  syncCursorStyle = () => {
    const stage = this.manager.stage;
    const isMouseDown = this.$isMouseDown.get();
    const tool = this.$tool.get();

    if (tool === 'view') {
      stage.setCursor(isMouseDown ? 'grabbing' : 'grab');
    } else if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      stage.setCursor('not-allowed');
    } else if (this.manager.$isBusy.get()) {
      stage.setCursor('not-allowed');
    } else if (!this.manager.stateApi.getSelectedEntityAdapter()?.getIsInteractable()) {
      stage.setCursor('not-allowed');
    } else if (tool === 'colorPicker' || tool === 'brush' || tool === 'eraser') {
      stage.setCursor('none');
    } else if (tool === 'move' || tool === 'bbox') {
      stage.setCursor('default');
    } else if (tool === 'rect') {
      stage.setCursor('crosshair');
    } else {
      stage.setCursor('not-allowed');
    }
  };

  render = () => {
    const stage = this.manager.stage;
    const renderedEntityCount = this.manager.stateApi.getRenderedEntityCount();
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    const cursorPos = this.$cursorPos.get();
    const tool = this.$tool.get();

    const isDrawable =
      !!selectedEntity &&
      selectedEntity.state.isEnabled &&
      !selectedEntity.state.isLocked &&
      isRenderableEntity(selectedEntity.state);

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
    this.$cursorPos.set(pos);
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
    window.addEventListener('pointerup', this.onWindowPointerUp);

    return () => {
      this.konva.stage.off('mouseenter', this.onStageMouseEnter);
      this.konva.stage.off('mousedown', this.onStageMouseDown);
      this.konva.stage.off('mouseup', this.onStageMouseUp);
      this.konva.stage.off('mousemove', this.onStageMouseMove);
      this.konva.stage.off('mouseleave', this.onStageMouseLeave);

      this.konva.stage.off('wheel', this.onStageMouseWheel);
      window.removeEventListener('keydown', this.onKeyDown);
      window.removeEventListener('keyup', this.onKeyUp);
      window.removeEventListener('pointerup', this.onWindowPointerUp);
    };
  };

  getCanDraw = (): boolean => {
    if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      return false;
    } else if (this.manager.stateApi.$isTranforming.get()) {
      return false;
    } else if (this.manager.filter.$isFiltering.get()) {
      return false;
    } else if (!this.manager.stateApi.getSelectedEntityAdapter()?.getIsInteractable()) {
      return false;
    } else {
      return true;
    }
  };

  onStageMouseEnter = async (_: KonvaEventObject<MouseEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    const cursorPos = this.syncLastCursorPos();
    try {
      const isMouseDown = this.$isMouseDown.get();
      const settings = this.manager.stateApi.getSettings();
      const tool = this.$tool.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (!cursorPos || !isMouseDown || !selectedEntity?.state.isEnabled || selectedEntity.state.isLocked) {
        return;
      }

      if (selectedEntity.renderer.bufferState?.type !== 'rect' && selectedEntity.renderer.hasBuffer()) {
        selectedEntity.renderer.commitBuffer();
        return;
      }

      if (tool === 'brush') {
        const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
        await selectedEntity.renderer.setBuffer({
          id: getPrefixedId('brush_line'),
          type: 'brush_line',
          points: [alignedPoint.x, alignedPoint.y],
          strokeWidth: settings.brushWidth,
          color: this.manager.stateApi.getCurrentColor(),
          clip: this.getClip(selectedEntity.state),
        });
        return;
      }

      if (tool === 'eraser') {
        const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
        if (selectedEntity.renderer.bufferState && selectedEntity.renderer.hasBuffer()) {
          selectedEntity.renderer.commitBuffer();
        }
        await selectedEntity.renderer.setBuffer({
          id: getPrefixedId('eraser_line'),
          type: 'eraser_line',
          points: [alignedPoint.x, alignedPoint.y],
          strokeWidth: settings.eraserWidth,
          clip: this.getClip(selectedEntity.state),
        });
        return;
      }
    } finally {
      this.render();
    }
  };

  onStageMouseDown = async (e: KonvaEventObject<MouseEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    this.$isMouseDown.set(getIsPrimaryMouseDown(e));
    const cursorPos = this.syncLastCursorPos();

    try {
      const tool = this.$tool.get();
      const settings = this.manager.stateApi.getSettings();

      if (tool === 'colorPicker') {
        const color = this.getColorUnderCursor();
        if (color) {
          this.manager.stateApi.setColor({ ...settings.color, ...color });
        }
        return;
      }

      const isMouseDown = this.$isMouseDown.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (!cursorPos || !isMouseDown || !selectedEntity?.state.isEnabled || selectedEntity?.state.isLocked) {
        return;
      }

      const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);

      if (tool === 'brush') {
        const lastLinePoint = getLastPointOfLastLine(selectedEntity.state.objects, 'brush_line');
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
        if (e.evt.shiftKey && lastLinePoint) {
          // Create a straight line from the last line point
          if (selectedEntity.renderer.hasBuffer()) {
            selectedEntity.renderer.commitBuffer();
          }

          await selectedEntity.renderer.setBuffer({
            id: getPrefixedId('brush_line'),
            type: 'brush_line',
            points: [
              // The last point of the last line is already normalized to the entity's coordinates
              lastLinePoint.x,
              lastLinePoint.y,
              alignedPoint.x,
              alignedPoint.y,
            ],
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          if (selectedEntity.renderer.hasBuffer()) {
            selectedEntity.renderer.commitBuffer();
          }
          await selectedEntity.renderer.setBuffer({
            id: getPrefixedId('brush_line'),
            type: 'brush_line',
            points: [alignedPoint.x, alignedPoint.y],
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        }
      }

      if (tool === 'eraser') {
        const lastLinePoint = getLastPointOfLastLine(selectedEntity.state.objects, 'eraser_line');
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.eraserWidth);
        if (e.evt.shiftKey && lastLinePoint) {
          // Create a straight line from the last line point
          if (selectedEntity.renderer.hasBuffer()) {
            selectedEntity.renderer.commitBuffer();
          }
          await selectedEntity.renderer.setBuffer({
            id: getPrefixedId('eraser_line'),
            type: 'eraser_line',
            points: [
              // The last point of the last line is already normalized to the entity's coordinates
              lastLinePoint.x,
              lastLinePoint.y,
              alignedPoint.x,
              alignedPoint.y,
            ],
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          if (selectedEntity.renderer.hasBuffer()) {
            selectedEntity.renderer.commitBuffer();
          }
          await selectedEntity.renderer.setBuffer({
            id: getPrefixedId('eraser_line'),
            type: 'eraser_line',
            points: [alignedPoint.x, alignedPoint.y],
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        }
      }

      if (tool === 'rect') {
        if (selectedEntity.renderer.hasBuffer()) {
          selectedEntity.renderer.commitBuffer();
        }
        await selectedEntity.renderer.setBuffer({
          id: getPrefixedId('rect'),
          type: 'rect',
          rect: { x: Math.round(normalizedPoint.x), y: Math.round(normalizedPoint.y), width: 0, height: 0 },
          color: this.manager.stateApi.getCurrentColor(),
        });
      }
    } finally {
      this.render();
    }
  };

  onStageMouseUp = (_: KonvaEventObject<MouseEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    try {
      this.$isMouseDown.set(false);
      const cursorPos = this.syncLastCursorPos();
      if (!cursorPos) {
        return;
      }
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
      const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked;
      if (!isDrawable) {
        return;
      }
      const tool = this.$tool.get();

      if (tool === 'brush') {
        if (selectedEntity.renderer.bufferState?.type === 'brush_line' && selectedEntity.renderer.hasBuffer()) {
          selectedEntity.renderer.commitBuffer();
        } else {
          selectedEntity.renderer.clearBuffer();
        }
      }

      if (tool === 'eraser') {
        if (selectedEntity.renderer.bufferState?.type === 'eraser_line' && selectedEntity.renderer.hasBuffer()) {
          selectedEntity.renderer.commitBuffer();
        } else {
          selectedEntity.renderer.clearBuffer();
        }
      }

      if (tool === 'rect') {
        if (selectedEntity.renderer.bufferState?.type === 'rect' && selectedEntity.renderer.hasBuffer()) {
          selectedEntity.renderer.commitBuffer();
        } else {
          selectedEntity.renderer.clearBuffer();
        }
      }
    } finally {
      this.render();
    }
  };

  onStageMouseMove = async (_: KonvaEventObject<MouseEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    try {
      const tool = this.$tool.get();
      const cursorPos = this.syncLastCursorPos();

      if (tool === 'colorPicker') {
        const color = this.getColorUnderCursor();
        if (color) {
          this.$colorUnderCursor.set(color);
        }
        return;
      }

      const isMouseDown = this.$isMouseDown.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
      const isDrawable = selectedEntity?.state.isEnabled && !selectedEntity.state.isLocked && cursorPos && isMouseDown;

      if (!isDrawable) {
        return;
      }

      const bufferState = selectedEntity.renderer.bufferState;

      if (!bufferState) {
        return;
      }

      const settings = this.manager.stateApi.getSettings();

      if (tool === 'brush' && bufferState.type === 'brush_line') {
        const lastPoint = getLastPointOfLine(bufferState.points);
        const minDistance = settings.brushWidth * this.config.BRUSH_SPACING_TARGET_SCALE;
        if (!lastPoint || !isDistanceMoreThanMin(cursorPos, lastPoint, minDistance)) {
          return;
        }

        const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);

        if (lastPoint.x === alignedPoint.x && lastPoint.y === alignedPoint.y) {
          // Do not add duplicate points
          return;
        }

        bufferState.points.push(alignedPoint.x, alignedPoint.y);
        await selectedEntity.renderer.setBuffer(bufferState);
      } else if (tool === 'eraser' && bufferState.type === 'eraser_line') {
        const lastPoint = getLastPointOfLine(bufferState.points);
        const minDistance = settings.eraserWidth * this.config.BRUSH_SPACING_TARGET_SCALE;
        if (!lastPoint || !isDistanceMoreThanMin(cursorPos, lastPoint, minDistance)) {
          return;
        }

        const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.eraserWidth);

        if (lastPoint.x === alignedPoint.x && lastPoint.y === alignedPoint.y) {
          // Do not add duplicate points
          return;
        }

        bufferState.points.push(alignedPoint.x, alignedPoint.y);
        await selectedEntity.renderer.setBuffer(bufferState);
      } else if (tool === 'rect' && bufferState.type === 'rect') {
        const normalizedPoint = offsetCoord(cursorPos, selectedEntity.state.position);
        const alignedPoint = floorCoord(normalizedPoint);
        bufferState.rect.width = Math.round(alignedPoint.x - bufferState.rect.x);
        bufferState.rect.height = Math.round(alignedPoint.y - bufferState.rect.y);
        await selectedEntity.renderer.setBuffer(bufferState);
      } else {
        selectedEntity?.renderer.clearBuffer();
      }
    } finally {
      this.render();
    }
  };

  onStageMouseLeave = (_: KonvaEventObject<MouseEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    this.$cursorPos.set(null);
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (selectedEntity && selectedEntity.renderer.bufferState?.type !== 'rect' && selectedEntity.renderer.hasBuffer()) {
      selectedEntity.renderer.commitBuffer();
    }

    this.render();
  };

  onStageMouseWheel = (e: KonvaEventObject<WheelEvent>) => {
    if (!this.getCanDraw()) {
      return;
    }

    e.evt.preventDefault();

    if (!e.evt.ctrlKey && !e.evt.metaKey) {
      return;
    }

    const settings = this.manager.stateApi.getSettings();
    const tool = this.$tool.get();

    let delta = e.evt.deltaY;

    if (settings.invertScrollForToolWidth) {
      delta = -delta;
    }

    // Holding ctrl or meta while scrolling changes the brush size
    if (tool === 'brush') {
      this.manager.stateApi.setBrushWidth(calculateNewBrushSizeFromWheelDelta(settings.brushWidth, delta));
    } else if (tool === 'eraser') {
      this.manager.stateApi.setEraserWidth(calculateNewBrushSizeFromWheelDelta(settings.eraserWidth, delta));
    }

    this.render();
  };

  onWindowPointerUp = () => {
    this.$isMouseDown.set(false);
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (selectedEntity && selectedEntity.renderer.hasBuffer() && !this.manager.$isBusy.get()) {
      selectedEntity.renderer.commitBuffer();
    }
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
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
      if (selectedEntity) {
        selectedEntity.renderer.clearBuffer();
      }
      return;
    }

    if (e.key === ' ') {
      // Select the view tool on space key down
      this.$toolBuffer.set(this.$tool.get());
      this.$tool.set('view');
      this.manager.stateApi.$spaceKey.set(true);
      this.$cursorPos.set(null);
      return;
    }

    if (e.key === 'Alt') {
      // Select the color picker on alt key down
      this.$toolBuffer.set(this.$tool.get());
      this.$tool.set('colorPicker');
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
      const toolBuffer = this.$toolBuffer.get();
      this.$tool.set(toolBuffer ?? 'move');
      this.$toolBuffer.set(null);
      this.manager.stateApi.$spaceKey.set(false);
      return;
    }

    if (e.key === 'Alt') {
      // Revert the tool to the previous tool on alt key up
      const toolBuffer = this.$toolBuffer.get();
      this.$tool.set(toolBuffer ?? 'move');
      this.$toolBuffer.set(null);
      return;
    }
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };
}
