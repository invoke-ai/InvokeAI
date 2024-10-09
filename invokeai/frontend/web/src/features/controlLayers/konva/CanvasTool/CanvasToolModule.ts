import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasToolBrush } from 'features/controlLayers/konva/CanvasTool/CanvasToolBrush';
import { CanvasToolColorPicker } from 'features/controlLayers/konva/CanvasTool/CanvasToolColorPicker';
import { CanvasToolEraser } from 'features/controlLayers/konva/CanvasTool/CanvasToolEraser';
import {
  alignCoordForTool,
  calculateNewBrushSizeFromWheelDelta,
  floorCoord,
  getColorAtCoordinate,
  getIsPrimaryMouseDown,
  getLastPointOfLastLine,
  getLastPointOfLastLineWithPressure,
  getLastPointOfLine,
  getPrefixedId,
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
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import rafThrottle from 'raf-throttle';
import type { Logger } from 'roarr';

// Konva's docs say the default drag buttons are [0], but it's actually [0,1]. We only want left-click to drag, so we
// need to override the default. The stage handles middle-mouse dragging on its own with dedicated event listeners.
// TODO(psyche): Fix the docs upstream!
Konva.dragButtons = [0];

// Typo insurance
const KEY_ESCAPE = 'Escape';
const KEY_SPACE = ' ';
const KEY_ALT = 'Alt';

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

  brushToolPreview: CanvasToolBrush;
  eraserToolPreview: CanvasToolEraser;
  colorPickerToolPreview: CanvasToolColorPicker;

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
  $cursorPos = atom<{ relative: Coordinate; absolute: Coordinate } | null>(null);
  /**
   * The color currently under the cursor. Only has a value when the color picker tool is active.
   */
  $colorUnderCursor = atom<RgbColor>(RGBA_BLACK);
  /**
   * The last pointer type that was used on the stage. This is used to determine if we should show a tool preview. For
   * example, when using a pen, we should not show a brush preview.
   */
  $lastPointerType = atom<string | null>(null);

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

    this.brushToolPreview = new CanvasToolBrush(this);
    this.eraserToolPreview = new CanvasToolEraser(this);
    this.colorPickerToolPreview = new CanvasToolColorPicker(this);

    this.konva = {
      stage: this.manager.stage.konva.stage,
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
    };

    this.konva.group.add(this.brushToolPreview.konva.group);
    this.konva.group.add(this.eraserToolPreview.konva.group);
    this.konva.group.add(this.colorPickerToolPreview.konva.group);

    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.render));
    this.subscriptions.add(this.manager.$isBusy.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSettingsSlice, this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.render));
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
    const tool = this.$tool.get();
    const isStageDragging = this.manager.stage.konva.stage.isDragging();

    if (tool === 'view' && !isStageDragging) {
      stage.setCursor('grab');
    } else if (this.manager.stage.konva.stage.isDragging()) {
      stage.setCursor('grabbing');
    } else if (this.manager.stateApi.$isTransforming.get()) {
      stage.setCursor('default');
    } else if (this.manager.stateApi.$isFiltering.get()) {
      stage.setCursor('not-allowed');
    } else if (this.manager.stagingArea.$isStaging.get()) {
      stage.setCursor('not-allowed');
    } else if (tool === 'bbox') {
      stage.setCursor('default');
    } else if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      stage.setCursor('not-allowed');
    } else if (!this.manager.stateApi.getSelectedEntityAdapter()?.$isInteractable.get()) {
      stage.setCursor('not-allowed');
    } else if (tool === 'colorPicker' || tool === 'brush' || tool === 'eraser') {
      stage.setCursor('none');
    } else if (tool === 'move') {
      stage.setCursor('default');
    } else if (tool === 'rect') {
      stage.setCursor('crosshair');
    } else {
      stage.setCursor('not-allowed');
    }
  };

  render = () => {
    const renderedEntityCount = this.manager.stateApi.getRenderedEntityCount();
    const cursorPos = this.$cursorPos.get();
    const isFiltering = this.manager.stateApi.$isFiltering.get();
    const isStaging = this.manager.stagingArea.$isStaging.get();
    const isStageDragging = this.manager.stage.konva.stage.isDragging();

    this.syncCursorStyle();

    /**
     * The tool should not be rendered when:
     * - There is no cursor position (i.e. the cursor is outside of the stage)
     * - The user is filtering, in which case the user is not allowed to use the tools. Note that we do not disable
     * the group while transforming, bc that requires use of the move tool.
     * - The canvas is staging, in which case the user is not allowed to use the tools.
     * - There are no entities rendered on the canvas. Maybe we should allow the user to draw on an empty canvas,
     * creating a new layer when they start?
     * - The stage is being dragged, in which case the user is not allowed to use the tools.
     */
    if (!cursorPos || isFiltering || isStaging || renderedEntityCount === 0 || isStageDragging) {
      this.konva.group.visible(false);
    } else {
      this.konva.group.visible(true);
      this.brushToolPreview.render();
      this.eraserToolPreview.render();
      this.colorPickerToolPreview.render();
    }
  };

  syncCursorPositions = () => {
    const relative = this.konva.stage.getRelativePointerPosition();
    const absolute = this.konva.stage.getPointerPosition();

    if (!relative || !absolute) {
      return;
    }

    this.$cursorPos.set({ relative, absolute });
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
    this.konva.stage.on('pointerenter', this.onStagePointerEnter);
    this.konva.stage.on('pointerdown', this.onStagePointerDown);
    this.konva.stage.on('pointerup', this.onStagePointerUp);
    this.konva.stage.on('pointermove', this.onStagePointerMove);

    // The Konva stage doesn't appear to handle pointerleave events, so we need to listen to the container instead
    this.manager.stage.container.addEventListener('pointerleave', this.onStagePointerLeave);

    this.konva.stage.on('wheel', this.onStageMouseWheel);

    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('pointerup', this.onWindowPointerUp);
    window.addEventListener('blur', this.onWindowBlur);

    return () => {
      this.konva.stage.off('pointerenter', this.onStagePointerEnter);
      this.konva.stage.off('pointerdown', this.onStagePointerDown);
      this.konva.stage.off('pointerup', this.onStagePointerUp);
      this.konva.stage.off('pointermove', this.onStagePointerMove);

      this.manager.stage.container.removeEventListener('pointerleave', this.onStagePointerLeave);

      this.konva.stage.off('wheel', this.onStageMouseWheel);

      window.removeEventListener('keydown', this.onKeyDown);
      window.removeEventListener('keyup', this.onKeyUp);
      window.removeEventListener('pointerup', this.onWindowPointerUp);
      window.removeEventListener('blur', this.onWindowBlur);
    };
  };

  getCanDraw = (): boolean => {
    if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      return false;
    }

    if (this.manager.$isBusy.get()) {
      return false;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity) {
      return false;
    }

    if (!selectedEntity.$isInteractable.get()) {
      return false;
    }

    return true;
  };

  onStagePointerEnter = async (e: KonvaEventObject<PointerEvent>) => {
    try {
      this.$lastPointerType.set(e.evt.pointerType);

      if (!this.getCanDraw()) {
        return;
      }

      this.syncCursorPositions();
      const cursorPos = this.$cursorPos.get();

      const isMouseDown = this.$isMouseDown.get();
      const settings = this.manager.stateApi.getSettings();
      const tool = this.$tool.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (!cursorPos || !isMouseDown || !selectedEntity?.$isInteractable.get()) {
        return;
      }

      if (selectedEntity.bufferRenderer.state?.type !== 'rect' && selectedEntity.bufferRenderer.hasBuffer()) {
        selectedEntity.bufferRenderer.commitBuffer();
        return;
      }

      if (tool === 'brush') {
        const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
        if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('brush_line_with_pressure'),
            type: 'brush_line_with_pressure',
            points: [alignedPoint.x, alignedPoint.y, e.evt.pressure],
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('brush_line'),
            type: 'brush_line',
            points: [alignedPoint.x, alignedPoint.y],
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        }
        return;
      }

      if (tool === 'eraser') {
        const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
        if (selectedEntity.bufferRenderer.state && selectedEntity.bufferRenderer.hasBuffer()) {
          selectedEntity.bufferRenderer.commitBuffer();
        }
        if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('eraser_line_with_pressure'),
            type: 'eraser_line_with_pressure',
            points: [alignedPoint.x, alignedPoint.y],
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('eraser_line'),
            type: 'eraser_line',
            points: [alignedPoint.x, alignedPoint.y],
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        }
        return;
      }
    } finally {
      this.render();
    }
  };

  onStagePointerDown = async (e: KonvaEventObject<PointerEvent>) => {
    try {
      this.$lastPointerType.set(e.evt.pointerType);

      if (!this.getCanDraw()) {
        return;
      }

      const isMouseDown = getIsPrimaryMouseDown(e);
      this.$isMouseDown.set(isMouseDown);

      this.syncCursorPositions();
      const cursorPos = this.$cursorPos.get();
      const tool = this.$tool.get();
      const settings = this.manager.stateApi.getSettings();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (!cursorPos || !isMouseDown || !selectedEntity?.$isInteractable.get()) {
        return;
      }

      const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);

      if (tool === 'brush') {
        if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
          const lastLinePoint = getLastPointOfLastLineWithPressure(
            selectedEntity.state.objects,
            'brush_line_with_pressure'
          );
          const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);
          if (selectedEntity.bufferRenderer.hasBuffer()) {
            selectedEntity.bufferRenderer.commitBuffer();
          }
          let points: number[];
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            points = [
              lastLinePoint.x,
              lastLinePoint.y,
              lastLinePoint.pressure,
              alignedPoint.x,
              alignedPoint.y,
              e.evt.pressure,
            ];
          } else {
            points = [alignedPoint.x, alignedPoint.y, e.evt.pressure];
          }
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('brush_line_with_pressure'),
            type: 'brush_line_with_pressure',
            points,
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          const lastLinePoint = getLastPointOfLastLine(selectedEntity.state.objects, 'brush_line');
          const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);

          if (selectedEntity.bufferRenderer.hasBuffer()) {
            selectedEntity.bufferRenderer.commitBuffer();
          }

          let points: number[];
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            points = [lastLinePoint.x, lastLinePoint.y, alignedPoint.x, alignedPoint.y];
          } else {
            points = [alignedPoint.x, alignedPoint.y];
          }

          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('brush_line'),
            type: 'brush_line',
            points,
            strokeWidth: settings.brushWidth,
            color: this.manager.stateApi.getCurrentColor(),
            clip: this.getClip(selectedEntity.state),
          });
        }
      }

      if (tool === 'eraser') {
        if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
          const lastLinePoint = getLastPointOfLastLineWithPressure(
            selectedEntity.state.objects,
            'eraser_line_with_pressure'
          );
          const alignedPoint = alignCoordForTool(normalizedPoint, settings.eraserWidth);
          if (selectedEntity.bufferRenderer.hasBuffer()) {
            selectedEntity.bufferRenderer.commitBuffer();
          }
          let points: number[];
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            points = [
              lastLinePoint.x,
              lastLinePoint.y,
              lastLinePoint.pressure,
              alignedPoint.x,
              alignedPoint.y,
              e.evt.pressure,
            ];
          } else {
            points = [alignedPoint.x, alignedPoint.y, e.evt.pressure];
          }
          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('eraser_line_with_pressure'),
            type: 'eraser_line_with_pressure',
            points,
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        } else {
          const lastLinePoint = getLastPointOfLastLine(selectedEntity.state.objects, 'eraser_line');
          const alignedPoint = alignCoordForTool(normalizedPoint, settings.eraserWidth);

          if (selectedEntity.bufferRenderer.hasBuffer()) {
            selectedEntity.bufferRenderer.commitBuffer();
          }

          let points: number[];
          if (e.evt.shiftKey && lastLinePoint) {
            // Create a straight line from the last line point
            points = [lastLinePoint.x, lastLinePoint.y, alignedPoint.x, alignedPoint.y];
          } else {
            points = [alignedPoint.x, alignedPoint.y];
          }

          await selectedEntity.bufferRenderer.setBuffer({
            id: getPrefixedId('eraser_line'),
            type: 'eraser_line',
            points,
            strokeWidth: settings.eraserWidth,
            clip: this.getClip(selectedEntity.state),
          });
        }
      }

      if (tool === 'rect') {
        if (selectedEntity.bufferRenderer.hasBuffer()) {
          selectedEntity.bufferRenderer.commitBuffer();
        }
        await selectedEntity.bufferRenderer.setBuffer({
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

  onStagePointerUp = (e: KonvaEventObject<PointerEvent>) => {
    try {
      this.$lastPointerType.set(e.evt.pointerType);

      if (!this.getCanDraw()) {
        return;
      }

      const tool = this.$tool.get();
      const settings = this.manager.stateApi.getSettings();

      if (tool === 'colorPicker') {
        const color = this.$colorUnderCursor.get();
        if (color) {
          this.manager.stateApi.setColor({ ...settings.color, ...color });
        }
        return;
      }

      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
      if (!selectedEntity?.$isInteractable.get()) {
        return;
      }

      if (tool === 'brush') {
        if (
          (selectedEntity.bufferRenderer.state?.type === 'brush_line' ||
            selectedEntity.bufferRenderer.state?.type === 'brush_line_with_pressure') &&
          selectedEntity.bufferRenderer.hasBuffer()
        ) {
          selectedEntity.bufferRenderer.commitBuffer();
        } else {
          selectedEntity.bufferRenderer.clearBuffer();
        }
      }

      if (tool === 'eraser') {
        if (
          (selectedEntity.bufferRenderer.state?.type === 'eraser_line' ||
            selectedEntity.bufferRenderer.state?.type === 'eraser_line_with_pressure') &&
          selectedEntity.bufferRenderer.hasBuffer()
        ) {
          selectedEntity.bufferRenderer.commitBuffer();
        } else {
          selectedEntity.bufferRenderer.clearBuffer();
        }
      }

      if (tool === 'rect') {
        if (selectedEntity.bufferRenderer.state?.type === 'rect' && selectedEntity.bufferRenderer.hasBuffer()) {
          selectedEntity.bufferRenderer.commitBuffer();
        } else {
          selectedEntity.bufferRenderer.clearBuffer();
        }
      }
    } finally {
      this.render();
    }
  };

  syncColorUnderCursor = rafThrottle(() => {
    const cursorPos = this.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    const color = getColorAtCoordinate(this.konva.stage, cursorPos.absolute);
    if (color) {
      this.$colorUnderCursor.set(color);
    }
  });

  onStagePointerMove = async (e: KonvaEventObject<PointerEvent>) => {
    try {
      this.$lastPointerType.set(e.evt.pointerType);

      if (!this.getCanDraw()) {
        return;
      }

      this.syncCursorPositions();
      const cursorPos = this.$cursorPos.get();

      if (!cursorPos) {
        return;
      }

      const tool = this.$tool.get();

      if (tool === 'colorPicker') {
        this.syncColorUnderCursor();
      }

      const isMouseDown = this.$isMouseDown.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (!isMouseDown || !selectedEntity?.$isInteractable.get()) {
        return;
      }

      const bufferState = selectedEntity.bufferRenderer.state;

      if (!bufferState) {
        return;
      }

      const settings = this.manager.stateApi.getSettings();

      if (tool === 'brush' && (bufferState.type === 'brush_line' || bufferState.type === 'brush_line_with_pressure')) {
        const lastPoint = getLastPointOfLine(bufferState.points);
        const minDistance = settings.brushWidth * this.config.BRUSH_SPACING_TARGET_SCALE;
        if (!lastPoint || !isDistanceMoreThanMin(cursorPos.relative, lastPoint, minDistance)) {
          return;
        }

        const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);

        if (lastPoint.x === alignedPoint.x && lastPoint.y === alignedPoint.y) {
          // Do not add duplicate points
          return;
        }

        bufferState.points.push(alignedPoint.x, alignedPoint.y);

        if (bufferState.type === 'brush_line_with_pressure') {
          bufferState.points.push(e.evt.pressure);
        }

        await selectedEntity.bufferRenderer.setBuffer(bufferState);
      } else if (
        tool === 'eraser' &&
        (bufferState.type === 'eraser_line' || bufferState.type === 'eraser_line_with_pressure')
      ) {
        const lastPoint = getLastPointOfLine(bufferState.points);
        const minDistance = settings.eraserWidth * this.config.BRUSH_SPACING_TARGET_SCALE;
        if (!lastPoint || !isDistanceMoreThanMin(cursorPos.relative, lastPoint, minDistance)) {
          return;
        }

        const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
        const alignedPoint = alignCoordForTool(normalizedPoint, settings.eraserWidth);

        if (lastPoint.x === alignedPoint.x && lastPoint.y === alignedPoint.y) {
          // Do not add duplicate points
          return;
        }

        bufferState.points.push(alignedPoint.x, alignedPoint.y);

        if (bufferState.type === 'eraser_line_with_pressure') {
          bufferState.points.push(e.evt.pressure);
        }

        await selectedEntity.bufferRenderer.setBuffer(bufferState);
      } else if (tool === 'rect' && bufferState.type === 'rect') {
        const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
        const alignedPoint = floorCoord(normalizedPoint);
        bufferState.rect.width = Math.round(alignedPoint.x - bufferState.rect.x);
        bufferState.rect.height = Math.round(alignedPoint.y - bufferState.rect.y);
        await selectedEntity.bufferRenderer.setBuffer(bufferState);
      } else {
        selectedEntity?.bufferRenderer.clearBuffer();
      }
    } finally {
      this.render();
    }
  };

  onStagePointerLeave = (e: PointerEvent) => {
    try {
      this.$lastPointerType.set(e.pointerType);
      this.$cursorPos.set(null);

      if (!this.getCanDraw()) {
        return;
      }

      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (
        selectedEntity &&
        selectedEntity.bufferRenderer.state?.type !== 'rect' &&
        selectedEntity.bufferRenderer.hasBuffer()
      ) {
        selectedEntity.bufferRenderer.commitBuffer();
      }
    } finally {
      this.render();
    }
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

  /**
   * Commit the buffer on window pointer up.
   *
   * The user may start drawing inside the stage and then release the mouse button outside of the stage. To prevent
   * whatever the user was drawing from being lost, or ending up with stale state, we need to commit the buffer
   * on window pointer up.
   */
  onWindowPointerUp = (_: PointerEvent) => {
    try {
      this.$isMouseDown.set(false);
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (selectedEntity && selectedEntity.bufferRenderer.hasBuffer() && !this.manager.$isBusy.get()) {
        selectedEntity.bufferRenderer.commitBuffer();
      }
    } finally {
      this.render();
    }
  };

  /**
   * We want to reset any "quick-switch" tool selection on window blur. Fixes an issue where you alt-tab out of the app
   * and the color picker tool is still active when you come back.
   */
  onWindowBlur = () => {
    this.revertToolBuffer();
  };

  onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) {
      return;
    }

    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }

    if (e.key === KEY_ESCAPE) {
      // Cancel shape drawing on escape
      e.preventDefault();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
      if (
        selectedEntity &&
        !selectedEntity.filterer?.$isFiltering.get() &&
        !selectedEntity.transformer.$isTransforming.get() &&
        selectedEntity.bufferRenderer.hasBuffer()
      ) {
        selectedEntity.bufferRenderer.clearBuffer();
      }
      return;
    }

    if (e.key === KEY_SPACE) {
      // Select the view tool on space key down
      e.preventDefault();
      this.$toolBuffer.set(this.$tool.get());
      this.$tool.set('view');
      this.manager.stateApi.$spaceKey.set(true);
      this.$cursorPos.set(null);
      return;
    }

    if (e.key === KEY_ALT) {
      // Select the color picker on alt key down
      e.preventDefault();
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

    if (e.key === KEY_SPACE) {
      // Revert the tool to the previous tool on space key up
      e.preventDefault();
      this.revertToolBuffer();
      this.manager.stateApi.$spaceKey.set(false);
      return;
    }

    if (e.key === KEY_ALT) {
      // Revert the tool to the previous tool on alt key up
      e.preventDefault();
      this.revertToolBuffer();
      return;
    }
  };

  revertToolBuffer = () => {
    const toolBuffer = this.$toolBuffer.get();
    if (toolBuffer) {
      this.$tool.set(toolBuffer);
      this.$toolBuffer.set(null);
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
      $tool: this.$tool.get(),
      $toolBuffer: this.$toolBuffer.get(),
      $isMouseDown: this.$isMouseDown.get(),
      $cursorPos: this.$cursorPos.get(),
      $colorUnderCursor: this.$colorUnderCursor.get(),
      brushToolPreview: this.brushToolPreview.repr(),
      eraserToolPreview: this.eraserToolPreview.repr(),
      colorPickerToolPreview: this.colorPickerToolPreview.repr(),
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };
}
