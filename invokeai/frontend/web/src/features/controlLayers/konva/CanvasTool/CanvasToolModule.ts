import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasBboxToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasBboxToolModule';
import { CanvasBrushToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasBrushToolModule';
import { CanvasColorPickerToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasColorPickerToolModule';
import { CanvasEraserToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasEraserToolModule';
import { CanvasMoveToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasMoveToolModule';
import { CanvasRectToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasRectToolModule';
import { CanvasViewToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasViewToolModule';
import {
  calculateNewBrushSizeFromWheelDelta,
  getIsPrimaryMouseDown,
  getPrefixedId,
} from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  Coordinate,
  Tool,
} from 'features/controlLayers/store/types';
import { isRenderableEntityType } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
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

  tools: {
    brush: CanvasBrushToolModule;
    eraser: CanvasEraserToolModule;
    rect: CanvasRectToolModule;
    colorPicker: CanvasColorPickerToolModule;
    bbox: CanvasBboxToolModule;
    view: CanvasViewToolModule;
    move: CanvasMoveToolModule;
  };

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
   * Whether the primary pointer (left mouse, pen, first touch) is currently down on the stage.
   *
   * This is set true when the pointer down is fired on the stage and false when the pointer up is fired anywhere,
   * including outside of the stage. This flag is thus true when the user is actively drawing on the stage.
   *
   * For example, if the pointer down was fired on the stage and the cursor then leaves the stage without a pointer up
   * event, this will still be true. If the cursor then moves back onto the stage, this will still be true.
   *
   * However, if the pointer down was initially fired _outside_ the stage, and the cursor moves onto the stage, this
   * will be false.
   */
  $isPrimaryPointerDown = atom<boolean>(false);
  /**
   * The last cursor position.
   */
  $cursorPos = atom<{ relative: Coordinate; absolute: Coordinate } | null>(null);
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

    this.tools = {
      brush: new CanvasBrushToolModule(this),
      eraser: new CanvasEraserToolModule(this),
      rect: new CanvasRectToolModule(this),
      colorPicker: new CanvasColorPickerToolModule(this),
      bbox: new CanvasBboxToolModule(this),
      view: new CanvasViewToolModule(this),
      move: new CanvasMoveToolModule(this),
    };

    this.konva = {
      stage: this.manager.stage.konva.stage,
      group: new Konva.Group({ name: `${this.type}:group`, listening: true }),
    };

    this.konva.group.add(this.tools.brush.konva.group);
    this.konva.group.add(this.tools.eraser.konva.group);
    this.konva.group.add(this.tools.colorPicker.konva.group);
    this.konva.group.add(this.tools.bbox.konva.group);

    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.render));
    this.subscriptions.add(this.manager.$isBusy.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSettingsSlice, this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.render));
    this.subscriptions.add(
      this.$tool.listen(() => {
        // On tool switch, reset mouse state
        this.manager.tool.$isPrimaryPointerDown.set(false);
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

  syncCursorStyle = () => {
    const stage = this.manager.stage;
    const tool = this.$tool.get();
    const segmentingAdapter = this.manager.stateApi.$segmentingAdapter.get();
    const transformingAdapter = this.manager.stateApi.$transformingAdapter.get();
    const selectedEntityAdapter = this.manager.stateApi.getSelectedEntityAdapter();

    if (this.manager.stage.getIsDragging()) {
      this.tools.view.syncCursorStyle();
    } else if (tool === 'view') {
      this.tools.view.syncCursorStyle();
    } else if (segmentingAdapter) {
      segmentingAdapter.segmentAnything.syncCursorStyle();
    } else if (transformingAdapter) {
      transformingAdapter.transformer.syncCursorStyle();
    } else if (this.manager.stateApi.$isFiltering.get()) {
      stage.setCursor('not-allowed');
    } else if (this.manager.stagingArea.$isStaging.get()) {
      stage.setCursor('not-allowed');
    } else if (tool === 'bbox') {
      this.tools.bbox.syncCursorStyle();
    } else if (tool === 'colorPicker') {
      this.tools.colorPicker.syncCursorStyle();
    } else if (selectedEntityAdapter && isRenderableEntityType(selectedEntityAdapter.entityIdentifier.type)) {
      if (selectedEntityAdapter.$isDisabled.get()) {
        stage.setCursor('not-allowed');
      } else if (selectedEntityAdapter.$isEntityTypeHidden.get()) {
        stage.setCursor('not-allowed');
      } else if (selectedEntityAdapter.$isLocked.get()) {
        stage.setCursor('not-allowed');
      } else if (tool === 'brush') {
        this.tools.brush.syncCursorStyle();
      } else if (tool === 'eraser') {
        this.tools.eraser.syncCursorStyle();
      } else if (tool === 'move') {
        this.tools.move.syncCursorStyle();
      } else if (tool === 'rect') {
        this.tools.rect.syncCursorStyle();
      }
    } else if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      stage.setCursor('not-allowed');
    } else {
      stage.setCursor('not-allowed');
    }
  };

  render = () => {
    this.syncCursorStyle();

    this.tools.brush.render();
    this.tools.eraser.render();
    this.tools.colorPicker.render();
    this.tools.bbox.render();
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

  /**
   * Gets whether the user is allowed to draw on the canvas.
   * - There must be at least one entity rendered on the canvas.
   * - The canvas must not be busy (e.g. transforming, filtering, rasterizing, staging, compositing, segment-anything-ing).
   * - There must be a selected entity.
   * - The selected entity must be interactable (e.g. not hidden, disabled or locked).
   * @returns Whether the user is allowed to draw on the canvas.
   */
  getCanDraw = (): boolean => {
    if (this.manager.stateApi.getRenderedEntityCount() === 0) {
      return false;
    }

    if (this.manager.$isBusy.get()) {
      return false;
    }

    if (this.manager.stage.getIsDragging()) {
      return false;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity) {
      return false;
    }

    if (selectedEntity.$isDisabled.get()) {
      return false;
    }

    if (selectedEntity.$isEntityTypeHidden.get()) {
      return false;
    }

    if (selectedEntity.$isLocked.get()) {
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

      const tool = this.$tool.get();
      const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

      if (selectedEntity?.bufferRenderer.state?.type !== 'rect' && selectedEntity?.bufferRenderer.hasBuffer()) {
        selectedEntity.bufferRenderer.commitBuffer();
        return;
      }

      if (tool === 'brush') {
        await this.tools.brush.onStagePointerEnter(e);
      } else if (tool === 'eraser') {
        await this.tools.eraser.onStagePointerEnter(e);
      }
    } finally {
      this.render();
    }
  };

  onStagePointerDown = async (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    try {
      this.$lastPointerType.set(e.evt.pointerType);

      if (!this.getCanDraw()) {
        return;
      }

      this.$isPrimaryPointerDown.set(getIsPrimaryMouseDown(e));

      this.syncCursorPositions();

      const tool = this.$tool.get();

      if (tool === 'brush') {
        await this.tools.brush.onStagePointerDown(e);
      } else if (tool === 'eraser') {
        await this.tools.eraser.onStagePointerDown(e);
      } else if (tool === 'rect') {
        await this.tools.rect.onStagePointerDown(e);
      }
    } finally {
      this.render();
    }
  };

  onStagePointerUp = (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    try {
      this.$lastPointerType.set(e.evt.pointerType);

      const tool = this.$tool.get();

      if (tool === 'colorPicker') {
        this.tools.colorPicker.onStagePointerUp(e);
      }

      if (!this.getCanDraw()) {
        return;
      }

      if (tool === 'brush') {
        this.tools.brush.onStagePointerUp(e);
      } else if (tool === 'eraser') {
        this.tools.eraser.onStagePointerUp(e);
      } else if (tool === 'rect') {
        this.tools.rect.onStagePointerUp(e);
      }
    } finally {
      this.render();
    }
  };

  onStagePointerMove = async (e: KonvaEventObject<PointerEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    try {
      this.$lastPointerType.set(e.evt.pointerType);
      this.syncCursorPositions();

      const tool = this.$tool.get();

      if (tool === 'colorPicker') {
        this.tools.colorPicker.onStagePointerMove(e);
      }

      if (!this.getCanDraw()) {
        return;
      }

      if (tool === 'brush') {
        await this.tools.brush.onStagePointerMove(e);
      } else if (tool === 'eraser') {
        await this.tools.eraser.onStagePointerMove(e);
      } else if (tool === 'rect') {
        await this.tools.rect.onStagePointerMove(e);
      } else {
        this.manager.stateApi.getSelectedEntityAdapter()?.bufferRenderer.clearBuffer();
      }
    } finally {
      this.render();
    }
  };

  onStagePointerLeave = (e: PointerEvent) => {
    if (e.target !== this.manager.stage.container) {
      return;
    }

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
    if (e.target !== this.konva.stage) {
      return;
    }

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
      this.$isPrimaryPointerDown.set(false);
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
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }

    // Handle nudging - must be before repeat, as we may want to catch repeating keys
    if (this.tools.move.isNudgeKey(e.key)) {
      this.tools.move.nudge(e.key);
    }

    if (e.repeat) {
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
      $isPrimaryPointerDown: this.$isPrimaryPointerDown.get(),
      $cursorPos: this.$cursorPos.get(),
      $lastPointerType: this.$lastPointerType.get(),
      tools: {
        brush: this.tools.brush.repr(),
        eraser: this.tools.eraser.repr(),
        colorPicker: this.tools.colorPicker.repr(),
        rect: this.tools.rect.repr(),
        bbox: this.tools.bbox.repr(),
        view: this.tools.view.repr(),
        move: this.tools.move.repr(),
      },
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };
}
