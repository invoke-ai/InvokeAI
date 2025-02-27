import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import {
  alignCoordForTool,
  getLastPointOfLastLine,
  getLastPointOfLastLineWithPressure,
  getLastPointOfLine,
  getPrefixedId,
  isDistanceMoreThanMin,
  offsetCoord,
} from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Logger } from 'roarr';

type CanvasEraserToolModuleConfig = {
  /**
   * The inner border color for the eraser tool preview.
   */
  BORDER_INNER_COLOR: string;
  /**
   * The outer border color for the eraser tool preview.
   */
  BORDER_OUTER_COLOR: string;
};

const DEFAULT_CONFIG: CanvasEraserToolModuleConfig = {
  BORDER_INNER_COLOR: 'rgba(0,0,0,1)',
  BORDER_OUTER_COLOR: 'rgba(255,255,255,0.8)',
};

export class CanvasEraserToolModule extends CanvasModuleBase {
  readonly type = 'eraser_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasEraserToolModuleConfig = DEFAULT_CONFIG;

  konva: {
    group: Konva.Group;
    cutoutCircle: Konva.Circle;
    innerBorder: Konva.Ring;
    outerBorder: Konva.Ring;
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
      group: new Konva.Group({ name: `${this.type}:eraser_group`, listening: false }),
      cutoutCircle: new Konva.Circle({
        name: `${this.type}:eraser_cutout_circle`,
        listening: false,
        strokeEnabled: false,
        // The fill is used only to erase what is underneath it, so its color doesn't matter - just needs to be opaque
        fill: 'white',
        globalCompositeOperation: 'destination-out',
        perfectDrawEnabled: false,
      }),
      innerBorder: new Konva.Ring({
        name: `${this.type}:eraser_inner_border_ring`,
        listening: false,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.BORDER_INNER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      outerBorder: new Konva.Ring({
        listening: false,
        name: `${this.type}:eraser_outer_border_ring`,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.BORDER_OUTER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.cutoutCircle, this.konva.innerBorder, this.konva.outerBorder);
  }

  syncCursorStyle = () => {
    this.manager.stage.setCursor('none');
  };

  render = () => {
    if (this.parent.$tool.get() !== 'eraser') {
      this.setVisibility(false);
      return;
    }

    if (!this.parent.getCanDraw()) {
      this.setVisibility(false);
      return;
    }

    const cursorPos = this.parent.$cursorPos.get();

    if (!cursorPos) {
      this.setVisibility(false);
      return;
    }

    const isPrimaryPointerDown = this.parent.$isPrimaryPointerDown.get();
    const lastPointerType = this.parent.$lastPointerType.get();

    if (lastPointerType !== 'mouse' && isPrimaryPointerDown) {
      this.setVisibility(false);
      return;
    }

    this.setVisibility(true);

    const settings = this.manager.stateApi.getSettings();
    const alignedCursorPos = alignCoordForTool(cursorPos.relative, settings.eraserWidth);
    const radius = settings.eraserWidth / 2;

    // The circle is scaled
    this.konva.cutoutCircle.setAttrs({
      x: alignedCursorPos.x,
      y: alignedCursorPos.y,
      radius,
    });

    // But the borders are in screen-pixels
    const onePixel = this.manager.stage.unscale(1);
    const twoPixels = this.manager.stage.unscale(2);

    this.konva.innerBorder.setAttrs({
      x: cursorPos.relative.x,
      y: cursorPos.relative.y,
      innerRadius: radius,
      outerRadius: radius + onePixel,
    });
    this.konva.outerBorder.setAttrs({
      x: cursorPos.relative.x,
      y: cursorPos.relative.y,
      innerRadius: radius + onePixel,
      outerRadius: radius + twoPixels,
    });
  };

  setVisibility = (visible: boolean) => {
    this.konva.group.visible(visible);
  };

  /**
   * Handles the pointer enter event on the stage, when the eraser tool is active. This may create a new eraser line if
   * the mouse is down as the cursor enters the stage.
   *
   * The tool module will pass on the event to this method if the tool is 'eraser', after doing any necessary checks
   * and non-tool-specific handling.
   *
   * @param e The Konva event object.
   */
  onStagePointerEnter = async (e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();
    const isPrimaryPointerDown = this.parent.$isPrimaryPointerDown.get();
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!cursorPos || !isPrimaryPointerDown || !selectedEntity) {
      /**
       * Can't do anything without:
       * - A cursor position: the cursor is not on the stage
       * - The mouse is down: the user is not drawing
       * - A selected entity: there is no entity to draw on
       */
      return;
    }

    const settings = this.manager.stateApi.getSettings();
    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);
    const alignedPoint = alignCoordForTool(normalizedPoint, settings.brushWidth);

    if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
      // If the pen is down and pressure sensitivity is enabled, add the point with pressure
      await selectedEntity.bufferRenderer.setBuffer({
        id: getPrefixedId('eraser_line_with_pressure'),
        type: 'eraser_line_with_pressure',
        points: [alignedPoint.x, alignedPoint.y, e.evt.pressure],
        strokeWidth: settings.eraserWidth,
        clip: this.parent.getClip(selectedEntity.state),
      });
    } else {
      // Else, add the point without pressure
      await selectedEntity.bufferRenderer.setBuffer({
        id: getPrefixedId('eraser_line'),
        type: 'eraser_line',
        points: [alignedPoint.x, alignedPoint.y],
        strokeWidth: settings.eraserWidth,
        clip: this.parent.getClip(selectedEntity.state),
      });
    }
  };

  /**
   * Handles the pointer down event on the stage, when the eraser tool is active. If the shift key is held, this will
   * create a straight line from the last point of the last line to the current point. Else, it will create a new line
   * with the current point.
   *
   * The tool module will pass on the event to this method if the tool is 'eraser', after doing any necessary checks
   * and non-tool-specific handling.
   *
   * @param e The Konva event object.
   */
  onStagePointerDown = async (e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();
    const isPrimaryPointerDown = this.parent.$isPrimaryPointerDown.get();
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!cursorPos || !selectedEntity || !isPrimaryPointerDown) {
      /**
       * Can't do anything without:
       * - A cursor position: the cursor is not on the stage
       * - The mouse is down: the user is not drawing
       * - A selected entity: there is no entity to draw on
       */
      return;
    }

    const settings = this.manager.stateApi.getSettings();

    const normalizedPoint = offsetCoord(cursorPos.relative, selectedEntity.state.position);

    if (e.evt.pointerType === 'pen' && settings.pressureSensitivity) {
      // We need to get the last point of the last line to create a straight line if shift is held
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
        // Create a new line with the current point
        points = [alignedPoint.x, alignedPoint.y, e.evt.pressure];
      }
      await selectedEntity.bufferRenderer.setBuffer({
        id: getPrefixedId('eraser_line_with_pressure'),
        type: 'eraser_line_with_pressure',
        points,
        strokeWidth: settings.eraserWidth,
        clip: this.parent.getClip(selectedEntity.state),
      });
    } else {
      // We need to get the last point of the last line to create a straight line if shift is held
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
        // Create a new line with the current point
        points = [alignedPoint.x, alignedPoint.y];
      }

      await selectedEntity.bufferRenderer.setBuffer({
        id: getPrefixedId('eraser_line'),
        type: 'eraser_line',
        points,
        strokeWidth: settings.eraserWidth,
        clip: this.parent.getClip(selectedEntity.state),
      });
    }
  };

  /**
   * Handles the pointer up event on the stage, when the eraser tool is active. This handles finalizing the eraser line
   * that was being drawn (if any).
   *
   * The tool module will pass on the event to this method if the tool is 'eraser', after doing any necessary checks
   * and non-tool-specific handling.
   *
   * @param e The Konva event object.
   */
  onStagePointerUp = (_e: KonvaEventObject<PointerEvent>) => {
    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();
    if (!selectedEntity) {
      return;
    }

    if (
      (selectedEntity.bufferRenderer.state?.type === 'eraser_line' ||
        selectedEntity.bufferRenderer.state?.type === 'eraser_line_with_pressure') &&
      selectedEntity.bufferRenderer.hasBuffer()
    ) {
      selectedEntity.bufferRenderer.commitBuffer();
    } else {
      selectedEntity.bufferRenderer.clearBuffer();
    }
  };

  /**
   * Handles the pointer move event on the stage, when the brush tool is active. This handles extending the brush line
   * that is being drawn (if any).
   *
   * The tool module will pass on the event to this method if the tool is 'brush', after doing any necessary checks
   * and non-tool-specific handling.
   *
   * @param e The Konva event object.
   */
  onStagePointerMove = async (e: KonvaEventObject<PointerEvent>) => {
    const cursorPos = this.parent.$cursorPos.get();

    if (!cursorPos) {
      return;
    }

    if (!this.parent.$isPrimaryPointerDown.get()) {
      return;
    }

    const selectedEntity = this.manager.stateApi.getSelectedEntityAdapter();

    if (!selectedEntity) {
      return;
    }

    const bufferState = selectedEntity.bufferRenderer.state;

    if (!bufferState) {
      return;
    }

    if (bufferState.type !== 'eraser_line' && bufferState.type !== 'eraser_line_with_pressure') {
      return;
    }
    const settings = this.manager.stateApi.getSettings();

    const lastPoint = getLastPointOfLine(bufferState.points);
    const minDistance = settings.eraserWidth * this.parent.config.BRUSH_SPACING_TARGET_SCALE;
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

    // Add pressure if the pen is down and pressure sensitivity is enabled
    if (bufferState.type === 'eraser_line_with_pressure' && settings.pressureSensitivity) {
      bufferState.points.push(e.evt.pressure);
    }

    await selectedEntity.bufferRenderer.setBuffer(bufferState);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
    };
  };

  destroy = () => {
    this.log.debug('Destroying eraser tool preview module');
    this.konva.group.destroy();
  };
}
