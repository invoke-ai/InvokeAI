import { rgbaColorToString, rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import type { CanvasPreviewModule } from 'features/controlLayers/konva/CanvasPreviewModule';
import {
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
  BRUSH_SPACING_TARGET_SCALE,
} from 'features/controlLayers/konva/constants';
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

export class CanvasToolModule extends CanvasModuleABC {
  readonly type = 'tool';
  static readonly COLOR_PICKER_RADIUS = 25;
  static readonly COLOR_PICKER_THICKNESS = 15;
  static readonly COLOR_PICKER_CROSSHAIR_SPACE = 5;
  static readonly COLOR_PICKER_CROSSHAIR_INNER_THICKNESS = 1.5;
  static readonly COLOR_PICKER_CROSSHAIR_OUTER_THICKNESS = 3;
  static readonly COLOR_PICKER_CROSSHAIR_SIZE = 10;

  id: string;
  path: string[];
  parent: CanvasPreviewModule;
  manager: CanvasManager;
  log: Logger;

  konva: {
    stage: Konva.Stage;
    group: Konva.Group;
    brush: {
      group: Konva.Group;
      fillCircle: Konva.Circle;
      innerBorder: Konva.Ring;
      outerBorder: Konva.Ring;
    };
    eraser: {
      group: Konva.Group;
      fillCircle: Konva.Circle;
      innerBorder: Konva.Ring;
      outerBorder: Konva.Ring;
    };
    colorPicker: {
      group: Konva.Group;
      newColor: Konva.Ring;
      oldColor: Konva.Arc;
      innerBorder: Konva.Ring;
      outerBorder: Konva.Ring;
      crosshairNorthInner: Konva.Line;
      crosshairNorthOuter: Konva.Line;
      crosshairEastInner: Konva.Line;
      crosshairEastOuter: Konva.Line;
      crosshairSouthInner: Konva.Line;
      crosshairSouthOuter: Konva.Line;
      crosshairWestInner: Konva.Line;
      crosshairWestOuter: Konva.Line;
    };
  };

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  constructor(parent: CanvasPreviewModule) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug('Creating tool module');

    this.konva = {
      stage: this.manager.stage.konva.stage,
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      brush: {
        group: new Konva.Group({ name: `${this.type}:brush_group`, listening: false }),
        fillCircle: new Konva.Circle({
          name: `${this.type}:brush_fill_circle`,
          listening: false,
          strokeEnabled: false,
        }),
        innerBorder: new Konva.Ring({
          name: `${this.type}:brush_inner_border_ring`,
          listening: false,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_INNER_COLOR,
          strokeEnabled: false,
        }),
        outerBorder: new Konva.Ring({
          name: `${this.type}:brush_outer_border_ring`,
          listening: false,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_OUTER_COLOR,
          strokeEnabled: false,
        }),
      },
      eraser: {
        group: new Konva.Group({ name: `${this.type}:eraser_group`, listening: false }),
        fillCircle: new Konva.Circle({
          name: `${this.type}:eraser_fill_circle`,
          listening: false,
          strokeEnabled: false,
          fill: 'white',
          globalCompositeOperation: 'destination-out',
        }),
        innerBorder: new Konva.Ring({
          name: `${this.type}:eraser_inner_border_ring`,
          listening: false,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_INNER_COLOR,
          strokeEnabled: false,
        }),
        outerBorder: new Konva.Ring({
          name: `${this.type}:eraser_outer_border_ring`,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_OUTER_COLOR,
          strokeEnabled: false,
        }),
      },
      colorPicker: {
        group: new Konva.Group({ name: `${this.type}:color_picker_group`, listening: false }),
        newColor: new Konva.Ring({
          name: `${this.type}:color_picker_new_color_ring`,
          innerRadius: 0,
          outerRadius: 0,
          strokeEnabled: false,
        }),
        oldColor: new Konva.Arc({
          name: `${this.type}:color_picker_old_color_arc`,
          innerRadius: 0,
          outerRadius: 0,
          angle: 180,
          strokeEnabled: false,
        }),
        innerBorder: new Konva.Ring({
          name: `${this.type}:color_picker_inner_border_ring`,
          listening: false,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_INNER_COLOR,
          strokeEnabled: false,
        }),
        outerBorder: new Konva.Ring({
          name: `${this.type}:color_picker_outer_border_ring`,
          innerRadius: 0,
          outerRadius: 0,
          fill: BRUSH_BORDER_OUTER_COLOR,
          strokeEnabled: false,
        }),
        crosshairNorthInner: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_north1_line`,
          stroke: BRUSH_BORDER_INNER_COLOR,
        }),
        crosshairNorthOuter: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_north2_line`,
          stroke: BRUSH_BORDER_OUTER_COLOR,
        }),
        crosshairEastInner: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_east1_line`,
          stroke: BRUSH_BORDER_INNER_COLOR,
        }),
        crosshairEastOuter: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_east2_line`,
          stroke: BRUSH_BORDER_OUTER_COLOR,
        }),
        crosshairSouthInner: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_south1_line`,
          stroke: BRUSH_BORDER_INNER_COLOR,
        }),
        crosshairSouthOuter: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_south2_line`,
          stroke: BRUSH_BORDER_OUTER_COLOR,
        }),
        crosshairWestInner: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_west1_line`,
          stroke: BRUSH_BORDER_INNER_COLOR,
        }),
        crosshairWestOuter: new Konva.Line({
          name: `${this.type}:color_picker_crosshair_west2_line`,
          stroke: BRUSH_BORDER_OUTER_COLOR,
        }),
      },
    };
    this.konva.brush.group.add(this.konva.brush.fillCircle, this.konva.brush.innerBorder, this.konva.brush.outerBorder);
    this.konva.group.add(this.konva.brush.group);

    this.konva.eraser.group.add(
      this.konva.eraser.fillCircle,
      this.konva.eraser.innerBorder,
      this.konva.eraser.outerBorder
    );
    this.konva.group.add(this.konva.eraser.group);

    this.konva.colorPicker.group.add(
      this.konva.colorPicker.newColor,
      this.konva.colorPicker.oldColor,
      this.konva.colorPicker.innerBorder,
      this.konva.colorPicker.outerBorder,
      this.konva.colorPicker.crosshairNorthOuter,
      this.konva.colorPicker.crosshairNorthInner,
      this.konva.colorPicker.crosshairEastOuter,
      this.konva.colorPicker.crosshairEastInner,
      this.konva.colorPicker.crosshairSouthOuter,
      this.konva.colorPicker.crosshairSouthInner,
      this.konva.colorPicker.crosshairWestOuter,
      this.konva.colorPicker.crosshairWestInner
    );
    this.konva.group.add(this.konva.colorPicker.group);

    this.subscriptions.add(this.manager.stateApi.$stageAttrs.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.$toolState.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.$tool.listen(this.render));

    const cleanupListeners = this.setEventListeners();

    this.subscriptions.add(cleanupListeners);
  }

  setToolVisibility = (tool: Tool) => {
    this.konva.brush.group.visible(tool === 'brush');
    this.konva.eraser.group.visible(tool === 'eraser');
    this.konva.colorPicker.group.visible(tool === 'colorPicker');
  };

  render = () => {
    const stage = this.manager.stage;
    const renderedEntityCount = this.manager.stateApi.getRenderedEntityCount();
    const toolState = this.manager.stateApi.getToolState();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const cursorPos = this.manager.stateApi.$lastCursorPos.get();
    const isMouseDown = this.manager.stateApi.$isMouseDown.get();
    const tool = this.manager.stateApi.$tool.get();

    const isDrawable = selectedEntity && selectedEntity.state.isEnabled && isDrawableEntity(selectedEntity.state);

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

    stage.setIsDraggable(tool === 'view');

    if (!cursorPos || renderedEntityCount === 0) {
      // We can bail early if the mouse isn't over the stage or there are no layers
      this.konva.group.visible(false);
    } else {
      this.konva.group.visible(true);

      // No need to render the brush preview if the cursor position or color is missing
      if (cursorPos && tool === 'brush') {
        const brushPreviewFill = this.manager.stateApi.getBrushPreviewFill();
        const alignedCursorPos = alignCoordForTool(cursorPos, toolState.brush.width);
        const onePixel = this.manager.stage.getScaledPixels(1);
        const twoPixels = this.manager.stage.getScaledPixels(2);
        const radius = toolState.brush.width / 2;

        // The circle is scaled
        this.konva.brush.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: rgbaColorToString(brushPreviewFill),
        });

        // But the borders are in screen-pixels
        this.konva.brush.innerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: radius,
          outerRadius: radius + onePixel,
        });
        this.konva.brush.outerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: radius + onePixel,
          outerRadius: radius + twoPixels,
        });
      } else if (cursorPos && tool === 'eraser') {
        const alignedCursorPos = alignCoordForTool(cursorPos, toolState.eraser.width);
        const onePixel = this.manager.stage.getScaledPixels(1);
        const twoPixels = this.manager.stage.getScaledPixels(2);
        const radius = toolState.eraser.width / 2;

        // The circle is scaled
        this.konva.eraser.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: 'white',
        });

        // But the borders are in screen-pixels
        this.konva.eraser.innerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: radius,
          outerRadius: radius + onePixel,
        });
        this.konva.eraser.outerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: radius + onePixel,
          outerRadius: radius + twoPixels,
        });
      } else if (cursorPos && tool === 'colorPicker') {
        const colorUnderCursor = this.manager.stateApi.$colorUnderCursor.get();
        const colorPickerInnerRadius = this.manager.stage.getScaledPixels(CanvasToolModule.COLOR_PICKER_RADIUS);
        const colorPickerOuterRadius = this.manager.stage.getScaledPixels(
          CanvasToolModule.COLOR_PICKER_RADIUS + CanvasToolModule.COLOR_PICKER_THICKNESS
        );
        const onePixel = this.manager.stage.getScaledPixels(1);
        const twoPixels = this.manager.stage.getScaledPixels(2);

        this.konva.colorPicker.newColor.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          fill: rgbColorToString(colorUnderCursor),
          innerRadius: colorPickerInnerRadius,
          outerRadius: colorPickerOuterRadius,
        });
        this.konva.colorPicker.oldColor.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          fill: rgbColorToString(toolState.fill),
          innerRadius: colorPickerInnerRadius,
          outerRadius: colorPickerOuterRadius,
        });
        this.konva.colorPicker.innerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: colorPickerOuterRadius,
          outerRadius: colorPickerOuterRadius + onePixel,
        });
        this.konva.colorPicker.outerBorder.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          innerRadius: colorPickerOuterRadius + onePixel,
          outerRadius: colorPickerOuterRadius + twoPixels,
        });

        const size = this.manager.stage.getScaledPixels(CanvasToolModule.COLOR_PICKER_CROSSHAIR_SIZE);
        const space = this.manager.stage.getScaledPixels(CanvasToolModule.COLOR_PICKER_CROSSHAIR_SPACE);
        const innerThickness = this.manager.stage.getScaledPixels(
          CanvasToolModule.COLOR_PICKER_CROSSHAIR_INNER_THICKNESS
        );
        const outerThickness = this.manager.stage.getScaledPixels(
          CanvasToolModule.COLOR_PICKER_CROSSHAIR_OUTER_THICKNESS
        );
        this.konva.colorPicker.crosshairNorthOuter.setAttrs({
          strokeWidth: outerThickness,
          points: [cursorPos.x, cursorPos.y - size, cursorPos.x, cursorPos.y - space],
        });
        this.konva.colorPicker.crosshairNorthInner.setAttrs({
          strokeWidth: innerThickness,
          points: [cursorPos.x, cursorPos.y - size, cursorPos.x, cursorPos.y - space],
        });
        this.konva.colorPicker.crosshairEastOuter.setAttrs({
          strokeWidth: outerThickness,
          points: [cursorPos.x + space, cursorPos.y, cursorPos.x + size, cursorPos.y],
        });
        this.konva.colorPicker.crosshairEastInner.setAttrs({
          strokeWidth: innerThickness,
          points: [cursorPos.x + space, cursorPos.y, cursorPos.x + size, cursorPos.y],
        });
        this.konva.colorPicker.crosshairSouthOuter.setAttrs({
          strokeWidth: outerThickness,
          points: [cursorPos.x, cursorPos.y + space, cursorPos.x, cursorPos.y + size],
        });
        this.konva.colorPicker.crosshairSouthInner.setAttrs({
          strokeWidth: innerThickness,
          points: [cursorPos.x, cursorPos.y + space, cursorPos.x, cursorPos.y + size],
        });
        this.konva.colorPicker.crosshairWestOuter.setAttrs({
          strokeWidth: outerThickness,
          points: [cursorPos.x - space, cursorPos.y, cursorPos.x - size, cursorPos.y],
        });
        this.konva.colorPicker.crosshairWestInner.setAttrs({
          strokeWidth: innerThickness,
          points: [cursorPos.x - space, cursorPos.y, cursorPos.x - size, cursorPos.y],
        });
      }

      this.setToolVisibility(tool);
    }
  };

  syncLastCursorPos = (): Coordinate | null => {
    const pos = getScaledCursorPosition(this.konva.stage);
    if (!pos) {
      return null;
    }
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
      const isDrawable = selectedEntity?.state.isEnabled;
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
    const isDrawable = selectedEntity?.state.isEnabled;
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
      const isDrawable = selectedEntity?.state.isEnabled;
      if (pos && isDrawable && !this.manager.stateApi.$spaceKey.get() && getIsPrimaryMouseDown(e)) {
        if (tool === 'brush') {
          const drawingBuffer = selectedEntity.adapter.renderer.bufferState;
          if (drawingBuffer) {
            if (drawingBuffer.type === 'brush_line') {
              const lastPoint = getLastPointOfLine(drawingBuffer.points);
              const minDistance = toolState.brush.width * BRUSH_SPACING_TARGET_SCALE;
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
              const minDistance = toolState.eraser.width * BRUSH_SPACING_TARGET_SCALE;
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
    const isDrawable = selectedEntity?.state.isEnabled;
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
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
    };
  };

  destroy = () => {
    this.log.debug('Destroying tool module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };

  getLoggingContext = () => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
