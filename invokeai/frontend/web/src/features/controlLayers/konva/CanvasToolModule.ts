import type { SerializableObject } from 'common/types';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasPreviewModule } from 'features/controlLayers/konva/CanvasPreviewModule';
import { BRUSH_BORDER_INNER_COLOR, BRUSH_BORDER_OUTER_COLOR } from 'features/controlLayers/konva/constants';
import { alignCoordForTool, getPrefixedId } from 'features/controlLayers/konva/util';
import type { Tool } from 'features/controlLayers/store/types';
import { isDrawableEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasToolModule {
  readonly type = 'tool_preview';
  static readonly COLOR_PICKER_RADIUS = 25;
  static readonly COLOR_PICKER_THICKNESS = 15;

  id: string;
  path: string[];
  parent: CanvasPreviewModule;
  manager: CanvasManager;
  log: Logger;

  konva: {
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
    };
  };

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  constructor(parent: CanvasPreviewModule) {
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.konva = {
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
      this.konva.colorPicker.outerBorder
    );
    this.konva.group.add(this.konva.colorPicker.group);

    this.subscriptions.add(
      this.manager.stateApi.$stageAttrs.listen(() => {
        this.render();
      })
    );

    this.subscriptions.add(
      this.manager.stateApi.$toolState.listen(() => {
        this.render();
      })
    );
  }

  destroy = () => {
    for (const cleanup of this.subscriptions) {
      cleanup();
    }
    this.konva.group.destroy();
  };

  scaleTool = () => {
    const toolState = this.manager.stateApi.getToolState();
    const onePixel = this.manager.stage.getScaledPixels(1);
    const twoPixels = this.manager.stage.getScaledPixels(2);

    const brushRadius = toolState.brush.width / 2;
    this.konva.brush.innerBorder.innerRadius(brushRadius);
    this.konva.brush.innerBorder.outerRadius(brushRadius + onePixel);

    this.konva.brush.outerBorder.innerRadius(brushRadius + onePixel);
    this.konva.brush.outerBorder.outerRadius(brushRadius + twoPixels);

    const eraserRadius = toolState.eraser.width / 2;
    this.konva.eraser.innerBorder.innerRadius(eraserRadius);
    this.konva.eraser.innerBorder.outerRadius(eraserRadius + onePixel);

    this.konva.eraser.outerBorder.innerRadius(eraserRadius + onePixel);
    this.konva.eraser.outerBorder.outerRadius(eraserRadius + twoPixels);

    const colorPickerInnerRadius = this.manager.stage.getScaledPixels(CanvasToolModule.COLOR_PICKER_RADIUS);
    const colorPickerOuterRadius = this.manager.stage.getScaledPixels(
      CanvasToolModule.COLOR_PICKER_RADIUS + CanvasToolModule.COLOR_PICKER_THICKNESS
    );
    this.konva.colorPicker.oldColor.innerRadius(colorPickerInnerRadius);
    this.konva.colorPicker.oldColor.outerRadius(colorPickerOuterRadius);

    this.konva.colorPicker.newColor.innerRadius(colorPickerInnerRadius);
    this.konva.colorPicker.newColor.outerRadius(colorPickerOuterRadius);

    this.konva.colorPicker.innerBorder.innerRadius(colorPickerOuterRadius);
    this.konva.colorPicker.innerBorder.outerRadius(colorPickerOuterRadius + onePixel);

    this.konva.colorPicker.outerBorder.innerRadius(colorPickerOuterRadius + onePixel);
    this.konva.colorPicker.outerBorder.outerRadius(colorPickerOuterRadius + twoPixels);
  };

  setToolVisibility = (tool: Tool) => {
    this.konva.brush.group.visible(tool === 'brush');
    this.konva.eraser.group.visible(tool === 'eraser');
    this.konva.colorPicker.group.visible(tool === 'colorPicker');
  };

  render() {
    const stage = this.manager.stage;
    const renderedEntityCount: number = 1; // TODO(psyche): this.manager should be renderable entity count
    const toolState = this.manager.stateApi.getToolState();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const cursorPos = this.manager.stateApi.$lastCursorPos.get();
    const isDrawing = this.manager.stateApi.$isDrawing.get();
    const isMouseDown = this.manager.stateApi.$isMouseDown.get();
    const colorUnderCursor = this.manager.stateApi.$colorUnderCursor.get();

    const tool = toolState.selected;

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

    if (!cursorPos || renderedEntityCount === 0 || !isDrawable) {
      // We can bail early if the mouse isn't over the stage or there are no layers
      this.konva.group.visible(false);
    } else {
      this.konva.group.visible(true);

      // No need to render the brush preview if the cursor position or color is missing
      if (cursorPos && tool === 'brush') {
        const brushPreviewFill = this.manager.stateApi.getBrushPreviewFill();
        const alignedCursorPos = alignCoordForTool(cursorPos, toolState.brush.width);

        // Update the fill circle
        const radius = toolState.brush.width / 2;

        this.konva.brush.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: isDrawing ? '' : rgbaColorToString(brushPreviewFill),
        });
        this.konva.brush.innerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
        this.konva.brush.outerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
      } else if (cursorPos && tool === 'eraser') {
        const alignedCursorPos = alignCoordForTool(cursorPos, toolState.eraser.width);
        // Update the fill circle
        const radius = toolState.eraser.width / 2;
        this.konva.eraser.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: 'white',
        });
        this.konva.eraser.innerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
        this.konva.eraser.outerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
      } else if (cursorPos && colorUnderCursor) {
        this.konva.colorPicker.newColor.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          fill: rgbaColorToString(colorUnderCursor),
        });
        this.konva.colorPicker.oldColor.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          fill: rgbaColorToString(toolState.fill),
        });
        this.konva.colorPicker.innerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
        this.konva.colorPicker.outerBorder.setAttrs({ x: cursorPos.x, y: cursorPos.y });
      }

      this.scaleTool();
      this.setToolVisibility(tool);
    }
  }

  getLoggingContext = (): SerializableObject => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
