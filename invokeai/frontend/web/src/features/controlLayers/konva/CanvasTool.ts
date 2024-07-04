import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import {
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
  BRUSH_ERASER_BORDER_WIDTH,
} from 'features/controlLayers/konva/constants';
import { PREVIEW_RECT_ID } from 'features/controlLayers/konva/naming';
import Konva from 'konva';

export class CanvasTool {
  manager: CanvasManager;
  group: Konva.Group;
  brush: {
    group: Konva.Group;
    fillCircle: Konva.Circle;
    innerBorderCircle: Konva.Circle;
    outerBorderCircle: Konva.Circle;
  };
  eraser: {
    group: Konva.Group;
    fillCircle: Konva.Circle;
    innerBorderCircle: Konva.Circle;
    outerBorderCircle: Konva.Circle;
  };
  rect: {
    group: Konva.Group;
    fillRect: Konva.Rect;
  };

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.group = new Konva.Group();

    // Create the brush preview group & circles
    this.brush = {
      group: new Konva.Group(),
      fillCircle: new Konva.Circle({
        listening: false,
        strokeEnabled: false,
      }),
      innerBorderCircle: new Konva.Circle({
        listening: false,
        stroke: BRUSH_BORDER_INNER_COLOR,
        strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
        strokeEnabled: true,
      }),
      outerBorderCircle: new Konva.Circle({
        listening: false,
        stroke: BRUSH_BORDER_OUTER_COLOR,
        strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
        strokeEnabled: true,
      }),
    };
    this.brush.group.add(this.brush.fillCircle);
    this.brush.group.add(this.brush.innerBorderCircle);
    this.brush.group.add(this.brush.outerBorderCircle);
    this.group.add(this.brush.group);

    this.eraser = {
      group: new Konva.Group(),
      fillCircle: new Konva.Circle({
        listening: false,
        strokeEnabled: false,
        fill: 'white',
        globalCompositeOperation: 'destination-out',
      }),
      innerBorderCircle: new Konva.Circle({
        listening: false,
        stroke: BRUSH_BORDER_INNER_COLOR,
        strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
        strokeEnabled: true,
      }),
      outerBorderCircle: new Konva.Circle({
        listening: false,
        stroke: BRUSH_BORDER_OUTER_COLOR,
        strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
        strokeEnabled: true,
      }),
    };
    this.eraser.group.add(this.eraser.fillCircle);
    this.eraser.group.add(this.eraser.innerBorderCircle);
    this.eraser.group.add(this.eraser.outerBorderCircle);
    this.group.add(this.eraser.group);

    // Create the rect preview - this is a rectangle drawn from the last mouse down position to the current cursor position
    this.rect = {
      group: new Konva.Group(),
      fillRect: new Konva.Rect({
        id: PREVIEW_RECT_ID,
        listening: false,
        strokeEnabled: false,
      }),
    };
    this.rect.group.add(this.rect.fillRect);
    this.group.add(this.rect.group);
  }

  scaleTool = () => {
    const toolState = this.manager.stateApi.getToolState();
    const scale = this.manager.stage.scaleX();

    const brushRadius = toolState.brush.width / 2;
    this.brush.innerBorderCircle.strokeWidth(BRUSH_ERASER_BORDER_WIDTH / scale);
    this.brush.outerBorderCircle.setAttrs({
      strokeWidth: BRUSH_ERASER_BORDER_WIDTH / scale,
      radius: brushRadius + BRUSH_ERASER_BORDER_WIDTH / scale,
    });

    const eraserRadius = toolState.eraser.width / 2;
    this.eraser.innerBorderCircle.strokeWidth(BRUSH_ERASER_BORDER_WIDTH / scale);
    this.eraser.outerBorderCircle.setAttrs({
      strokeWidth: BRUSH_ERASER_BORDER_WIDTH / scale,
      radius: eraserRadius + BRUSH_ERASER_BORDER_WIDTH / scale,
    });
  };

  render() {
    const stage = this.manager.stage;
    const renderedEntityCount: number = 1; // TODO(psyche): this.manager should be renderable entity count
    const toolState = this.manager.stateApi.getToolState();
    const currentFill = this.manager.stateApi.getCurrentFill();
    const selectedEntity = this.manager.stateApi.getSelectedEntity();
    const cursorPos = this.manager.stateApi.getLastCursorPos();
    const lastMouseDownPos = this.manager.stateApi.getLastMouseDownPos();
    const isDrawing = this.manager.stateApi.getIsDrawing();
    const isMouseDown = this.manager.stateApi.getIsMouseDown();

    const tool = toolState.selected;
    const isDrawableEntity =
      selectedEntity?.type === 'regional_guidance' ||
      selectedEntity?.type === 'layer' ||
      selectedEntity?.type === 'inpaint_mask';

    // Update the stage's pointer style
    if (tool === 'view') {
      // View gets a hand
      stage.container().style.cursor = isMouseDown ? 'grabbing' : 'grab';
    } else if (renderedEntityCount === 0) {
      // We have no layers, so we should not render any tool
      stage.container().style.cursor = 'default';
    } else if (!isDrawableEntity) {
      // Non-drawable layers don't have tools
      stage.container().style.cursor = 'not-allowed';
    } else if (tool === 'move') {
      // Move tool gets a pointer
      stage.container().style.cursor = 'default';
    } else if (tool === 'rect') {
      // Rect gets a crosshair
      stage.container().style.cursor = 'crosshair';
    } else if (tool === 'brush' || tool === 'eraser') {
      // Hide the native cursor and use the konva-rendered brush preview
      stage.container().style.cursor = 'none';
    } else if (tool === 'bbox') {
      stage.container().style.cursor = 'default';
    }

    stage.draggable(tool === 'view');

    if (!cursorPos || renderedEntityCount === 0 || !isDrawableEntity) {
      // We can bail early if the mouse isn't over the stage or there are no layers
      this.group.visible(false);
    } else {
      this.group.visible(true);

      // No need to render the brush preview if the cursor position or color is missing
      if (cursorPos && tool === 'brush') {
        const scale = stage.scaleX();
        // Update the fill circle
        const radius = toolState.brush.width / 2;
        this.brush.fillCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius,
          fill: isDrawing ? '' : rgbaColorToString(currentFill),
        });

        // Update the inner border of the brush preview
        this.brush.innerBorderCircle.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius });

        // Update the outer border of the brush preview
        this.brush.outerBorderCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius: radius + BRUSH_ERASER_BORDER_WIDTH / scale,
        });

        this.scaleTool();

        this.brush.group.visible(true);
        this.eraser.group.visible(false);
        this.rect.group.visible(false);
      } else if (cursorPos && tool === 'eraser') {
        const scale = stage.scaleX();
        // Update the fill circle
        const radius = toolState.eraser.width / 2;
        this.eraser.fillCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius,
          fill: 'white',
        });

        // Update the inner border of the eraser preview
        this.eraser.innerBorderCircle.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius });

        // Update the outer border of the eraser preview
        this.eraser.outerBorderCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius: radius + BRUSH_ERASER_BORDER_WIDTH / scale,
        });

        this.scaleTool();

        this.brush.group.visible(false);
        this.eraser.group.visible(true);
        this.rect.group.visible(false);
      } else if (cursorPos && lastMouseDownPos && tool === 'rect') {
        this.rect.fillRect.setAttrs({
          x: Math.min(cursorPos.x, lastMouseDownPos.x),
          y: Math.min(cursorPos.y, lastMouseDownPos.y),
          width: Math.abs(cursorPos.x - lastMouseDownPos.x),
          height: Math.abs(cursorPos.y - lastMouseDownPos.y),
          fill: rgbaColorToString(currentFill),
          visible: true,
        });
        this.brush.group.visible(false);
        this.eraser.group.visible(false);
        this.rect.group.visible(true);
      } else {
        this.brush.group.visible(false);
        this.eraser.group.visible(false);
        this.rect.group.visible(false);
      }
    }
  }
}
