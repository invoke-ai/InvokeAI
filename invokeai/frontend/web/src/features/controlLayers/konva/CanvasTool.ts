import type { JSONObject } from 'common/types';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasPreview } from 'features/controlLayers/konva/CanvasPreview';
import {
  BRUSH_BORDER_INNER_COLOR,
  BRUSH_BORDER_OUTER_COLOR,
  BRUSH_ERASER_BORDER_WIDTH,
} from 'features/controlLayers/konva/constants';
import { alignCoordForTool, getPrefixedId } from 'features/controlLayers/konva/util';
import type { Tool } from 'features/controlLayers/store/types';
import { isDrawableEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasTool {
  readonly type = 'tool_preview';

  id: string;
  path: string[];
  parent: CanvasPreview;
  manager: CanvasManager;
  log: Logger;

  konva: {
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
    eyeDropper: {
      group: Konva.Group;
      fillCircle: Konva.Circle;
      transparentCenterCircle: Konva.Circle;
    };
  };

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  constructor(parent: CanvasPreview) {
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
        innerBorderCircle: new Konva.Circle({
          name: `${this.type}:brush_inner_border_circle`,
          listening: false,
          stroke: BRUSH_BORDER_INNER_COLOR,
          strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
          strokeEnabled: true,
        }),
        outerBorderCircle: new Konva.Circle({
          name: `${this.type}:brush_outer_border_circle`,
          listening: false,
          stroke: BRUSH_BORDER_OUTER_COLOR,
          strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
          strokeEnabled: true,
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
        innerBorderCircle: new Konva.Circle({
          name: `${this.type}:eraser_inner_border_circle`,
          listening: false,
          stroke: BRUSH_BORDER_INNER_COLOR,
          strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
          strokeEnabled: true,
        }),
        outerBorderCircle: new Konva.Circle({
          name: `${this.type}:eraser_outer_border_circle`,
          listening: false,
          stroke: BRUSH_BORDER_OUTER_COLOR,
          strokeWidth: BRUSH_ERASER_BORDER_WIDTH,
          strokeEnabled: true,
        }),
      },
      eyeDropper: {
        group: new Konva.Group({ name: `${this.type}:eyeDropper_group`, listening: false }),
        fillCircle: new Konva.Circle({
          name: `${this.type}:eyeDropper_fill_circle`,
          listening: false,
          fill: '',
          radius: 20,
          strokeWidth: 1,
          stroke: 'black',
          strokeScaleEnabled: false,
        }),
        transparentCenterCircle: new Konva.Circle({
          name: `${this.type}:eyeDropper_fill_circle`,
          listening: false,
          strokeEnabled: false,
          fill: 'white',
          radius: 5,
          globalCompositeOperation: 'destination-out',
        }),
      },
    };
    this.konva.brush.group.add(this.konva.brush.fillCircle);
    this.konva.brush.group.add(this.konva.brush.innerBorderCircle);
    this.konva.brush.group.add(this.konva.brush.outerBorderCircle);
    this.konva.group.add(this.konva.brush.group);

    this.konva.eraser.group.add(this.konva.eraser.fillCircle);
    this.konva.eraser.group.add(this.konva.eraser.innerBorderCircle);
    this.konva.eraser.group.add(this.konva.eraser.outerBorderCircle);
    this.konva.group.add(this.konva.eraser.group);

    this.konva.eyeDropper.group.add(this.konva.eyeDropper.fillCircle);
    this.konva.eyeDropper.group.add(this.konva.eyeDropper.transparentCenterCircle);
    this.konva.group.add(this.konva.eyeDropper.group);

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
    const scale = this.manager.stage.getScale();

    const brushRadius = toolState.brush.width / 2;
    this.konva.brush.innerBorderCircle.strokeWidth(BRUSH_ERASER_BORDER_WIDTH / scale);
    this.konva.brush.outerBorderCircle.setAttrs({
      strokeWidth: BRUSH_ERASER_BORDER_WIDTH / scale,
      radius: brushRadius + BRUSH_ERASER_BORDER_WIDTH / scale,
    });

    const eraserRadius = toolState.eraser.width / 2;
    this.konva.eraser.innerBorderCircle.strokeWidth(BRUSH_ERASER_BORDER_WIDTH / scale);
    this.konva.eraser.outerBorderCircle.setAttrs({
      strokeWidth: BRUSH_ERASER_BORDER_WIDTH / scale,
      radius: eraserRadius + BRUSH_ERASER_BORDER_WIDTH / scale,
    });
  };

  setToolVisibility = (tool: Tool) => {
    this.konva.brush.group.visible(tool === 'brush');
    this.konva.eraser.group.visible(tool === 'eraser');
    this.konva.eyeDropper.group.visible(tool === 'eyeDropper');
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
    } else if (tool === 'eyeDropper') {
      // Eyedropper gets none
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
        const scale = stage.getScale();
        // Update the fill circle
        const radius = toolState.brush.width / 2;

        this.konva.brush.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: isDrawing ? '' : rgbaColorToString(brushPreviewFill),
        });

        // Update the inner border of the brush preview
        this.konva.brush.innerBorderCircle.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius });

        // Update the outer border of the brush preview
        this.konva.brush.outerBorderCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius: radius + BRUSH_ERASER_BORDER_WIDTH / scale,
        });
      } else if (cursorPos && tool === 'eraser') {
        const alignedCursorPos = alignCoordForTool(cursorPos, toolState.eraser.width);

        const scale = stage.getScale();
        // Update the fill circle
        const radius = toolState.eraser.width / 2;
        this.konva.eraser.fillCircle.setAttrs({
          x: alignedCursorPos.x,
          y: alignedCursorPos.y,
          radius,
          fill: 'white',
        });

        // Update the inner border of the eraser preview
        this.konva.eraser.innerBorderCircle.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius });

        // Update the outer border of the eraser preview
        this.konva.eraser.outerBorderCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          radius: radius + BRUSH_ERASER_BORDER_WIDTH / scale,
        });
      } else if (cursorPos && colorUnderCursor) {
        this.konva.eyeDropper.fillCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
          fill: rgbaColorToString(colorUnderCursor),
        });
        this.konva.eyeDropper.transparentCenterCircle.setAttrs({
          x: cursorPos.x,
          y: cursorPos.y,
        });
      }

      this.scaleTool();
      this.setToolVisibility(tool);
    }
  }

  getLoggingContext = (): JSONObject => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
