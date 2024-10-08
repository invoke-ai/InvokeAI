import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { alignCoordForTool, getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

type CanvasToolEraserConfig = {
  /**
   * The inner border color for the eraser tool preview.
   */
  BORDER_INNER_COLOR: string;
  /**
   * The outer border color for the eraser tool preview.
   */
  BORDER_OUTER_COLOR: string;
};

const DEFAULT_CONFIG: CanvasToolEraserConfig = {
  BORDER_INNER_COLOR: 'rgba(0,0,0,1)',
  BORDER_OUTER_COLOR: 'rgba(255,255,255,0.8)',
};

export class CanvasToolEraser extends CanvasModuleBase {
  readonly type = 'eraser_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasToolEraserConfig = DEFAULT_CONFIG;

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

  render = () => {
    const tool = this.parent.$tool.get();

    if (tool !== 'eraser') {
      this.setVisibility(false);
      return;
    }

    const cursorPos = this.parent.$cursorPos.get();
    const canDraw = this.parent.getCanDraw();

    if (!cursorPos || !canDraw) {
      this.setVisibility(false);
      return;
    }

    const isMouseDown = this.parent.$isMouseDown.get();
    const lastPointerType = this.parent.$lastPointerType.get();

    if (lastPointerType !== 'mouse' && isMouseDown) {
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
