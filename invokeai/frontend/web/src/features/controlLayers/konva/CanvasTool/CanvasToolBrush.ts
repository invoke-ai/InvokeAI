import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { alignCoordForTool, getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

type CanvasToolBrushConfig = {
  /**
   * The inner border color for the brush tool preview.
   */
  BORDER_INNER_COLOR: string;
  /**
   * The outer border color for the brush tool preview.
   */
  BORDER_OUTER_COLOR: string;
};

const DEFAULT_CONFIG: CanvasToolBrushConfig = {
  BORDER_INNER_COLOR: 'rgba(0,0,0,1)',
  BORDER_OUTER_COLOR: 'rgba(255,255,255,0.8)',
};

/**
 * Renders a preview of the brush tool on the canvas.
 */
export class CanvasToolBrush extends CanvasModuleBase {
  readonly type = 'brush_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasToolBrushConfig = DEFAULT_CONFIG;

  /**
   * The Konva objects that make up the brush tool preview:
   * - A group to hold the fill circle and borders
   * - A circle to fill the brush area
   * - An inner border ring
   * - An outer border ring
   */
  konva: {
    group: Konva.Group;
    fillCircle: Konva.Circle;
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
        fill: this.config.BORDER_INNER_COLOR,
        strokeEnabled: false,
      }),
      outerBorder: new Konva.Ring({
        name: `${this.type}:brush_outer_border_ring`,
        listening: false,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.BORDER_OUTER_COLOR,
        strokeEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.fillCircle, this.konva.innerBorder, this.konva.outerBorder);
  }

  render = () => {
    const cursorPos = this.manager.tool.$cursorPos.get();

    // If the cursor position is not available, do not update the brush preview. The tool module will handle visiblity.
    if (!cursorPos) {
      return;
    }

    const settings = this.manager.stateApi.getSettings();
    const brushPreviewFill = this.manager.stateApi.getBrushPreviewColor();
    const alignedCursorPos = alignCoordForTool(cursorPos, settings.brushWidth);
    const radius = settings.brushWidth / 2;

    // The circle is scaled
    this.konva.fillCircle.setAttrs({
      x: alignedCursorPos.x,
      y: alignedCursorPos.y,
      radius,
      fill: rgbaColorToString(brushPreviewFill),
    });

    // But the borders are in screen-pixels
    const onePixel = this.manager.stage.getScaledPixels(1);
    const twoPixels = this.manager.stage.getScaledPixels(2);

    this.konva.innerBorder.setAttrs({
      x: cursorPos.x,
      y: cursorPos.y,
      innerRadius: radius,
      outerRadius: radius + onePixel,
    });
    this.konva.outerBorder.setAttrs({
      x: cursorPos.x,
      y: cursorPos.y,
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
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };
}
