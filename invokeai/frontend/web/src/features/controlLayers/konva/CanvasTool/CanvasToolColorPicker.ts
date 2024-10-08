import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

type CanvasToolColorPickerConfig = {
  /**
   * The inner radius of the ring.
   */
  RING_INNER_RADIUS: number;
  /**
   * The outer radius of the ring.
   */
  RING_OUTER_RADIUS: number;
  /**
   * The inner border color of the outside edge of ring.
   */
  RING_BORDER_INNER_COLOR: string;
  /**
   * The outer border color of the outside edge of ring.
   */
  RING_BORDER_OUTER_COLOR: string;
  /**
   * The radius of the space between the center of the ring and start of the crosshair lines.
   */
  CROSSHAIR_INNER_RADIUS: number;
  /**
   * The length of the crosshair lines.
   */
  CROSSHAIR_LINE_LENGTH: number;
  /**
   * The thickness of the crosshair lines.
   */
  CROSSHAIR_LINE_THICKNESS: number;
  /**
   * The color of the crosshair lines.
   */
  CROSSHAIR_LINE_COLOR: string;
  /**
   * The thickness of the crosshair lines borders
   */
  CROSSHAIR_LINE_BORDER_THICKNESS: number;
  /**
   * The color of the crosshair line borders.
   */
  CROSSHAIR_BORDER_COLOR: string;
};

const DEFAULT_CONFIG: CanvasToolColorPickerConfig = {
  RING_INNER_RADIUS: 25,
  RING_OUTER_RADIUS: 35,
  RING_BORDER_INNER_COLOR: 'rgba(0,0,0,1)',
  RING_BORDER_OUTER_COLOR: 'rgba(255,255,255,0.8)',
  CROSSHAIR_INNER_RADIUS: 5,
  CROSSHAIR_LINE_THICKNESS: 1.5,
  CROSSHAIR_LINE_BORDER_THICKNESS: 0.75,
  CROSSHAIR_LINE_LENGTH: 10,
  CROSSHAIR_LINE_COLOR: 'rgba(0,0,0,1)',
  CROSSHAIR_BORDER_COLOR: 'rgba(255,255,255,0.8)',
};

/**
 * Renders a preview of the color picker tool on the canvas.
 */
export class CanvasToolColorPicker extends CanvasModuleBase {
  readonly type = 'color_picker_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasToolColorPickerConfig = DEFAULT_CONFIG;

  /**
   * The Konva objects that make up the color picker tool preview:
   * - A group to hold all the objects
   * - A ring that shows the candidate and current color
   * - A crosshair to help with color selection
   */
  konva: {
    group: Konva.Group;
    ringCandidateColor: Konva.Ring;
    ringCurrentColor: Konva.Arc;
    ringInnerBorder: Konva.Ring;
    ringOuterBorder: Konva.Ring;
    crosshairNorthInner: Konva.Line;
    crosshairNorthOuter: Konva.Line;
    crosshairEastInner: Konva.Line;
    crosshairEastOuter: Konva.Line;
    crosshairSouthInner: Konva.Line;
    crosshairSouthOuter: Konva.Line;
    crosshairWestInner: Konva.Line;
    crosshairWestOuter: Konva.Line;
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
      group: new Konva.Group({ name: `${this.type}:color_picker_group`, listening: false }),
      ringCandidateColor: new Konva.Ring({
        name: `${this.type}:color_picker_candidate_color_ring`,
        innerRadius: 0,
        outerRadius: 0,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringCurrentColor: new Konva.Arc({
        name: `${this.type}:color_picker_current_color_arc`,
        innerRadius: 0,
        outerRadius: 0,
        angle: 180,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringInnerBorder: new Konva.Ring({
        name: `${this.type}:color_picker_inner_border_ring`,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.RING_BORDER_INNER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringOuterBorder: new Konva.Ring({
        name: `${this.type}:color_picker_outer_border_ring`,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.RING_BORDER_OUTER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      crosshairNorthInner: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_north1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairNorthOuter: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_north2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairEastInner: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_east1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairEastOuter: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_east2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairSouthInner: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_south1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairSouthOuter: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_south2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairWestInner: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_west1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairWestOuter: new Konva.Line({
        name: `${this.type}:color_picker_crosshair_west2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
    };

    this.konva.group.add(
      this.konva.ringCandidateColor,
      this.konva.ringCurrentColor,
      this.konva.ringInnerBorder,
      this.konva.ringOuterBorder,
      this.konva.crosshairNorthOuter,
      this.konva.crosshairNorthInner,
      this.konva.crosshairEastOuter,
      this.konva.crosshairEastInner,
      this.konva.crosshairSouthOuter,
      this.konva.crosshairSouthInner,
      this.konva.crosshairWestOuter,
      this.konva.crosshairWestInner
    );
  }

  /**
   * Renders the color picker tool preview on the canvas.
   */
  render = () => {
    const tool = this.parent.$tool.get();

    if (tool !== 'colorPicker') {
      this.setVisibility(false);
      return;
    }

    const cursorPos = this.parent.$cursorPos.get();
    const canDraw = this.parent.getCanDraw();

    if (!cursorPos || tool !== 'colorPicker' || !canDraw) {
      this.setVisibility(false);
      return;
    }

    this.setVisibility(true);

    const { x, y } = cursorPos.relative;

    const settings = this.manager.stateApi.getSettings();
    const colorUnderCursor = this.parent.$colorUnderCursor.get();
    const colorPickerInnerRadius = this.manager.stage.unscale(this.config.RING_INNER_RADIUS);
    const colorPickerOuterRadius = this.manager.stage.unscale(this.config.RING_OUTER_RADIUS);
    const onePixel = this.manager.stage.unscale(1);
    const twoPixels = this.manager.stage.unscale(2);

    this.konva.ringCandidateColor.setAttrs({
      x,
      y,
      fill: rgbColorToString(colorUnderCursor),
      innerRadius: colorPickerInnerRadius,
      outerRadius: colorPickerOuterRadius,
    });
    this.konva.ringCurrentColor.setAttrs({
      x,
      y,
      fill: rgbColorToString(settings.color),
      innerRadius: colorPickerInnerRadius,
      outerRadius: colorPickerOuterRadius,
    });
    this.konva.ringInnerBorder.setAttrs({
      x,
      y,
      innerRadius: colorPickerOuterRadius,
      outerRadius: colorPickerOuterRadius + onePixel,
    });
    this.konva.ringOuterBorder.setAttrs({
      x,
      y,
      innerRadius: colorPickerOuterRadius + onePixel,
      outerRadius: colorPickerOuterRadius + twoPixels,
    });

    const size = this.manager.stage.unscale(this.config.CROSSHAIR_LINE_LENGTH);
    const space = this.manager.stage.unscale(this.config.CROSSHAIR_INNER_RADIUS);
    const innerThickness = this.manager.stage.unscale(this.config.CROSSHAIR_LINE_THICKNESS);
    const outerThickness = this.manager.stage.unscale(
      this.config.CROSSHAIR_LINE_THICKNESS + this.config.CROSSHAIR_LINE_BORDER_THICKNESS * 2
    );
    this.konva.crosshairNorthOuter.setAttrs({
      strokeWidth: outerThickness,
      points: [x, y - size, x, y - space],
    });
    this.konva.crosshairNorthInner.setAttrs({
      strokeWidth: innerThickness,
      points: [x, y - size, x, y - space],
    });
    this.konva.crosshairEastOuter.setAttrs({
      strokeWidth: outerThickness,
      points: [x + space, y, x + size, y],
    });
    this.konva.crosshairEastInner.setAttrs({
      strokeWidth: innerThickness,
      points: [x + space, y, x + size, y],
    });
    this.konva.crosshairSouthOuter.setAttrs({
      strokeWidth: outerThickness,
      points: [x, y + space, x, y + size],
    });
    this.konva.crosshairSouthInner.setAttrs({
      strokeWidth: innerThickness,
      points: [x, y + space, x, y + size],
    });
    this.konva.crosshairWestOuter.setAttrs({
      strokeWidth: outerThickness,
      points: [x - space, y, x - size, y],
    });
    this.konva.crosshairWestInner.setAttrs({
      strokeWidth: innerThickness,
      points: [x - space, y, x - size, y],
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
    this.log.debug('Destroying color picker tool preview module');
    this.konva.group.destroy();
  };
}
