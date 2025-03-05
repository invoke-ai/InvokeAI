import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { getColorAtCoordinate, getPrefixedId } from 'features/controlLayers/konva/util';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import rafThrottle from 'raf-throttle';
import type { Logger } from 'roarr';

type CanvasColorPickerToolModuleConfig = {
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
  /**
   * The color of the RGBA value text.
   */
  TEXT_COLOR: string;
  /**
   * The padding of the RGBA value text within the background rect.
   */

  TEXT_PADDING: number;
  /**
   * The font size of the RGBA value text.
   */
  TEXT_FONT_SIZE: number;
  /**
   * The color of the RGBA value text background rect.
   */
  TEXT_BG_COLOR: string;
  /**
   * The width of the RGBA value text background rect.
   */
  TEXT_BG_WIDTH: number;
  /**
   * The height of the RGBA value text background rect.
   */
  TEXT_BG_HEIGHT: number;
  /**
   * The corner radius of the RGBA value text background rect.
   */
  TEXT_BG_CORNER_RADIUS: number;
  /**
   * The x offset of the RGBA value text background rect from the color picker ring.
   */
  TEXT_BG_X_OFFSET: number;
};

const DEFAULT_CONFIG: CanvasColorPickerToolModuleConfig = {
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
  TEXT_COLOR: 'rgba(255,255,255,1)',
  TEXT_BG_COLOR: 'rgba(0,0,0,0.8)',
  TEXT_BG_HEIGHT: 62,
  TEXT_BG_WIDTH: 62,
  TEXT_BG_CORNER_RADIUS: 7,
  TEXT_PADDING: 8,
  TEXT_FONT_SIZE: 12,
  TEXT_BG_X_OFFSET: 7,
};

/**
 * Renders a preview of the color picker tool on the canvas.
 */
export class CanvasColorPickerToolModule extends CanvasModuleBase {
  readonly type = 'color_picker_tool';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasToolModule;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasColorPickerToolModuleConfig = DEFAULT_CONFIG;

  /**
   * The color currently under the cursor. Only has a value when the color picker tool is active.
   */
  $colorUnderCursor = atom<RgbaColor>(RGBA_BLACK);

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
    rgbaTextGroup: Konva.Group;
    rgbaText: Konva.Text;
    rgbaTextBackground: Konva.Rect;
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
        listening: false,
        name: `${this.type}:color_picker_candidate_color_ring`,
        innerRadius: 0,
        outerRadius: 0,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringCurrentColor: new Konva.Arc({
        listening: false,
        name: `${this.type}:color_picker_current_color_arc`,
        innerRadius: 0,
        outerRadius: 0,
        angle: 180,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringInnerBorder: new Konva.Ring({
        listening: false,
        name: `${this.type}:color_picker_inner_border_ring`,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.RING_BORDER_INNER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      ringOuterBorder: new Konva.Ring({
        listening: false,
        name: `${this.type}:color_picker_outer_border_ring`,
        innerRadius: 0,
        outerRadius: 0,
        fill: this.config.RING_BORDER_OUTER_COLOR,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      crosshairNorthInner: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_north1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairNorthOuter: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_north2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairEastInner: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_east1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairEastOuter: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_east2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairSouthInner: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_south1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairSouthOuter: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_south2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairWestInner: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_west1_line`,
        stroke: this.config.CROSSHAIR_LINE_COLOR,
        perfectDrawEnabled: false,
      }),
      crosshairWestOuter: new Konva.Line({
        listening: false,
        name: `${this.type}:color_picker_crosshair_west2_line`,
        stroke: this.config.CROSSHAIR_BORDER_COLOR,
        perfectDrawEnabled: false,
      }),
      rgbaTextGroup: new Konva.Group({
        listening: false,
        name: `${this.type}:color_picker_text_group`,
      }),
      rgbaText: new Konva.Text({
        listening: false,
        name: `${this.type}:color_picker_text`,
        fill: this.config.TEXT_COLOR,
        fontFamily: 'monospace',
        align: 'left',
        fontStyle: 'bold',
        verticalAlign: 'middle',
      }),
      rgbaTextBackground: new Konva.Rect({
        listening: false,
        name: `${this.type}:color_picker_text_background`,
        fill: this.config.TEXT_BG_COLOR,
      }),
    };

    this.konva.rgbaTextGroup.add(this.konva.rgbaTextBackground, this.konva.rgbaText);

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
      this.konva.crosshairWestInner,
      this.konva.rgbaTextGroup
    );
  }

  syncCursorStyle = () => {
    this.manager.stage.setCursor('none');
  };

  /**
   * Renders the color picker tool preview on the canvas.
   */
  render = () => {
    if (this.parent.$tool.get() !== 'colorPicker') {
      this.setVisibility(false);
      return;
    }

    const cursorPos = this.parent.$cursorPos.get();

    if (!cursorPos) {
      this.setVisibility(false);
      return;
    }

    this.setVisibility(true);

    const { x, y } = cursorPos.relative;

    const settings = this.manager.stateApi.getSettings();
    const colorUnderCursor = this.$colorUnderCursor.get();
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

    const textBgWidth = this.manager.stage.unscale(this.config.TEXT_BG_WIDTH);
    const textBgHeight = this.manager.stage.unscale(this.config.TEXT_BG_HEIGHT);

    this.konva.rgbaTextBackground.setAttrs({
      width: textBgWidth,
      height: textBgHeight,
      cornerRadius: this.manager.stage.unscale(this.config.TEXT_BG_CORNER_RADIUS),
    });
    this.konva.rgbaText.setAttrs({
      padding: this.manager.stage.unscale(this.config.TEXT_PADDING),
      fontSize: this.manager.stage.unscale(this.config.TEXT_FONT_SIZE),
      text: `R: ${colorUnderCursor.r}\nG: ${colorUnderCursor.g}\nB: ${colorUnderCursor.b}\nA: ${colorUnderCursor.a}`,
    });
    this.konva.rgbaTextGroup.setAttrs({
      x: x + this.manager.stage.unscale(this.config.RING_OUTER_RADIUS + this.config.TEXT_BG_X_OFFSET),
      y: y - textBgHeight / 2,
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

  onStagePointerUp = (_e: KonvaEventObject<PointerEvent>) => {
    const color = this.$colorUnderCursor.get();
    const settings = this.manager.stateApi.getSettings();
    this.manager.stateApi.setColor({ ...settings.color, ...color });
  };

  onStagePointerMove = (_e: KonvaEventObject<PointerEvent>) => {
    this.syncColorUnderCursor();
  };

  syncColorUnderCursor = rafThrottle(() => {
    const cursorPos = this.parent.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    // Hide the background layer so we can get the color under the cursor without the grid interfering
    this.manager.background.konva.layer.visible(false);
    const color = getColorAtCoordinate(this.manager.stage.konva.stage, cursorPos.absolute);
    this.manager.background.konva.layer.visible(true);

    if (color) {
      this.$colorUnderCursor.set(color);
    }
  });

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
      $colorUnderCursor: this.$colorUnderCursor.get(),
    };
  };

  destroy = () => {
    this.log.debug('Destroying color picker tool preview module');
    this.konva.group.destroy();
  };
}
