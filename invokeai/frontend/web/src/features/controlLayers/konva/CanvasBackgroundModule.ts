import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

type CanvasBackgroundModuleConfig = {
  GRID_LINE_COLOR_COARSE: string;
  GRID_LINE_COLOR_FINE: string;
};

const DEFAULT_CONFIG: CanvasBackgroundModuleConfig = {
  GRID_LINE_COLOR_COARSE: getArbitraryBaseColor(27),
  GRID_LINE_COLOR_FINE: getArbitraryBaseColor(18),
};

/**
 * Renders a background grid on the canvas, where the grid spacing changes based on the stage scale.
 *
 * The grid is only visible when the dynamic grid setting is enabled.
 */
export class CanvasBackgroundModule extends CanvasModuleBase {
  readonly type = 'background';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;

  subscriptions = new Set<() => void>();
  config: CanvasBackgroundModuleConfig = DEFAULT_CONFIG;

  /**
   * The Konva objects that make up the background grid:
   * - A layer to hold the grid lines
   * - An array of grid lines
   */
  konva: {
    layer: Konva.Layer;
    lines: Konva.Line[];
  };

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.manager = manager;
    this.parent = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = { layer: new Konva.Layer({ name: `${this.type}:layer`, listening: false }), lines: [] };

    /**
     * The background grid should be rendered when the stage attributes change:
     * - scale
     * - position
     * - size
     */
    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.render));
  }

  /**
   * Renders the background grid.
   */
  render = () => {
    const settings = this.manager.stateApi.getSettings();

    if (!settings.dynamicGrid) {
      this.konva.layer.visible(false);
      return;
    }

    this.konva.layer.visible(true);

    const scale = this.manager.stage.getScale();
    const { x, y } = this.manager.stage.getPosition();
    const { width, height } = this.manager.stage.getSize();
    const gridSpacing = CanvasBackgroundModule.getGridSpacing(scale);
    const stageRect = {
      x1: 0,
      y1: 0,
      x2: width,
      y2: height,
    };

    const gridOffset = {
      x: Math.ceil(x / scale / gridSpacing) * gridSpacing,
      y: Math.ceil(y / scale / gridSpacing) * gridSpacing,
    };

    const gridRect = {
      x1: -gridOffset.x,
      y1: -gridOffset.y,
      x2: width / scale - gridOffset.x + gridSpacing,
      y2: height / scale - gridOffset.y + gridSpacing,
    };

    const gridFullRect = {
      x1: Math.min(stageRect.x1, gridRect.x1),
      y1: Math.min(stageRect.y1, gridRect.y1),
      x2: Math.max(stageRect.x2, gridRect.x2),
      y2: Math.max(stageRect.y2, gridRect.y2),
    };

    // find the x & y size of the grid
    const xSize = gridFullRect.x2 - gridFullRect.x1;
    const ySize = gridFullRect.y2 - gridFullRect.y1;
    // compute the number of steps required on each axis.
    const xSteps = Math.round(xSize / gridSpacing) + 1;
    const ySteps = Math.round(ySize / gridSpacing) + 1;

    const strokeWidth = 1 / scale;
    let _x = 0;
    let _y = 0;

    this.konva.layer.destroyChildren();
    this.konva.lines = [];

    for (let i = 0; i < xSteps; i++) {
      _x = gridFullRect.x1 + i * gridSpacing;
      const line = new Konva.Line({
        x: _x,
        y: gridFullRect.y1,
        points: [0, 0, 0, ySize],
        stroke: _x % 64 ? this.config.GRID_LINE_COLOR_FINE : this.config.GRID_LINE_COLOR_COARSE,
        strokeWidth,
        listening: false,
      });
      this.konva.lines.push(line);
      this.konva.layer.add(line);
    }
    for (let i = 0; i < ySteps; i++) {
      _y = gridFullRect.y1 + i * gridSpacing;
      const line = new Konva.Line({
        x: gridFullRect.x1,
        y: _y,
        points: [0, 0, xSize, 0],
        stroke: _y % 64 ? this.config.GRID_LINE_COLOR_FINE : this.config.GRID_LINE_COLOR_COARSE,
        strokeWidth,
        listening: false,
      });
      this.konva.lines.push(line);
      this.konva.layer.add(line);
    }
  };

  /**
   * Gets the grid line spacing for the dynamic grid.
   *
   * The value depends on the stage scale - at higher scales, the grid spacing is smaller.
   *
   * @param scale The stage scale
   */
  static getGridSpacing = (scale: number): number => {
    if (scale >= 2) {
      return 8;
    }
    if (scale >= 1 && scale < 2) {
      return 16;
    }
    if (scale >= 0.5 && scale < 1) {
      return 32;
    }
    if (scale >= 0.25 && scale < 0.5) {
      return 64;
    }
    if (scale >= 0.125 && scale < 0.25) {
      return 128;
    }
    return 256;
  };

  destroy = () => {
    this.log.trace('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.layer.destroy();
  };
}
