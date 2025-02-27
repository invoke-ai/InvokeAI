import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import { $authToken } from 'app/store/nanostores/authToken';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectDynamicGrid } from 'features/controlLayers/store/canvasSettingsSlice';
import Konva from 'konva';
import type { Logger } from 'roarr';

type CanvasBackgroundModuleConfig = {
  GRID_LINE_COLOR_COARSE: string;
  GRID_LINE_COLOR_FINE: string;
  CHECKERBOARD_PATTERN_DATAURL: string;
};

const DEFAULT_CONFIG: CanvasBackgroundModuleConfig = {
  GRID_LINE_COLOR_COARSE: getArbitraryBaseColor(27),
  GRID_LINE_COLOR_FINE: getArbitraryBaseColor(18),
  CHECKERBOARD_PATTERN_DATAURL: TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL,
};

/**
 * Renders a background for the canvas - either a checkboard pattern or a grid.
 *
 * The grid is dynamic and changes based on the scale of the stage, while the checkboard pattern is static.
 */
export class CanvasBackgroundModule extends CanvasModuleBase {
  readonly type = 'background';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions = new Set<() => void>();
  config: CanvasBackgroundModuleConfig = DEFAULT_CONFIG;

  /**
   * The checkboard pattern image used when the grid is disabled.
   */
  checkboardPattern = new Image();

  /**
   * The Konva objects that make up the background grid:
   * - A layer to hold the grid lines
   * - An array of grid lines
   * - A group to hold the grid lines
   * - A rectangle to hold the checkboard pattern
   */
  konva: {
    layer: Konva.Layer;
    patternRect: Konva.Rect;
    linesGroup: Konva.Group;
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

    this.konva = {
      layer: new Konva.Layer({ name: `${this.type}:layer`, listening: false, imageSmoothingEnabled: false }),
      linesGroup: new Konva.Group({ name: `${this.type}:linesGroup` }),
      lines: [],
      patternRect: new Konva.Rect({ name: `${this.type}:patternRect`, perfectDrawEnabled: false }),
    };

    this.konva.layer.add(this.konva.patternRect);
    this.konva.layer.add(this.konva.linesGroup);

    /**
     * The background should be rendered when the stage attributes change:
     * - scale
     * - position
     * - size
     */
    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.render));

    /**
     * The background should be rendered when the dynamic grid setting changes.
     */
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectDynamicGrid, this.render));
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.checkboardPattern.onload = () => {
      this.konva.patternRect.fillPatternImage(this.checkboardPattern);
      this.render();
    };
    this.checkboardPattern.src = $authToken.get() ? 'use-credentials' : 'anonymous';
    this.checkboardPattern.src = this.config.CHECKERBOARD_PATTERN_DATAURL;
    this.render();
  };

  /**
   * Renders the background.
   */
  render = () => {
    const dynamicGrid = this.manager.stateApi.runSelector(selectDynamicGrid);

    if (!dynamicGrid) {
      this.konva.linesGroup.visible(false);
      const patternScale = this.manager.stage.unscale(1);
      this.konva.patternRect.setAttrs({
        visible: true,
        ...this.manager.stage.getScaledStageRect(),
        fillPatternScaleX: patternScale,
        fillPatternScaleY: patternScale,
      });
      return;
    }

    this.konva.linesGroup.visible(true);
    this.konva.patternRect.visible(false);

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

    this.konva.linesGroup.destroyChildren();
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
        perfectDrawEnabled: false,
      });
      this.konva.lines.push(line);
      this.konva.linesGroup.add(line);
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
        perfectDrawEnabled: false,
      });
      this.konva.lines.push(line);
      this.konva.linesGroup.add(line);
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

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
    };
  };

  destroy = () => {
    this.log.trace('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.layer.destroy();
  };
}
