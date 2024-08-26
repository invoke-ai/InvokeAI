import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasBackgroundModule extends CanvasModuleBase {
  readonly type = 'background';

  static GRID_LINE_COLOR_COARSE = getArbitraryBaseColor(27);
  static GRID_LINE_COLOR_FINE = getArbitraryBaseColor(18);

  id: string;
  path: string[];
  manager: CanvasManager;
  subscriptions = new Set<() => void>();
  log: Logger;

  konva: {
    layer: Konva.Layer;
  };

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug('Creating background module');

    this.konva = { layer: new Konva.Layer({ name: `${this.type}:layer`, listening: false }) };

    this.subscriptions.add(
      this.manager.stateApi.$stageAttrs.listen(() => {
        this.render();
      })
    );
  }

  render() {
    const settings = this.manager.stateApi.getSettings();

    if (!settings.dynamicGrid) {
      this.konva.layer.visible(false);
      return;
    }

    this.konva.layer.visible(true);

    this.konva.layer.zIndex(0);
    const scale = this.manager.stage.getScale();
    const { x, y } = this.manager.stage.getPosition();
    const { width, height } = this.manager.stage.getSize();
    const gridSpacing = this.getGridSpacing(scale);
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

    for (let i = 0; i < xSteps; i++) {
      _x = gridFullRect.x1 + i * gridSpacing;
      this.konva.layer.add(
        new Konva.Line({
          x: _x,
          y: gridFullRect.y1,
          points: [0, 0, 0, ySize],
          stroke: _x % 64 ? CanvasBackgroundModule.GRID_LINE_COLOR_FINE : CanvasBackgroundModule.GRID_LINE_COLOR_COARSE,
          strokeWidth,
          listening: false,
        })
      );
    }
    for (let i = 0; i < ySteps; i++) {
      _y = gridFullRect.y1 + i * gridSpacing;
      this.konva.layer.add(
        new Konva.Line({
          x: gridFullRect.x1,
          y: _y,
          points: [0, 0, xSize, 0],
          stroke: _y % 64 ? CanvasBackgroundModule.GRID_LINE_COLOR_FINE : CanvasBackgroundModule.GRID_LINE_COLOR_COARSE,
          strokeWidth,
          listening: false,
        })
      );
    }
  }

  destroy = () => {
    this.log.trace('Destroying background module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.layer.destroy();
  };

  /**
   * Gets the grid spacing. The value depends on the stage scale - at higher scales, the grid spacing is smaller.
   * @param scale The stage scale
   * @returns The grid spacing based on the stage scale
   */
  getGridSpacing = (scale: number): number => {
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
      path: this.path,
      type: this.type,
    };
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
