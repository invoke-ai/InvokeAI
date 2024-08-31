import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasToolModule';
import { alignCoordForTool, getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

type BrushToolPreviewConfig = {
  /**
   * The inner border color for the brush tool preview.
   */
  BORDER_INNER_COLOR: string;
  /**
   * The outer border color for the brush tool preview.
   */
  BORDER_OUTER_COLOR: string;
};

const DEFAULT_CONFIG: BrushToolPreviewConfig = {
  BORDER_INNER_COLOR: 'rgba(0,0,0,1)',
  BORDER_OUTER_COLOR: 'rgba(255,255,255,0.8)',
};

export class CanvasBrushToolPreview extends CanvasModuleABC {
  readonly type = 'brush_tool_preview';
  id: string;
  path: string[];
  parent: CanvasToolModule;
  manager: CanvasManager;
  log: Logger;
  subscriptions: Set<() => void> = new Set();

  config: BrushToolPreviewConfig;

  konva: {
    group: Konva.Group;
    fillCircle: Konva.Circle;
    innerBorder: Konva.Ring;
    outerBorder: Konva.Ring;
  };

  constructor(parent: CanvasToolModule, config?: Partial<BrushToolPreviewConfig>) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.log.debug('Creating brush tool preview module');

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
    const cursorPos = this.manager.stateApi.$lastCursorPos.get();

    if (!cursorPos) {
      return;
    }

    const toolState = this.manager.stateApi.getToolState();
    const brushPreviewFill = this.manager.stateApi.getBrushPreviewFill();
    const alignedCursorPos = alignCoordForTool(cursorPos, toolState.brush.width);
    const radius = toolState.brush.width / 2;

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
    this.log.debug('Destroying brush tool preview module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };

  getLoggingContext = () => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
