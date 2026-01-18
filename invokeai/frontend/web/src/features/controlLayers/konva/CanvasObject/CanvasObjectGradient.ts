import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasGradientState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectGradient extends CanvasModuleBase {
  readonly type = 'object_gradient';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasGradientState;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  isFirstRender: boolean = false;

  constructor(state: CanvasGradientState, parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer) {
    super();
    this.id = state.id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      rect: new Konva.Rect({ name: `${this.type}:rect`, listening: false, perfectDrawEnabled: false }),
    };
    this.konva.group.add(this.konva.rect);
    this.state = state;
  }

  update(state: CanvasGradientState, force = false): boolean {
    if (force || this.state !== state) {
      this.isFirstRender = false;

      const { rect, fgColor, bgColor } = state;
      const fg = rgbaColorToString(fgColor.a === 0 ? { ...fgColor, r: 0, g: 0, b: 0 } : fgColor);
      const bg = rgbaColorToString(bgColor.a === 0 ? { ...bgColor, r: 0, g: 0, b: 0 } : bgColor);

      this.log.trace({ state }, 'Updating gradient');
      this.konva.rect.setAttrs({
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      });

      this.konva.group.clipFunc((ctx) => {
        ctx.beginPath();
        ctx.rect(state.bboxRect.x, state.bboxRect.y, state.bboxRect.width, state.bboxRect.height);
        ctx.clip();
        if (state.clipEnabled) {
          ctx.beginPath();
          if (state.gradientType === 'linear') {
            const startX = state.rect.x + state.start.x;
            const startY = state.rect.y + state.start.y;
            const endX = state.rect.x + state.end.x;
            const endY = state.rect.y + state.end.y;
            const dirX = Math.cos(state.clipAngle);
            const dirY = Math.sin(state.clipAngle);
            const perpX = -dirY;
            const perpY = dirX;
            const clipExtent = Math.max(Math.hypot(state.bboxRect.width, state.bboxRect.height) * 2, 1);
            const drawHalfPlane = (originX: number, originY: number, dirSign: number) => {
              const dX = dirX * dirSign;
              const dY = dirY * dirSign;
              const p1x = originX + perpX * clipExtent;
              const p1y = originY + perpY * clipExtent;
              const p2x = originX - perpX * clipExtent;
              const p2y = originY - perpY * clipExtent;
              const p3x = p2x + dX * clipExtent;
              const p3y = p2y + dY * clipExtent;
              const p4x = p1x + dX * clipExtent;
              const p4y = p1y + dY * clipExtent;
              ctx.moveTo(p1x, p1y);
              ctx.lineTo(p2x, p2y);
              ctx.lineTo(p3x, p3y);
              ctx.lineTo(p4x, p4y);
              ctx.closePath();
            };
            drawHalfPlane(startX, startY, 1);
            ctx.clip();
            ctx.beginPath();
            drawHalfPlane(endX, endY, -1);
          } else {
            ctx.arc(state.clipCenter.x, state.clipCenter.y, state.clipRadius, 0, Math.PI * 2);
          }
          ctx.clip();
        }
      });

      if (state.gradientType === 'linear') {
        this.konva.rect.setAttrs({
          fillPriority: 'linear-gradient',
          fillLinearGradientStartPoint: { x: state.start.x, y: state.start.y },
          fillLinearGradientEndPoint: { x: state.end.x, y: state.end.y },
          fillLinearGradientColorStops: [0, fg, 1, bg],
        });
      } else {
        this.konva.rect.setAttrs({
          fillPriority: 'radial-gradient',
          fillRadialGradientStartPoint: { x: state.center.x, y: state.center.y },
          fillRadialGradientEndPoint: { x: state.center.x, y: state.center.y },
          fillRadialGradientStartRadius: 0,
          fillRadialGradientEndRadius: state.radius,
          fillRadialGradientColorStops: [0, fg, 1, bg],
        });
      }

      this.state = state;
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting gradient visibility');
    this.konva.group.visible(isVisible);
  }

  destroy = () => {
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      isFirstRender: this.isFirstRender,
      state: deepClone(this.state),
    };
  };
}
