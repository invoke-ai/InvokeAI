import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId, getRectUnion } from 'features/controlLayers/konva/util';
import type {
  CanvasEntityIdentifier,
  Coordinate,
  Dimensions,
  Rect,
  StageAttrs,
} from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { clamp } from 'lodash-es';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasStageModuleConfig = {
  /**
   * The minimum (furthest-zoomed-in) scale of the canvas
   */
  MIN_SCALE: number;
  /**
   * The maximum (furthest-zoomed-out) scale of the canvas
   */
  MAX_SCALE: number;
  /**
   * The factor by which the canvas should be scaled when zooming in/out
   */
  SCALE_FACTOR: number;
};

const DEFAULT_CONFIG: CanvasStageModuleConfig = {
  MIN_SCALE: 0.1,
  MAX_SCALE: 20,
  SCALE_FACTOR: 0.999,
};

export class CanvasStageModule extends CanvasModuleBase {
  readonly type = 'stage';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;

  container: HTMLDivElement;
  konva: { stage: Konva.Stage };

  config: CanvasStageModuleConfig = DEFAULT_CONFIG;

  $stageAttrs = atom<StageAttrs>({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    scale: 0,
  });

  subscriptions = new Set<() => void>();

  constructor(stage: Konva.Stage, container: HTMLDivElement, manager: CanvasManager) {
    super();
    this.id = getPrefixedId('stage');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.container = container;
    this.konva = { stage };
  }

  setEventListeners = () => {
    this.konva.stage.on('wheel', this.onStageMouseWheel);
    this.konva.stage.on('dragmove', this.onStageDragMove);
    this.konva.stage.on('dragend', this.onStageDragEnd);

    return () => {
      this.konva.stage.off('wheel', this.onStageMouseWheel);
      this.konva.stage.off('dragmove', this.onStageDragMove);
      this.konva.stage.off('dragend', this.onStageDragEnd);
    };
  };

  initialize = () => {
    this.log.debug('Initializing stage');
    this.konva.stage.container(this.container);
    const resizeObserver = new ResizeObserver(this.fitStageToContainer);
    resizeObserver.observe(this.container);
    this.fitStageToContainer();
    this.fitLayersToStage();
    const cleanupListeners = this.setEventListeners();

    this.subscriptions.add(cleanupListeners);
    this.subscriptions.add(() => {
      resizeObserver.disconnect();
    });
  };

  fitStageToContainer = () => {
    this.log.trace('Fitting stage to container');
    this.konva.stage.width(this.konva.stage.container().offsetWidth);
    this.konva.stage.height(this.konva.stage.container().offsetHeight);
    this.$stageAttrs.set({
      x: this.konva.stage.x(),
      y: this.konva.stage.y(),
      width: this.konva.stage.width(),
      height: this.konva.stage.height(),
      scale: this.konva.stage.scaleX(),
    });
  };

  getVisibleRect = (type?: Exclude<CanvasEntityIdentifier['type'], 'ip_adapter'>): Rect => {
    const rects = [];

    for (const adapter of this.manager.adapters.getAll()) {
      if (!adapter.state.isEnabled) {
        continue;
      }
      if (type && adapter.state.type !== type) {
        continue;
      }
      rects.push(adapter.transformer.getRelativeRect());
    }

    return getRectUnion(...rects);
  };

  fitBboxToStage = () => {
    const { rect } = this.manager.stateApi.getBbox();
    this.log.trace({ rect }, 'Fitting bbox to stage');
    this.fitRect(rect);
  };

  fitLayersToStage() {
    const rect = this.getVisibleRect();
    if (rect.width === 0 || rect.height === 0) {
      this.fitBboxToStage();
    } else {
      this.log.trace({ rect }, 'Fitting layers to stage');
      this.fitRect(rect);
    }
  }

  fitRect = (rect: Rect) => {
    const { width, height } = this.getSize();

    const padding = 20; // Padding in absolute pixels

    const availableWidth = width - padding * 2;
    const availableHeight = height - padding * 2;

    const scale = Math.min(Math.min(availableWidth / rect.width, availableHeight / rect.height), 1);
    const x = -rect.x * scale + padding + (availableWidth - rect.width * scale) / 2;
    const y = -rect.y * scale + padding + (availableHeight - rect.height * scale) / 2;

    this.konva.stage.setAttrs({
      x,
      y,
      scaleX: scale,
      scaleY: scale,
    });

    this.$stageAttrs.set({
      ...this.$stageAttrs.get(),
      x,
      y,
      scale,
    });
  };

  /**
   * Gets the center of the stage in either absolute or relative coordinates
   * @param absolute Whether to return the center in absolute coordinates
   */
  getCenter = (absolute = false): Coordinate => {
    const scale = this.getScale();
    const { x, y } = this.getPosition();
    const { width, height } = this.getSize();

    const center = {
      x: (width / 2 - x) / scale,
      y: (height / 2 - y) / scale,
    };

    if (!absolute) {
      return center;
    }

    return this.konva.stage.getAbsoluteTransform().point(center);
  };

  /**
   * Sets the scale of the stage. If center is provided, the stage will zoom in/out on that point.
   * @param scale The new scale to set
   * @param center The center of the stage to zoom in/out on
   */
  setScale = (scale: number, center: Coordinate = this.getCenter(true)) => {
    this.log.trace('Setting scale');
    const newScale = clamp(Math.round(scale * 100) / 100, this.config.MIN_SCALE, this.config.MAX_SCALE);

    const { x, y } = this.getPosition();
    const oldScale = this.getScale();

    const deltaX = (center.x - x) / oldScale;
    const deltaY = (center.y - y) / oldScale;

    const newX = center.x - deltaX * newScale;
    const newY = center.y - deltaY * newScale;

    this.konva.stage.setAttrs({
      x: newX,
      y: newY,
      scaleX: newScale,
      scaleY: newScale,
    });

    this.$stageAttrs.set({
      x: Math.floor(this.konva.stage.x()),
      y: Math.floor(this.konva.stage.y()),
      width: this.konva.stage.width(),
      height: this.konva.stage.height(),
      scale: this.konva.stage.scaleX(),
    });
  };

  onStageMouseWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();

    if (e.evt.ctrlKey || e.evt.metaKey) {
      return;
    }

    // We need the absolute cursor position - not the scaled position
    const cursorPos = this.konva.stage.getPointerPosition();

    if (cursorPos) {
      // When wheeling on trackpad, e.evt.ctrlKey is true - in that case, let's reverse the direction
      const delta = e.evt.ctrlKey ? -e.evt.deltaY : e.evt.deltaY;
      const scale = this.manager.stage.getScale() * this.config.SCALE_FACTOR ** delta;
      this.manager.stage.setScale(scale, cursorPos);
    }
  };

  onStageDragMove = (e: KonvaEventObject<MouseEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    this.$stageAttrs.set({
      // Stage position should always be an integer, else we get fractional pixels which are blurry
      x: Math.floor(this.konva.stage.x()),
      y: Math.floor(this.konva.stage.y()),
      width: this.konva.stage.width(),
      height: this.konva.stage.height(),
      scale: this.konva.stage.scaleX(),
    });
  };

  onStageDragEnd = (e: KonvaEventObject<DragEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    this.$stageAttrs.set({
      // Stage position should always be an integer, else we get fractional pixels which are blurry
      x: Math.floor(this.konva.stage.x()),
      y: Math.floor(this.konva.stage.y()),
      width: this.konva.stage.width(),
      height: this.konva.stage.height(),
      scale: this.konva.stage.scaleX(),
    });
  };

  /**
   * Gets the scale of the stage. The stage is always scaled uniformly in x and y.
   */
  getScale = (): number => {
    // The stage is never scaled differently in x and y
    return this.konva.stage.scaleX();
  };

  /**
   * Gets the position of the stage.
   */
  getPosition = (): Coordinate => {
    return this.konva.stage.position();
  };

  /**
   * Gets the size of the stage.
   */
  getSize(): Dimensions {
    return this.konva.stage.size();
  }

  /**
   * Scales a number of pixels by the current stage scale. For example, if the stage is scaled by 5, then 10 pixels
   * would be scaled to 10px / 5 = 2 pixels.
   * @param pixels The number of pixels to scale
   * @returns The number of pixels scaled by the current stage scale
   */
  getScaledPixels = (pixels: number): number => {
    return pixels / this.getScale();
  };

  setIsDraggable = (isDraggable: boolean) => {
    this.konva.stage.draggable(isDraggable);
  };

  addLayer = (layer: Konva.Layer) => {
    this.konva.stage.add(layer);
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.stage.destroy();
  };
}
