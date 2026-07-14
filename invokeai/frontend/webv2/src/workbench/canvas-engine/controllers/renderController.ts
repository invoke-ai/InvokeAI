import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { RenderScheduler, RenderSchedulerDeps } from '@workbench/canvas-engine/render/scheduler';

import { createRenderScheduler } from '@workbench/canvas-engine/render/scheduler';

import { PreviewStateController } from './previewStateController';

const wrapCanvasSurface = (canvas: HTMLCanvasElement): RasterSurface => {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to acquire a 2D context from the canvas element');
  }
  return {
    canvas,
    ctx,
    get height() {
      return canvas.height;
    },
    resize(width: number, height: number) {
      canvas.width = width;
      canvas.height = height;
    },
    get width() {
      return canvas.width;
    },
  };
};

export interface RenderControllerOptions extends RenderSchedulerDeps {
  readonly isEngineDisposed: () => boolean;
  readonly getInputHandlers: () => {
    onPointerDown: (event: PointerEvent) => void;
    onPointerMove: (event: PointerEvent) => void;
    onPointerUp: (event: PointerEvent) => void;
    onPointerCancel: (event: PointerEvent) => void;
    onPointerEnter: (event: PointerEvent) => void;
    onPointerLeave: (event: PointerEvent) => void;
    onKeyDown: (event: KeyboardEvent) => void;
    onKeyUp: (event: KeyboardEvent) => void;
    onWheel: (event: WheelEvent) => void;
    reset(): void;
  };
  readonly onPageHide: () => void;
  readonly onWindowBlur: () => void;
  readonly onVisibilityChange: () => void;
  readonly clearPreview: () => void;
  readonly applyCursor: (value: string) => void;
  readonly updateCursor: () => void;
  readonly updateAnimation: () => void;
  readonly setViewportReady: (ready: boolean) => void;
}

/** Owns render targets, scheduling, DOM input bindings, and attach/detach lifecycle. */
export class RenderController {
  readonly scheduler: RenderScheduler;
  readonly previews = new PreviewStateController();
  private screen: RasterSurface | null = null;
  private overlay: RasterSurface | null = null;
  private input: HTMLCanvasElement | null = null;
  private disposed = false;

  constructor(private readonly options: RenderControllerOptions) {
    this.scheduler = createRenderScheduler(options);
  }

  getScreen(): RasterSurface | null {
    return this.screen;
  }

  getOverlay(): RasterSurface | null {
    return this.overlay;
  }

  getInputElement(): HTMLCanvasElement | null {
    return this.input;
  }

  attach(screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement): void {
    if (this.disposed || this.options.isEngineDisposed()) {
      return;
    }
    if (this.input) {
      this.detach();
    }
    this.screen = wrapCanvasSurface(screenCanvas);
    this.overlay = wrapCanvasSurface(overlayCanvas);
    this.input = overlayCanvas;
    const handlers = this.options.getInputHandlers();
    this.input.addEventListener('pointerdown', handlers.onPointerDown);
    this.input.addEventListener('pointermove', handlers.onPointerMove);
    this.input.addEventListener('pointerup', handlers.onPointerUp);
    this.input.addEventListener('pointercancel', handlers.onPointerCancel);
    this.input.addEventListener('pointerenter', handlers.onPointerEnter);
    this.input.addEventListener('pointerleave', handlers.onPointerLeave);
    this.input.addEventListener('wheel', handlers.onWheel, { passive: false });
    if (typeof globalThis.addEventListener === 'function') {
      globalThis.addEventListener('keydown', handlers.onKeyDown);
      globalThis.addEventListener('keyup', handlers.onKeyUp);
      globalThis.addEventListener('pagehide', this.options.onPageHide);
      globalThis.addEventListener('blur', this.options.onWindowBlur);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.options.onVisibilityChange);
    }
    this.scheduler.resume();
    this.options.setViewportReady(true);
    this.options.updateCursor();
    this.scheduler.invalidate({ all: true });
    this.options.updateAnimation();
  }

  detach(): void {
    const handlers = this.options.getInputHandlers();
    if (this.input) {
      this.input.removeEventListener('pointerdown', handlers.onPointerDown);
      this.input.removeEventListener('pointermove', handlers.onPointerMove);
      this.input.removeEventListener('pointerup', handlers.onPointerUp);
      this.input.removeEventListener('pointercancel', handlers.onPointerCancel);
      this.input.removeEventListener('pointerenter', handlers.onPointerEnter);
      this.input.removeEventListener('pointerleave', handlers.onPointerLeave);
      this.input.removeEventListener('wheel', handlers.onWheel);
      this.options.applyCursor('');
    }
    if (typeof globalThis.removeEventListener === 'function') {
      globalThis.removeEventListener('keydown', handlers.onKeyDown);
      globalThis.removeEventListener('keyup', handlers.onKeyUp);
      globalThis.removeEventListener('pagehide', this.options.onPageHide);
      globalThis.removeEventListener('blur', this.options.onWindowBlur);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.options.onVisibilityChange);
    }
    handlers.reset();
    this.options.clearPreview();
    this.scheduler.pause();
    this.options.setViewportReady(false);
    this.screen = null;
    this.overlay = null;
    this.input = null;
    this.options.updateAnimation();
  }

  resize(width: number, height: number): void {
    this.screen?.resize(width, height);
    this.overlay?.resize(width, height);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.detach();
    this.disposed = true;
    this.scheduler.dispose();
    this.previews.dispose();
  }
}
