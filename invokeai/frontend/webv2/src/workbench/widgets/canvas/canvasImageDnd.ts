import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

export interface CanvasImageDropData {
  kind: 'canvas-image-target';
  destination: Exclude<GalleryCanvasImportDestination, 'regional-guidance'>;
}

export const isCanvasImageDropData = (value: unknown): value is CanvasImageDropData =>
  isRecord(value) && value.kind === 'canvas-image-target' && isCanvasImageDropDestination(value.destination);

export const getCanvasImageDropData = (destination: CanvasImageDropData['destination']): CanvasImageDropData => ({
  destination,
  kind: 'canvas-image-target',
});

const canvasImageDropDestinations = {
  control: true,
  'control-resized': true,
  'inpaint-mask': true,
  raster: true,
  'regional-reference': true,
} satisfies Record<CanvasImageDropData['destination'], true>;

const isCanvasImageDropDestination = (value: unknown): value is CanvasImageDropData['destination'] =>
  typeof value === 'string' && Object.hasOwn(canvasImageDropDestinations, value);

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;
