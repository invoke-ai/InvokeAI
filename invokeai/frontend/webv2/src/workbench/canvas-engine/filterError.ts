import type { Rect } from '@workbench/canvas-engine/types';

const filterName = (filterType: string): string =>
  filterType
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

export class LayerFilterOutputDimensionError extends Error {
  readonly code = 'output-dimension' as const;

  constructor(filterType: string, output: { width: number; height: number }, source: Rect) {
    super(
      `${filterName(filterType)} output dimensions ${String(output.width)}x${String(output.height)} do not match source dimensions ${source.width}x${source.height}.`
    );
    this.name = 'LayerFilterOutputDimensionError';
  }
}
