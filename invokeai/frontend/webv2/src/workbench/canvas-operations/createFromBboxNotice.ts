import type { CanvasImportNotice } from './canvasImportNotice';
import type { CreateFromBboxLayerDestination, CreateFromBboxResult } from './createFromBbox';

import { getCanvasImportNotice } from './canvasImportNotice';

export interface CreateFromBboxNotice {
  kind: 'error' | 'info' | 'success';
  titleKey:
    | CanvasImportNotice['titleKey']
    | 'widgets.canvas.contextMenu.createdFromBbox'
    | 'widgets.canvas.contextMenu.empty'
    | 'widgets.canvas.contextMenu.notReady'
    | 'widgets.canvas.contextMenu.stale';
  /** `destination` is itself a translation key naming the created entity. */
  options?: CanvasImportNotice['options'] | { destination: string };
}

const DESTINATION_LABEL_KEYS: Record<CreateFromBboxLayerDestination, string> = {
  control: 'widgets.canvas.import.control',
  raster: 'widgets.canvas.import.raster',
  'regional-reference': 'widgets.canvas.import.regionalReference',
};

const assertNever = (value: never): never => {
  throw new Error(`Unhandled create-from-bbox result: ${JSON.stringify(value)}`);
};

export const getCreateFromBboxNotice = (
  result: Exclude<CreateFromBboxResult, { status: 'uploaded' }>
): CreateFromBboxNotice => {
  switch (result.status) {
    case 'created':
      return {
        kind: 'success',
        options: { destination: DESTINATION_LABEL_KEYS[result.destination] },
        titleKey: 'widgets.canvas.contextMenu.createdFromBbox',
      };
    case 'empty':
      return { kind: 'info', titleKey: 'widgets.canvas.contextMenu.empty' };
    case 'stale':
      return { kind: 'info', titleKey: 'widgets.canvas.contextMenu.stale' };
    case 'not-ready':
    case 'over-budget':
      return { kind: 'info', titleKey: 'widgets.canvas.contextMenu.notReady' };
    case 'blocked':
    case 'stale-document':
    case 'stale-project':
      return getCanvasImportNotice({ status: result.status });
    default:
      return assertNever(result);
  }
};
