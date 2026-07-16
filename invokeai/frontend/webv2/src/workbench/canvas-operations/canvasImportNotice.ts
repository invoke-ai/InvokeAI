import type { ImportGalleryImagesResult } from './importGalleryImages';

export interface CanvasImportNotice {
  kind: 'error' | 'info' | 'success';
  titleKey:
    | 'widgets.canvas.import.blocked'
    | 'widgets.canvas.import.empty'
    | 'widgets.canvas.import.partial'
    | 'widgets.canvas.import.staleDocument'
    | 'widgets.canvas.import.staleProject'
    | 'widgets.canvas.import.success';
  options?: { count: number } | { failedCount: number; successCount: number };
}

const assertNever = (value: never): never => {
  throw new Error(`Unhandled canvas import result: ${JSON.stringify(value)}`);
};

export const getCanvasImportNotice = (result: ImportGalleryImagesResult): CanvasImportNotice => {
  switch (result.status) {
    case 'imported':
      if (result.failedImageNames.length > 0) {
        return {
          kind: 'info',
          options: { failedCount: result.failedImageNames.length, successCount: result.layerIds.length },
          titleKey: 'widgets.canvas.import.partial',
        };
      }
      return {
        kind: 'success',
        options: { count: result.layerIds.length },
        titleKey: 'widgets.canvas.import.success',
      };
    case 'blocked':
      return { kind: 'info', titleKey: 'widgets.canvas.import.blocked' };
    case 'empty':
      return { kind: 'info', titleKey: 'widgets.canvas.import.empty' };
    case 'stale-document':
      return { kind: 'error', titleKey: 'widgets.canvas.import.staleDocument' };
    case 'stale-project':
      return { kind: 'error', titleKey: 'widgets.canvas.import.staleProject' };
    default:
      return assertNever(result);
  }
};
