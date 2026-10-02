import type { RefusedWorkbenchProject } from './projectContracts';

import { isRecord, loadCanvasState } from './canvasMigration';

const asString = (value: unknown): string => (typeof value === 'string' ? value : '');

/** Refuses a project before migrations can rewrite an unsupported live canvas. */
export const gateProjectCanvases = (raw: unknown): RefusedWorkbenchProject | null => {
  if (!isRecord(raw)) {
    return null;
  }
  const canvas = loadCanvasState(raw.canvas);
  if (canvas.status !== 'loaded') {
    return {
      projectId: asString(raw.id),
      projectName: asString(raw.name),
      raw,
      refusal: canvas,
      source: 'canvas',
    };
  }
  return null;
};
