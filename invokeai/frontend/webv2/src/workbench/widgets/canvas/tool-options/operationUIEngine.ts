import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';

/** Narrow UI view of application-owned filter and Select Object coordination. */
export type CanvasOperationUIEngine = Pick<CanvasEngineHandle, 'interaction' | 'projectId'>;
