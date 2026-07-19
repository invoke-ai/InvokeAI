import type { WorkbenchNotificationCommands } from '@workbench/workbenchStore';

interface CanvasProjectIdentity {
  readonly projectId: string;
}

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const getCanvasGallerySaveErrorAction = (
  error: unknown,
  projectId: string,
  message: string
): Parameters<WorkbenchNotificationCommands['reportError']>[0] => ({
  area: 'canvas-save-to-gallery',
  context: { error: toErrorMessage(error) },
  message,
  namespace: 'canvas' as const,
  projectId,
});

export const withMatchingCanvasProject = <Engine extends CanvasProjectIdentity, Result>(
  engine: Engine | null,
  projectId: string,
  run: (engine: Engine) => Result
): Result | undefined => {
  if (!engine || engine.projectId !== projectId) {
    return undefined;
  }

  return run(engine);
};
