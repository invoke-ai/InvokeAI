import type { WorkbenchNotificationCommands } from '@workbench/workbenchStore';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const recordCanvasImportError = ({
  notifications,
  error,
  localizedMessage,
  projectId,
}: {
  notifications: WorkbenchNotificationCommands;
  error: unknown;
  localizedMessage: string;
  projectId?: string;
}): void => {
  notifications.reportError({
    area: 'image-actions',
    context: { error: toErrorMessage(error) },
    message: localizedMessage,
    namespace: 'gallery',
    projectId,
  });
};
