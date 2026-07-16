import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const recordCanvasImportError = ({
  dispatch,
  error,
  localizedMessage,
  projectId,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  error: unknown;
  localizedMessage: string;
  projectId?: string;
}): void => {
  dispatch({
    area: 'image-actions',
    context: { error: toErrorMessage(error) },
    message: localizedMessage,
    namespace: 'gallery',
    projectId,
    type: 'recordError',
  });
};
