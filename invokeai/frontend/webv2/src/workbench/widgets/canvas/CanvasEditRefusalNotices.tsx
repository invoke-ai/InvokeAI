import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { useNotify } from '@workbench/useNotify';
import { reportStructuralCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useTranslation } from 'react-i18next';

/** Explains edits the engine refused without a caller to report to, such as a stroke too large to undo. Keyed by engine. */
export const CanvasEditRefusalNotices = ({ engine }: { engine: Pick<CanvasEngineHandle, 'tools'> }) => {
  const notify = useNotify();
  const { t } = useTranslation();
  useMountEffect(() =>
    engine.tools.onEditRefused((status) =>
      // Elsewhere `busy` stays silent because its controls are disabled; a stroke can still meet it mid-gesture.
      status === 'busy'
        ? notify.error(t('widgets.canvas.structural.failed'), t('widgets.canvas.structural.busy'))
        : reportStructuralCommit({ status }, notify.error, t)
    )
  );
  return null;
};
