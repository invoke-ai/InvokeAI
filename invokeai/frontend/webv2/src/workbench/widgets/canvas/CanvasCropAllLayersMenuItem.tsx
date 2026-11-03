import type { CanvasEngineLayerCapability } from '@workbench/canvas-engine/api';

import { MenuActionItem } from '@platform/ui/Menu';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { CropIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { cropAllLayersToFrame, getCropAllUnavailableReasonKey } from './canvasContextMenu';

/**
 * Canvas-menu action that crops every unlocked layer to the frame as one undo step and reports the count. While it
 * cannot run it stays reachable by keyboard and says why.
 */
export const CanvasCropAllLayersMenuItem = ({
  documentEditingLocked,
  layers,
}: {
  /** An operation (Filter, Select Object) holds the document. */
  documentEditingLocked: boolean;
  layers: Pick<CanvasEngineLayerCapability, 'cropAllLayersToBbox'>;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const documentReasonKey = useActiveProjectSelector((project) =>
    getCropAllUnavailableReasonKey(project.canvas.document)
  );
  const reasonKey = documentEditingLocked ? 'widgets.layers.actions.operationLocked' : documentReasonKey;
  const onSelect = useCallback(() => void cropAllLayersToFrame(layers, notify, t), [layers, notify, t]);

  return (
    <MenuActionItem
      icon={CropIcon}
      label={t('widgets.canvas.contextMenu.cropAllLayersToBbox')}
      unavailableReason={reasonKey ? t(reasonKey) : undefined}
      value="crop-all-layers-to-bbox"
      onSelect={onSelect}
    />
  );
};
