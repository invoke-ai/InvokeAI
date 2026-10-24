import type { CanvasEngineLayerCapability } from '@workbench/canvas-engine/api';

import { MenuActionItem } from '@platform/ui/Menu';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { CropIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { cropAllLayersToFrame, hasCroppableLayers } from './canvasContextMenu';

/** Canvas-menu action that crops every unlocked layer to the frame as one undo step and reports the count. */
export const CanvasCropAllLayersMenuItem = ({
  disabled,
  layers,
}: {
  disabled: boolean;
  layers: Pick<CanvasEngineLayerCapability, 'cropAllLayersToBbox'>;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const hasCroppable = useActiveProjectSelector((project) => hasCroppableLayers(project.canvas.document));
  const onSelect = useCallback(() => void cropAllLayersToFrame(layers, notify, t), [layers, notify, t]);

  return (
    <MenuActionItem
      disabled={disabled || !hasCroppable}
      icon={CropIcon}
      label={t('widgets.canvas.contextMenu.cropAllLayersToBbox')}
      value="crop-all-layers-to-bbox"
      onSelect={onSelect}
    />
  );
};
