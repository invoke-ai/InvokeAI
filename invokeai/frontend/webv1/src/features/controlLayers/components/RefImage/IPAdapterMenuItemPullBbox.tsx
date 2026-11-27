import { MenuItem } from '@invoke-ai/ui-library';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { usePullBboxIntoGlobalReferenceImage } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const IPAdapterMenuItemPullBbox = memo(() => {
  const { t } = useTranslation();
  const id = useRefImageIdContext();
  const pullBboxIntoIPAdapter = usePullBboxIntoGlobalReferenceImage(id);
  const isBusy = useCanvasIsBusy();

  return (
    <MenuItem onClick={pullBboxIntoIPAdapter} icon={<PiBoundingBoxBold />} isDisabled={isBusy}>
      {t('controlLayers.pullBboxIntoReferenceImage')}
    </MenuItem>
  );
});

IPAdapterMenuItemPullBbox.displayName = 'IPAdapterMenuItemPullBbox';
