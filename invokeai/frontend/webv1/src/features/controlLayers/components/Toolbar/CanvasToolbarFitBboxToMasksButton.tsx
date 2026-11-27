import { IconButton } from '@invoke-ai/ui-library';
import { useAutoFitBBoxToMasks } from 'features/controlLayers/hooks/useAutoFitBBoxToMasks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useVisibleEntityCountByType } from 'features/controlLayers/hooks/useVisibleEntityCountByType';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionAllDuotone } from 'react-icons/pi';

export const CanvasToolbarFitBboxToMasksButton = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const fitBBoxToMasks = useAutoFitBBoxToMasks();

  // Check if there are any visible inpaint masks
  const visibleMaskCount = useVisibleEntityCountByType('inpaint_mask');
  const hasVisibleMasks = visibleMaskCount > 0;

  const onClick = useCallback(() => {
    fitBBoxToMasks();
  }, [fitBBoxToMasks]);

  // Register hotkey for Shift+B
  useRegisteredHotkeys({
    id: 'fitBboxToMasks',
    category: 'canvas',
    callback: fitBBoxToMasks,
    options: { enabled: !isBusy && hasVisibleMasks },
    dependencies: [fitBBoxToMasks, isBusy, hasVisibleMasks],
  });

  return (
    <IconButton
      onClick={onClick}
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.fitBboxToMasks')}
      tooltip={t('controlLayers.fitBboxToMasks')}
      icon={<PiSelectionAllDuotone />}
      isDisabled={isBusy || !hasVisibleMasks}
    />
  );
});

CanvasToolbarFitBboxToMasksButton.displayName = 'CanvasToolbarFitBboxToMasksButton';
