import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeSlashBold } from 'react-icons/pi';

export const StagingAreaToolbarToggleShowResultsButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const { t } = useTranslation();

  const toggleShowResults = useCallback(() => {
    canvasManager.stagingArea.$shouldShowStagedImage.set(!shouldShowStagedImage);
  }, [canvasManager.stagingArea.$shouldShowStagedImage, shouldShowStagedImage]);

  return (
    <IconButton
      tooltip={
        shouldShowStagedImage
          ? t('controlLayers.stagingArea.showResultsOn')
          : t('controlLayers.stagingArea.showResultsOff')
      }
      aria-label={
        shouldShowStagedImage
          ? t('controlLayers.stagingArea.showResultsOn')
          : t('controlLayers.stagingArea.showResultsOff')
      }
      data-alert={!shouldShowStagedImage}
      icon={shouldShowStagedImage ? <PiEyeBold /> : <PiEyeSlashBold />}
      onClick={toggleShowResults}
      colorScheme="invokeBlue"
    />
  );
});

StagingAreaToolbarToggleShowResultsButton.displayName = 'StagingAreaToolbarToggleShowResultsButton';
