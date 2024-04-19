import { IconButton } from '@invoke-ai/ui-library';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBugBold } from 'react-icons/pi';

const debugBlobs = () => {
  getRegionalPromptLayerBlobs(undefined, true);
};
export const DebugLayersButton = memo(() => {
  const { t } = useTranslation();
  return (
    <IconButton
      colorScheme="warning"
      aria-label={t('regionalPrompts.debugLayers')}
      tooltip={t('regionalPrompts.debugLayers')}
      icon={<PiBugBold />}
      onClick={debugBlobs}
    />
  );
});

DebugLayersButton.displayName = 'DebugLayersButton';
