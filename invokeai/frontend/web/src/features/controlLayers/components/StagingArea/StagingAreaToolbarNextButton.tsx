import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowRightBold } from 'react-icons/pi';

export const StagingAreaToolbarNextButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const itemCount = useStore(ctx.$itemCount);
  const isCanvasFocused = useIsRegionFocused('canvas');

  const { t } = useTranslation();

  const selectNext = useCallback(() => {
    ctx.selectNext();
  }, [ctx]);

  useHotkeys(
    ['right'],
    ctx.selectNext,
    {
      preventDefault: true,
      enabled: isCanvasFocused && !isDisabled && itemCount > 1,
    },
    [isCanvasFocused, isDisabled, itemCount, ctx.selectNext]
  );

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.next')} (Right)`}
      aria-label={`${t('controlLayers.stagingArea.next')} (Right)`}
      icon={<PiArrowRightBold />}
      onClick={selectNext}
      colorScheme="invokeBlue"
      isDisabled={itemCount <= 1 || isDisabled}
    />
  );
});

StagingAreaToolbarNextButton.displayName = 'StagingAreaToolbarNextButton';
