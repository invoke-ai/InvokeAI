import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityTypeCount } from 'features/controlLayers/hooks/useEntityTypeCount';
import { useMergeVisible } from 'features/controlLayers/hooks/useMergeVisible';
import type { CanvasRenderableEntityType } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStackBold } from 'react-icons/pi';

type Props = {
  type: CanvasRenderableEntityType;
};

export const CanvasEntityMergeVisibleButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const entityCount = useEntityTypeCount(type);
  const mergeVisible = useMergeVisible(type);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.mergeVisible')}
      tooltip={t('controlLayers.mergeVisible')}
      variant="link"
      icon={<PiStackBold />}
      onClick={mergeVisible}
      alignSelf="stretch"
      isDisabled={entityCount <= 1 || isBusy}
    />
  );
});

CanvasEntityMergeVisibleButton.displayName = 'CanvasEntityMergeVisibleButton';
