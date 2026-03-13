import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const selectBbox = useSelectTool('bbox');
  const isSelected = useToolIsSelected('bbox');

  useRegisteredHotkeys({
    id: 'selectBboxTool',
    category: 'canvas',
    callback: selectBbox,
    options: { enabled: !isSelected },
    dependencies: [selectBbox, isSelected],
  });

  return (
    <IAITooltip label={`${t('controlLayers.tool.bbox')} (C)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.bbox')} (C)`}
        icon={<PiBoundingBoxBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectBbox}
      />
    </IAITooltip>
  );
});

ToolBboxButton.displayName = 'ToolBboxButton';
