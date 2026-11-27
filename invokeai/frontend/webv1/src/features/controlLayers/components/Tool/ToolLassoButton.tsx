import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLassoBold } from 'react-icons/pi';

export const ToolLassoButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('lasso');
  const selectLasso = useSelectTool('lasso');

  useRegisteredHotkeys({
    id: 'selectLassoTool',
    category: 'canvas',
    callback: selectLasso,
    options: { enabled: !isSelected },
    dependencies: [isSelected, selectLasso],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.lasso', { defaultValue: 'Lasso' })} (L)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.lasso', { defaultValue: 'Lasso' })} (L)`}
        icon={<PiLassoBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectLasso}
      />
    </Tooltip>
  );
});

ToolLassoButton.displayName = 'ToolLassoButton';
