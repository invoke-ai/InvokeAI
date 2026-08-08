import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBezierCurveBold } from 'react-icons/pi';

export const ToolPathButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('path');
  const selectPath = useSelectTool('path');
  const label = t('controlLayers.tool.path', { defaultValue: 'Path' });

  useRegisteredHotkeys({
    id: 'selectPathTool',
    category: 'canvas',
    callback: selectPath,
    options: { enabled: !isSelected },
    dependencies: [isSelected, selectPath],
  });

  return (
    <Tooltip label={`${label} (P)`} placement="end">
      <IconButton
        aria-label={`${label} (P)`}
        icon={<PiBezierCurveBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectPath}
      />
    </Tooltip>
  );
});

ToolPathButton.displayName = 'ToolPathButton';
