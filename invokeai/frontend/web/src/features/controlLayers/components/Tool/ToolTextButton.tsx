import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTextTBold } from 'react-icons/pi';

export const ToolTextButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('text');
  const selectText = useSelectTool('text');

  return (
    <Tooltip label={`${t('controlLayers.tool.text', { defaultValue: 'Text' })}`} placement="end">
      <IconButton
        aria-label={t('controlLayers.tool.text', { defaultValue: 'Text' })}
        icon={<PiTextTBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectText}
      />
    </Tooltip>
  );
});

ToolTextButton.displayName = 'ToolTextButton';
