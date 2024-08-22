import {
  Divider,
  Flex,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  useShiftModifier,
} from '@invoke-ai/ui-library';
import { CanvasSettingsClearCachesButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearCachesButton';
import { CanvasSettingsClipToBboxCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsClipToBboxCheckbox';
import { CanvasSettingsDynamicGridSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsDynamicGridSwitch';
import { CanvasSettingsInvertScrollCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsInvertScrollCheckbox';
import { CanvasSettingsRecalculateRectsButton } from 'features/controlLayers/components/Settings/CanvasSettingsRecalculateRectsButton';
import { CanvasSettingsResetButton } from 'features/controlLayers/components/Settings/CanvasSettingsResetButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

export const CanvasSettingsPopover = memo(() => {
  const { t } = useTranslation();
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('common.settingsLabel')} icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <CanvasSettingsInvertScrollCheckbox />
            <CanvasSettingsClipToBboxCheckbox />
            <CanvasSettingsDynamicGridSwitch />
            <CanvasSettingsResetButton />
            <DebugSettings />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

CanvasSettingsPopover.displayName = 'CanvasSettingsPopover';

const DebugSettings = () => {
  const shift = useShiftModifier();

  if (!shift) {
    return null;
  }

  return (
    <>
      <Divider />
      <CanvasSettingsClearCachesButton />
      <CanvasSettingsRecalculateRectsButton />
    </>
  );
};
