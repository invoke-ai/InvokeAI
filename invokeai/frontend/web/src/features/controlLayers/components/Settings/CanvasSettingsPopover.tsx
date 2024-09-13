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
import { CanvasSettingsAutoSaveCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsAutoSaveCheckbox';
import { CanvasSettingsClearCachesButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearCachesButton';
import { CanvasSettingsClearHistoryButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearHistoryButton';
import { CanvasSettingsClipToBboxCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsClipToBboxCheckbox';
import { CanvasSettingsCompositeMaskedRegionsCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsCompositeMaskedRegionsCheckbox';
import { CanvasSettingsDynamicGridSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsDynamicGridSwitch';
import { CanvasSettingsSnapToGridCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsGridSize';
import { CanvasSettingsInvertScrollCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsInvertScrollCheckbox';
import { CanvasSettingsLogDebugInfoButton } from 'features/controlLayers/components/Settings/CanvasSettingsLogDebugInfo';
import { CanvasSettingsRecalculateRectsButton } from 'features/controlLayers/components/Settings/CanvasSettingsRecalculateRectsButton';
import { CanvasSettingsResetButton } from 'features/controlLayers/components/Settings/CanvasSettingsResetButton';
import { CanvasSettingsShowHUDSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsShowHUDSwitch';
import { CanvasSettingsShowProgressOnCanvas } from 'features/controlLayers/components/Settings/CanvasSettingsShowProgressOnCanvasSwitch';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

export const CanvasSettingsPopover = memo(() => {
  const { t } = useTranslation();
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('common.settingsLabel')} icon={<RiSettings4Fill />} variant="ghost" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <CanvasSettingsAutoSaveCheckbox />
            <CanvasSettingsInvertScrollCheckbox />
            <CanvasSettingsClipToBboxCheckbox />
            <CanvasSettingsCompositeMaskedRegionsCheckbox />
            <CanvasSettingsSnapToGridCheckbox />
            <CanvasSettingsShowProgressOnCanvas />
            <CanvasSettingsDynamicGridSwitch />
            <CanvasSettingsShowHUDSwitch />
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
      <CanvasSettingsLogDebugInfoButton />
      <CanvasSettingsClearHistoryButton />
    </>
  );
};
