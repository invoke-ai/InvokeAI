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
import { CanvasSettingsBboxOverlaySwitch } from 'features/controlLayers/components/Settings/CanvasSettingsBboxOverlaySwitch';
import { CanvasSettingsClearCachesButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearCachesButton';
import { CanvasSettingsClearHistoryButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearHistoryButton';
import { CanvasSettingsClipToBboxCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsClipToBboxCheckbox';
import { CanvasSettingsDynamicGridSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsDynamicGridSwitch';
import { CanvasSettingsSnapToGridCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsGridSize';
import { CanvasSettingsInvertScrollCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsInvertScrollCheckbox';
import { CanvasSettingsIsolatedFilteringPreviewSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsIsolatedFilteringPreviewSwitch';
import { CanvasSettingsIsolatedStagingPreviewSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsIsolatedStagingPreviewSwitch';
import { CanvasSettingsIsolatedTransformingPreviewSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsIsolatedTransformingPreviewSwitch';
import { CanvasSettingsLogDebugInfoButton } from 'features/controlLayers/components/Settings/CanvasSettingsLogDebugInfo';
import { CanvasSettingsOutputOnlyMaskedRegionsCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsOutputOnlyMaskedRegionsCheckbox';
import { CanvasSettingsPreserveMaskCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsPreserveMaskCheckbox';
import { CanvasSettingsPressureSensitivityCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsPressureSensitivity';
import { CanvasSettingsRecalculateRectsButton } from 'features/controlLayers/components/Settings/CanvasSettingsRecalculateRectsButton';
import { CanvasSettingsShowHUDSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsShowHUDSwitch';
import { CanvasSettingsShowProgressOnCanvas } from 'features/controlLayers/components/Settings/CanvasSettingsShowProgressOnCanvasSwitch';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

export const CanvasSettingsPopover = memo(() => {
  const { t } = useTranslation();
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          aria-label={t('common.settingsLabel')}
          icon={<PiGearSixFill />}
          variant="link"
          alignSelf="stretch"
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <CanvasSettingsInvertScrollCheckbox />
            <CanvasSettingsPreserveMaskCheckbox />
            <CanvasSettingsClipToBboxCheckbox />
            <CanvasSettingsOutputOnlyMaskedRegionsCheckbox />
            <CanvasSettingsSnapToGridCheckbox />
            <CanvasSettingsPressureSensitivityCheckbox />
            <CanvasSettingsShowProgressOnCanvas />
            <CanvasSettingsIsolatedStagingPreviewSwitch />
            <CanvasSettingsIsolatedFilteringPreviewSwitch />
            <CanvasSettingsIsolatedTransformingPreviewSwitch />
            <CanvasSettingsDynamicGridSwitch />
            <CanvasSettingsBboxOverlaySwitch />
            <CanvasSettingsShowHUDSwitch />
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
