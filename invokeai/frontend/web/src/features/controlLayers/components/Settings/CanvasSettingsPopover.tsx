import {
  Divider,
  Flex,
  Icon,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
  useShiftModifier,
} from '@invoke-ai/ui-library';
import { CanvasSettingsBboxOverlaySwitch } from 'features/controlLayers/components/Settings/CanvasSettingsBboxOverlaySwitch';
import { CanvasSettingsClearCachesButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearCachesButton';
import { CanvasSettingsClearHistoryButton } from 'features/controlLayers/components/Settings/CanvasSettingsClearHistoryButton';
import { CanvasSettingsClipToBboxCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsClipToBboxCheckbox';
import { CanvasSettingsDynamicGridSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsDynamicGridSwitch';
import { CanvasSettingsSnapToGridCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsGridSize';
import { CanvasSettingsInvertScrollCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsInvertScrollCheckbox';
import { CanvasSettingsIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsIsolatedLayerPreviewSwitch';
import { CanvasSettingsIsolatedStagingPreviewSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsIsolatedStagingPreviewSwitch';
import { CanvasSettingsLogDebugInfoButton } from 'features/controlLayers/components/Settings/CanvasSettingsLogDebugInfo';
import { CanvasSettingsOutputOnlyMaskedRegionsCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsOutputOnlyMaskedRegionsCheckbox';
import { CanvasSettingsPreserveMaskCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsPreserveMaskCheckbox';
import { CanvasSettingsPressureSensitivityCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsPressureSensitivity';
import { CanvasSettingsRecalculateRectsButton } from 'features/controlLayers/components/Settings/CanvasSettingsRecalculateRectsButton';
import { CanvasSettingsRuleOfThirdsSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsRuleOfThirdsGuideSwitch';
import { CanvasSettingsSaveAllImagesToGalleryCheckbox } from 'features/controlLayers/components/Settings/CanvasSettingsSaveAllImagesToGalleryCheckbox';
import { CanvasSettingsShowHUDSwitch } from 'features/controlLayers/components/Settings/CanvasSettingsShowHUDSwitch';
import { CanvasSettingsShowProgressOnCanvas } from 'features/controlLayers/components/Settings/CanvasSettingsShowProgressOnCanvasSwitch';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCodeFill, PiEyeFill, PiGearSixFill, PiPencilFill, PiSquaresFourFill } from 'react-icons/pi';

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
      <PopoverContent maxW="280px">
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            {/* Behavior Settings */}
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={2}>
                <Icon as={PiPencilFill} boxSize={4} />
                <Text fontWeight="bold" fontSize="sm" color="base.100">
                  {t('hotkeys.canvas.settings.behavior')}
                </Text>
              </Flex>
              <CanvasSettingsInvertScrollCheckbox />
              <CanvasSettingsPressureSensitivityCheckbox />
              <CanvasSettingsPreserveMaskCheckbox />
              <CanvasSettingsClipToBboxCheckbox />
              <CanvasSettingsOutputOnlyMaskedRegionsCheckbox />
              <CanvasSettingsSaveAllImagesToGalleryCheckbox />
            </Flex>

            <Divider />

            {/* Display Settings */}
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={2} color="base.200">
                <Icon as={PiEyeFill} boxSize={4} />
                <Text fontWeight="bold" fontSize="sm">
                  {t('hotkeys.canvas.settings.display')}
                </Text>
              </Flex>
              <CanvasSettingsShowProgressOnCanvas />
              <CanvasSettingsIsolatedStagingPreviewSwitch />
              <CanvasSettingsIsolatedLayerPreviewSwitch />
              <CanvasSettingsBboxOverlaySwitch />
              <CanvasSettingsShowHUDSwitch />
            </Flex>

            <Divider />

            {/* Grid Settings */}
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={2} color="base.200">
                <Icon as={PiSquaresFourFill} boxSize={4} />
                <Text fontWeight="bold" fontSize="sm">
                  {t('hotkeys.canvas.settings.grid')}
                </Text>
              </Flex>
              <CanvasSettingsSnapToGridCheckbox />
              <CanvasSettingsDynamicGridSwitch />
              <CanvasSettingsRuleOfThirdsSwitch />
            </Flex>

            <DebugSettings />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

CanvasSettingsPopover.displayName = 'CanvasSettingsPopover';

const DebugSettings = () => {
  const { t } = useTranslation();
  const shift = useShiftModifier();

  if (!shift) {
    return null;
  }

  return (
    <>
      <Divider />
      <Flex direction="column" gap={1}>
        <Flex align="center" gap={2} color="base.200">
          <Icon as={PiCodeFill} boxSize={4} />
          <Text fontWeight="bold" fontSize="sm">
            {t('hotkeys.canvas.settings.debug')}
          </Text>
        </Flex>
        <CanvasSettingsClearCachesButton />
        <CanvasSettingsRecalculateRectsButton />
        <CanvasSettingsLogDebugInfoButton />
        <CanvasSettingsClearHistoryButton />
      </Flex>
    </>
  );
};
