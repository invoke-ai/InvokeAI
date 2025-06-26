import {
  Divider,
  Flex,
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
          <Flex direction="column" gap={3}>
            {/* Behavior Settings */}
            <Flex direction="column" gap={2}>
              <Flex align="center" gap={2} mb={1}>
                <PiPencilFill size={16} />
                <Text fontWeight="bold" fontSize="sm" color="base.200">
                  {t('hotkeys.canvas.settings.behavior')}
                </Text>
              </Flex>
              <Flex direction="column" gap={2} pl={6}>
                <CanvasSettingsInvertScrollCheckbox />
                <CanvasSettingsPressureSensitivityCheckbox />
                <CanvasSettingsPreserveMaskCheckbox />
                <CanvasSettingsClipToBboxCheckbox />
                <CanvasSettingsOutputOnlyMaskedRegionsCheckbox />
              </Flex>
            </Flex>

            <Divider />

            {/* Display Settings */}
            <Flex direction="column" gap={2}>
              <Flex align="center" gap={2} mb={1}>
                <PiEyeFill size={16} />
                <Text fontWeight="bold" fontSize="sm" color="base.200">
                  {t('hotkeys.canvas.settings.display')}
                </Text>
              </Flex>
              <Flex direction="column" gap={2} pl={6}>
                <CanvasSettingsShowProgressOnCanvas />
                <CanvasSettingsIsolatedStagingPreviewSwitch />
                <CanvasSettingsIsolatedLayerPreviewSwitch />
                <CanvasSettingsBboxOverlaySwitch />
                <CanvasSettingsShowHUDSwitch />
              </Flex>
            </Flex>

            <Divider />

            {/* Grid Settings */}
            <Flex direction="column" gap={2}>
              <Flex align="center" gap={2} mb={1}>
                <PiSquaresFourFill size={16} />
                <Text fontWeight="bold" fontSize="sm" color="base.200">
                  {t('hotkeys.canvas.settings.grid')}
                </Text>
              </Flex>
              <Flex direction="column" gap={2} pl={6}>
                <CanvasSettingsSnapToGridCheckbox />
                <CanvasSettingsDynamicGridSwitch />
                <CanvasSettingsRuleOfThirdsSwitch />
              </Flex>
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
      <Flex direction="column" gap={2}>
        <Flex align="center" gap={2} mb={1}>
          <PiCodeFill size={16} />
          <Text fontWeight="bold" fontSize="sm" color="base.200">
            {t('hotkeys.canvas.settings.debug')}
          </Text>
        </Flex>
        <Flex direction="column" gap={2} pl={6}>
          <CanvasSettingsClearCachesButton />
          <CanvasSettingsRecalculateRectsButton />
          <CanvasSettingsLogDebugInfoButton />
          <CanvasSettingsClearHistoryButton />
        </Flex>
      </Flex>
    </>
  );
};
