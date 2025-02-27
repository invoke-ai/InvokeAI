import {
  Button,
  ButtonGroup,
  Flex,
  Heading,
  Icon,
  ListItem,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Spacer,
  Spinner,
  Text,
  Tooltip,
  UnorderedList,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasAutoProcessSwitch } from 'features/controlLayers/components/CanvasAutoProcessSwitch';
import { CanvasOperationIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/CanvasOperationIsolatedLayerPreviewSwitch';
import { SelectObjectInvert } from 'features/controlLayers/components/SelectObject/SelectObjectInvert';
import { SelectObjectPointType } from 'features/controlLayers/components/SelectObject/SelectObjectPointType';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useRef } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiInfoBold } from 'react-icons/pi';

const SelectObjectContent = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);
    useFocusRegion('canvas', ref, { focusOnMount: true });
    const isCanvasFocused = useIsRegionFocused('canvas');
    const isProcessing = useStore(adapter.segmentAnything.$isProcessing);
    const hasPoints = useStore(adapter.segmentAnything.$hasPoints);
    const hasImageState = useStore(adapter.segmentAnything.$hasImageState);
    const autoProcess = useAppSelector(selectAutoProcess);

    const saveAsInpaintMask = useCallback(() => {
      adapter.segmentAnything.saveAs('inpaint_mask');
    }, [adapter.segmentAnything]);

    const saveAsRegionalGuidance = useCallback(() => {
      adapter.segmentAnything.saveAs('regional_guidance');
    }, [adapter.segmentAnything]);

    const saveAsRasterLayer = useCallback(() => {
      adapter.segmentAnything.saveAs('raster_layer');
    }, [adapter.segmentAnything]);

    const saveAsControlLayer = useCallback(() => {
      adapter.segmentAnything.saveAs('control_layer');
    }, [adapter.segmentAnything]);

    useRegisteredHotkeys({
      id: 'applySegmentAnything',
      category: 'canvas',
      callback: adapter.segmentAnything.apply,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.segmentAnything, isProcessing, isCanvasFocused],
    });

    useRegisteredHotkeys({
      id: 'cancelSegmentAnything',
      category: 'canvas',
      callback: adapter.segmentAnything.cancel,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.segmentAnything, isProcessing, isCanvasFocused],
    });

    return (
      <Flex
        ref={ref}
        bg="base.800"
        borderRadius="base"
        p={4}
        flexDir="column"
        gap={4}
        minW={420}
        h="auto"
        shadow="dark-lg"
        transitionProperty="height"
        transitionDuration="normal"
      >
        <Flex w="full" gap={4} alignItems="center">
          <Flex gap={2}>
            <Heading size="md" color="base.300" userSelect="none">
              {t('controlLayers.selectObject.selectObject')}
            </Heading>
            <Tooltip label={<SelectObjectHelpTooltipContent />}>
              <Flex alignItems="center">
                <Icon as={PiInfoBold} color="base.500" />
              </Flex>
            </Tooltip>
          </Flex>
          <Spacer />
          <CanvasAutoProcessSwitch />
          <CanvasOperationIsolatedLayerPreviewSwitch />
        </Flex>

        <Flex w="full" justifyContent="space-between" py={2}>
          <SelectObjectPointType adapter={adapter} />
          <SelectObjectInvert adapter={adapter} />
        </Flex>

        <ButtonGroup isAttached={false} size="sm" w="full">
          <Button
            onClick={adapter.segmentAnything.processImmediate}
            loadingText={t('controlLayers.selectObject.process')}
            variant="ghost"
            isDisabled={isProcessing || !hasPoints || (autoProcess && hasImageState)}
          >
            {t('controlLayers.selectObject.process')}
            {isProcessing && <Spinner ms={3} boxSize={5} color="base.600" />}
          </Button>
          <Spacer />
          <Button
            onClick={adapter.segmentAnything.reset}
            isDisabled={isProcessing || !hasPoints}
            loadingText={t('controlLayers.selectObject.reset')}
            variant="ghost"
          >
            {t('controlLayers.selectObject.reset')}
          </Button>
          <Button
            onClick={adapter.segmentAnything.apply}
            loadingText={t('controlLayers.selectObject.apply')}
            variant="ghost"
            isDisabled={isProcessing || !hasImageState}
          >
            {t('controlLayers.selectObject.apply')}
          </Button>
          <Menu>
            <MenuButton
              as={Button}
              loadingText={t('controlLayers.selectObject.saveAs')}
              variant="ghost"
              isDisabled={isProcessing || !hasImageState}
              rightIcon={<PiCaretDownBold />}
            >
              {t('controlLayers.selectObject.saveAs')}
            </MenuButton>
            <MenuList>
              <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsInpaintMask}>
                {t('controlLayers.newInpaintMask')}
              </MenuItem>
              <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsRegionalGuidance}>
                {t('controlLayers.newRegionalGuidance')}
              </MenuItem>
              <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsControlLayer}>
                {t('controlLayers.newControlLayer')}
              </MenuItem>
              <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsRasterLayer}>
                {t('controlLayers.newRasterLayer')}
              </MenuItem>
            </MenuList>
          </Menu>
          <Button
            onClick={adapter.segmentAnything.cancel}
            isDisabled={isProcessing}
            loadingText={t('common.cancel')}
            variant="ghost"
          >
            {t('controlLayers.selectObject.cancel')}
          </Button>
        </ButtonGroup>
      </Flex>
    );
  }
);

SelectObjectContent.displayName = 'SegmentAnythingContent';

export const SelectObject = memo(() => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$segmentingAdapter);

  if (!adapter) {
    return null;
  }

  return <SelectObjectContent adapter={adapter} />;
});

SelectObject.displayName = 'SelectObject';

const Bold = (props: PropsWithChildren) => (
  <Text as="span" fontWeight="semibold">
    {props.children}
  </Text>
);

const SelectObjectHelpTooltipContent = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex gap={3} flexDir="column">
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help1" components={{ Bold: <Bold /> }} />
      </Text>
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help2" components={{ Bold: <Bold /> }} />
      </Text>
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help3" />
      </Text>
      <UnorderedList>
        <ListItem>{t('controlLayers.selectObject.clickToAdd')}</ListItem>
        <ListItem>{t('controlLayers.selectObject.dragToMove')}</ListItem>
        <ListItem>{t('controlLayers.selectObject.clickToRemove')}</ListItem>
      </UnorderedList>
    </Flex>
  );
});

SelectObjectHelpTooltipContent.displayName = 'SelectObjectHelpTooltipContent';
