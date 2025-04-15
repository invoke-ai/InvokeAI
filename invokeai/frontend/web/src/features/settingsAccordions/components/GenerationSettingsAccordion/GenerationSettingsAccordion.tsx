import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  Expander,
  Flex,
  FormControlGroup,
  FormLabel,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Spacer,
  StandaloneAccordion,
  Text,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ImperativeModelPickerHandle, OptionGroup } from 'common/components/Picker/Picker';
import { isMatch, Picker } from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
import { typedMemo } from 'common/util/typedMemo';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectIsCogView4, selectIsFLUX, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamGuidance from 'features/parameters/components/Core/ParamGuidance';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import ParamMainModelSelect from 'features/parameters/components/MainModel/ParamMainModelSelect';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import ParamUpscaleCFGScale from 'features/parameters/components/Upscale/ParamUpscaleCFGScale';
import ParamUpscaleScheduler from 'features/parameters/components/Upscale/ParamUpscaleScheduler';
import { modelSelected } from 'features/parameters/store/actions';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { filesize } from 'filesize';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';
import { isFluxFillMainModelModelConfig } from 'services/api/types';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

export const GenerationSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const modelConfig = useSelectedModelConfig();
  const activeTabName = useAppSelector(selectActiveTab);
  const isFLUX = useAppSelector(selectIsFLUX);
  const isSD3 = useAppSelector(selectIsSD3);
  const isCogView4 = useAppSelector(selectIsCogView4);

  const isUpscaling = useMemo(() => {
    return activeTabName === 'upscaling';
  }, [activeTabName]);
  const selectBadges = useMemo(
    () =>
      createMemoizedSelector(selectLoRAsSlice, (loras) => {
        const enabledLoRAsCount = loras.loras.filter((l) => l.isEnabled).length;
        const loraTabBadges = enabledLoRAsCount ? [`${enabledLoRAsCount} ${t('models.concepts')}`] : EMPTY_ARRAY;
        const accordionBadges = modelConfig ? [modelConfig.name, modelConfig.base] : EMPTY_ARRAY;
        return { loraTabBadges, accordionBadges };
      }),
    [modelConfig, t]
  );
  const { loraTabBadges, accordionBadges } = useAppSelector(selectBadges);
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'generation-settings-advanced',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: `generation-settings-${activeTabName}`,
    defaultIsOpen: activeTabName !== 'upscaling',
  });

  return (
    <StandaloneAccordion
      label={t('accordions.generation.title')}
      badges={[...accordionBadges, ...loraTabBadges]}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Box px={4} pt={4} data-testid="generation-accordion">
        <Flex gap={4} flexDir="column">
          <ParamMainModelSelect />
          <MainModelPicker />
          <LoRASelect />
          <LoRAList />
        </Flex>
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} flexDir="column" pb={4}>
            <FormControlGroup formLabelProps={formLabelProps}>
              {!isFLUX && !isSD3 && !isCogView4 && !isUpscaling && <ParamScheduler />}
              {isUpscaling && <ParamUpscaleScheduler />}
              <ParamSteps />
              {isFLUX && modelConfig && !isFluxFillMainModelModelConfig(modelConfig) && <ParamGuidance />}
              {isUpscaling && <ParamUpscaleCFGScale />}
              {!isFLUX && !isUpscaling && <ParamCFGScale />}
            </FormControlGroup>
          </Flex>
        </Expander>
      </Box>
    </StandaloneAccordion>
  );
});

GenerationSettingsAccordion.displayName = 'GenerationSettingsAccordion';

const getId = (modelConfig: AnyModelConfig) => modelConfig.key;
const getIsDisabled = (modelConfig: AnyModelConfig) => {
  return modelConfig.base === 'flux';
};

const MainModelPicker = memo(() => {
  const { t } = useTranslation();
  const [modelConfigs] = useMainModels();
  const grouped = useMemo<OptionGroup<AnyModelConfig>[]>(() => {
    const groups: { [base in BaseModelType]?: OptionGroup<AnyModelConfig, { name: string; description: string }> } = {};

    for (const modelConfig of modelConfigs) {
      let group = groups[modelConfig.base];
      if (!group) {
        group = {
          id: modelConfig.base,
          data: { name: modelConfig.base, description: 'asdfasdf' },
          options: [],
        };
        groups[modelConfig.base] = group;
      }

      group.options.push(modelConfig);
    }

    return Object.values(groups);
  }, [modelConfigs]);
  const modelConfig = useSelectedModelConfig();
  const popover = useDisclosure(false);
  const pickerRef = useRef<ImperativeModelPickerHandle>(null);
  const dispatch = useAppDispatch();

  const onClose = useCallback(() => {
    popover.close();
    pickerRef.current?.setSearchTerm('');
  }, [popover]);

  const onSelect = useCallback(
    (model: AnyModelConfig) => {
      dispatch(modelSelected(model));
      onClose();
    },
    [dispatch, onClose]
  );

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={onClose}
      initialFocusRef={pickerRef.current?.inputRef}
    >
      <Flex alignItems="center" gap={2}>
        <InformationalPopover feature="paramModel">
          <FormLabel>{t('modelManager.model')}</FormLabel>
        </InformationalPopover>
        <PopoverTrigger>
          <Button size="sm" flexGrow={1} variant="outline">
            {modelConfig?.name ?? 'Select Model'}
            <Spacer />
            <PiCaretDownBold />
          </Button>
        </PopoverTrigger>
        <NavigateToModelManagerButton />
        <UseDefaultSettingsButton />
      </Flex>
      <PopoverContent p={0} w={400} h={400}>
        <PopoverArrow />
        <PopoverBody p={0} w="full" h="full">
          <Picker<AnyModelConfig>
            handleRef={pickerRef}
            options={grouped}
            getId={getId}
            onSelect={onSelect}
            selectedItem={modelConfig}
            getIsDisabled={getIsDisabled}
            isMatch={isMatch}
            ItemComponent={PickerItemComponent}
            GroupHeaderComponent={PickerGroupHeaderComponent}
            noOptionsFallback={<Text>{t('common.noOptions')}</Text>}
            noMatchesFallback={<Text>{t('common.noMatches')}</Text>}
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
MainModelPicker.displayName = 'MainModelPicker';

const PickerGroupHeaderComponent = memo(
  ({ group }: { group: OptionGroup<AnyModelConfig, { name: string; description: string }> }) => {
    return (
      <>
        <Text fontSize="sm" fontWeight="semibold">
          {group.data.name}
        </Text>
        <Text color="base.200" fontSize="xs">
          {group.data.description}
        </Text>
      </>
    );
  }
);
PickerGroupHeaderComponent.displayName = 'PickerGroupHeaderComponent';

export const PickerItemComponent = typedMemo(({ item }: { item: AnyModelConfig }) => {
  return (
    <Flex tabIndex={-1} gap={2}>
      <ModelImage image_url={item.cover_image} />
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
            {item.name}
          </Text>
          <Spacer />
          <Text variant="subtext" fontStyle="italic" noOfLines={1} flexShrink={0} overflow="visible">
            {filesize(item.file_size)}
          </Text>
          <ModelBaseBadge base={item.base} />
        </Flex>
        {item.description && <Text color="base.200">{item.description}</Text>}
      </Flex>
    </Flex>
  );
});
PickerItemComponent.displayName = 'PickerItemComponent';
