import type { BoxProps, FormLabelProps, InputProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  Collapse,
  Expander,
  Flex,
  FormControlGroup,
  FormLabel,
  Icon,
  IconButton,
  Input,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Spacer,
  StandaloneAccordion,
  Text,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { Group, ImperativeModelPickerHandle } from 'common/components/Picker/Picker';
import { getRegex, Picker } from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
import { useStateImperative } from 'common/hooks/useStateImperative';
import { fixedForwardRef } from 'common/util/fixedForwardRef';
import { typedMemo } from 'common/util/typedMemo';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectIsCogView4, selectIsFLUX, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import { LoRAList } from 'features/lora/components/LoRAList';
import LoRASelect from 'features/lora/components/LoRASelect';
import { BASE_COLOR_MAP } from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import ParamCFGScale from 'features/parameters/components/Core/ParamCFGScale';
import ParamGuidance from 'features/parameters/components/Core/ParamGuidance';
import ParamScheduler from 'features/parameters/components/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Core/ParamSteps';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import ParamUpscaleCFGScale from 'features/parameters/components/Upscale/ParamUpscaleCFGScale';
import ParamUpscaleScheduler from 'features/parameters/components/Upscale/ParamUpscaleScheduler';
import { modelSelected } from 'features/parameters/store/actions';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { selectActiveTab, selectCompactModelPicker } from 'features/ui/store/uiSelectors';
import { compactModelPickerToggled } from 'features/ui/store/uiSlice';
import { filesize } from 'filesize';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsInLineVerticalBold, PiArrowsOutLineVerticalBold, PiCaretDownBold } from 'react-icons/pi';
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

const getOptionId = (modelConfig: AnyModelConfig) => modelConfig.key;

type GroupData = {
  base: BaseModelType;
  description: string;
};

const MainModelPicker = memo(() => {
  const { t } = useTranslation();
  const [modelConfigs] = useMainModels();
  const grouped = useMemo<Group<AnyModelConfig, GroupData>[]>(() => {
    const groups: {
      [base in BaseModelType]?: Group<AnyModelConfig, GroupData>;
    } = {};

    for (const modelConfig of modelConfigs) {
      let group = groups[modelConfig.base];
      if (!group) {
        group = {
          id: modelConfig.base,
          data: { base: modelConfig.base, description: `A brief description of ${modelConfig.base} models.` },
          options: [],
        };
        groups[modelConfig.base] = group;
      }

      group.options.push(modelConfig);
    }

    const sortedGroups: Group<AnyModelConfig, GroupData>[] = [];

    if (groups['flux']) {
      sortedGroups.push(groups['flux']);
      delete groups['flux'];
    }
    if (groups['cogview4']) {
      sortedGroups.push(groups['cogview4']);
      delete groups['cogview4'];
    }
    if (groups['sd-1']) {
      sortedGroups.push(groups['sd-1']);
      delete groups['sd-1'];
    }
    if (groups['sd-2']) {
      sortedGroups.push(groups['sd-2']);
      delete groups['sd-2'];
    }
    if (groups['sd-3']) {
      sortedGroups.push(groups['sd-3']);
      delete groups['sd-3'];
    }
    sortedGroups.push(...Object.values(groups));

    return sortedGroups;
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
      <Portal appendToParentPortal={false}>
        <PopoverContent p={0} w={448} h={512}>
          <PopoverArrow />
          <PopoverBody p={0} w="full" h="full">
            <Picker<AnyModelConfig, GroupData>
              handleRef={pickerRef}
              options={grouped}
              getOptionId={getOptionId}
              onSelect={onSelect}
              selectedItem={modelConfig}
              // getIsDisabled={getIsDisabled}
              isMatch={isMatch}
              OptionComponent={PickerOptionComponent}
              GroupComponent={PickerGroupComponent}
              SearchBarComponent={SearchBarComponent}
            />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});
MainModelPicker.displayName = 'MainModelPicker';

const SearchBarComponent = typedMemo(
  fixedForwardRef<HTMLInputElement, InputProps>((props, ref) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();
    const compactModelPicker = useAppSelector(selectCompactModelPicker);

    const onToggleCompact = useCallback(() => {
      dispatch(compactModelPickerToggled());
    }, [dispatch]);
    return (
      <Flex flexDir="column" w="full">
        <Flex gap={2} alignItems="center">
          <Input ref={ref} {...props} placeholder={t('modelManager.filterModels')} />
          <NavigateToModelManagerButton />
          <IconButton
            aria-label="Toggle compact view"
            size="sm"
            variant="ghost"
            icon={compactModelPicker ? <PiArrowsOutLineVerticalBold /> : <PiArrowsInLineVerticalBold />}
            onClick={onToggleCompact}
          />
        </Flex>
        <Flex gap={2} alignItems="center"></Flex>
      </Flex>
    );
  })
);
SearchBarComponent.displayName = 'SearchBarComponent';

const toggleButtonSx = {
  "&[data-expanded='true']": {
    transform: 'rotate(180deg)',
  },
} satisfies SystemStyleObject;

const PickerGroupComponent = memo(
  ({
    group,
    activeOptionId,
    children,
  }: PropsWithChildren<{ group: Group<AnyModelConfig, GroupData>; activeOptionId: string | undefined }>) => {
    const [isOpen, setIsOpen, getIsOpen] = useStateImperative(true);
    useEffect(() => {
      if (group.options.some((option) => option.key === activeOptionId) && !getIsOpen()) {
        setIsOpen(true);
      }
    }, [activeOptionId, getIsOpen, group.options, setIsOpen]);
    const toggle = useCallback(() => {
      setIsOpen((prev) => !prev);
    }, [setIsOpen]);

    return (
      <Flex
        flexDir="column"
        w="full"
        borderLeftColor={`${BASE_COLOR_MAP[group.data.base]}.300`}
        borderLeftWidth={4}
        ps={2}
      >
        <GroupHeader group={group} isOpen={isOpen} toggle={toggle} />
        <Collapse in={isOpen} animateOpacity>
          <Flex flexDir="column" gap={1} w="full" py={1}>
            {children}
          </Flex>
        </Collapse>
      </Flex>
    );
  }
);
PickerGroupComponent.displayName = 'PickerGroupComponent';

const groupSx = {
  alignItems: 'center',
  ps: 2,
  pe: 4,
  py: 1,
  userSelect: 'none',
  position: 'sticky',
  top: 0,
  bg: 'base.800',
  minH: 8,
  borderRadius: 'base',
  _hover: { bg: 'base.750' },
} satisfies SystemStyleObject;

const GroupHeader = memo(
  ({
    group,
    isOpen,
    toggle,
    ...rest
  }: { group: Group<AnyModelConfig, GroupData>; isOpen: boolean; toggle: () => void } & BoxProps) => {
    const { t } = useTranslation();
    const compactModelPicker = useAppSelector(selectCompactModelPicker);

    return (
      <Flex {...rest} role="button" sx={groupSx} onClick={toggle}>
        <Flex flexDir="column" flex={1}>
          <Flex gap={2} alignItems="center">
            <Text fontSize="sm" fontWeight="semibold" color={`${BASE_COLOR_MAP[group.data.base]}.300`}>
              {MODEL_TYPE_SHORT_MAP[group.data.base]}
            </Text>
            <Text fontSize="sm" color="base.300" noOfLines={1}>
              {t('common.model_withCount', { count: group.options.length })}
            </Text>
          </Flex>
          {!compactModelPicker && (
            <Text color="base.200" fontStyle="italic">
              {group.data.description}
            </Text>
          )}
          <Spacer />
        </Flex>
        <Icon color="base.300" as={PiCaretDownBold} sx={toggleButtonSx} data-expanded={isOpen} boxSize={4} />
      </Flex>
    );
  }
);
GroupHeader.displayName = 'GroupHeader';

const optionSx: SystemStyleObject = {
  p: 2,
  gap: 2,
  cursor: 'pointer',
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
    '&[data-active="true"]': {
      bg: 'base.650',
    },
  },
  '&[data-active="true"]': {
    bg: 'base.750',
  },
  '&[data-disabled="true"]': {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
  '&[data-is-compact="true"]': {
    py: 1,
  },
  scrollMarginTop: '42px', // magic number, this is the height of the header
};

export const PickerOptionComponent = typedMemo(({ option, ...rest }: { option: AnyModelConfig } & BoxProps) => {
  const compactModelPicker = useAppSelector(selectCompactModelPicker);

  return (
    <Flex {...rest} sx={optionSx} data-is-compact={compactModelPicker}>
      {!compactModelPicker && <ModelImage image_url={option.cover_image} />}
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
            {option.name}
          </Text>
          <Spacer />
          <Text variant="subtext" fontStyle="italic" noOfLines={1} flexShrink={0} overflow="visible">
            {filesize(option.file_size)}
          </Text>
        </Flex>
        {option.description && !compactModelPicker && <Text color="base.200">{option.description}</Text>}
      </Flex>
    </Flex>
  );
});
PickerOptionComponent.displayName = 'PickerItemComponent';

const BASE_KEYWORDS: { [key in BaseModelType]?: string[] } = {
  'sd-1': ['sd1', 'sd1.4', 'sd1.5', 'sd-1'],
  'sd-2': ['sd2', 'sd2.0', 'sd2.1', 'sd-2'],
  'sd-3': ['sd3', 'sd3.0', 'sd3.5', 'sd-3'],
};

const isMatch = (model: AnyModelConfig, searchTerm: string) => {
  const regex = getRegex(searchTerm);
  const bases = BASE_KEYWORDS[model.base] ?? [model.base];
  const testString =
    `${model.name} ${bases.join(' ')} ${model.type} ${model.description ?? ''} ${model.format}`.toLowerCase();

  if (testString.includes(searchTerm) || regex.test(testString)) {
    return true;
  }

  return false;
};
