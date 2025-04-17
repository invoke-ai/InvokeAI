import type { BoxProps, FormLabelProps, InputProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Badge,
  Box,
  Button,
  Expander,
  Flex,
  FormControlGroup,
  FormLabel,
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
import {
  DefaultNoMatchesFallback,
  DefaultNoOptionsFallback,
  getRegex,
  Picker,
  usePickerContext,
} from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
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
import { isEqual } from 'lodash-es';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
};

type BaseModelTypeFilters = { [key in BaseModelType]?: boolean };

type PickerExtraContext = {
  toggleBaseModelTypeFilter: (baseModelType: BaseModelType) => void;
  basesWithModels: BaseModelType[];
  baseModelTypeFilters: BaseModelTypeFilters;
};

const MainModelPicker = memo(() => {
  const { t } = useTranslation();
  const [modelConfigs] = useMainModels();
  const basesWithModels = useMemo(() => {
    const bases: BaseModelType[] = [];
    for (const modelConfig of modelConfigs) {
      if (!bases.includes(modelConfig.base)) {
        bases.push(modelConfig.base);
      }
    }
    return bases;
  }, [modelConfigs]);
  const [baseModelTypeFilters, setBaseModelTypeFilters] = useState<BaseModelTypeFilters>({});
  useEffect(() => {
    const newFilters: BaseModelTypeFilters = {};
    if (isEqual(Object.keys(baseModelTypeFilters), basesWithModels)) {
      return;
    }
    for (const base of basesWithModels) {
      if (newFilters[base] === undefined) {
        newFilters[base] = true;
      } else {
        newFilters[base] = baseModelTypeFilters[base];
      }
    }
    setBaseModelTypeFilters(newFilters);
  }, [baseModelTypeFilters, basesWithModels]);
  const toggleBaseModelTypeFilter = useCallback(
    (baseModelType: BaseModelType) => {
      setBaseModelTypeFilters((prev) => {
        const newFilters: BaseModelTypeFilters = {};
        for (const base of basesWithModels) {
          newFilters[base] = baseModelType === base ? !prev[base] : prev[base];
        }
        return newFilters;
      });
    },
    [basesWithModels]
  );
  const ctx = useMemo(
    () => ({ toggleBaseModelTypeFilter, basesWithModels, baseModelTypeFilters }),
    [toggleBaseModelTypeFilter, basesWithModels, baseModelTypeFilters]
  );
  const grouped = useMemo<Group<AnyModelConfig, GroupData>[]>(() => {
    const groups: {
      [base in BaseModelType]?: Group<AnyModelConfig, GroupData>;
    } = {};

    for (const modelConfig of modelConfigs) {
      let group = groups[modelConfig.base];
      if (!group && baseModelTypeFilters[modelConfig.base]) {
        group = {
          id: modelConfig.base,
          data: { base: modelConfig.base },
          options: [],
        };
        groups[modelConfig.base] = group;
      }
      if (group) {
        group.options.push(modelConfig);
      }
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
  }, [baseModelTypeFilters, modelConfigs]);
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
            <Picker<AnyModelConfig, GroupData, PickerExtraContext>
              handleRef={pickerRef}
              options={grouped}
              getOptionId={getOptionId}
              onSelect={onSelect}
              selectedItem={modelConfig}
              isMatch={isMatch}
              OptionComponent={PickerOptionComponent}
              GroupComponent={PickerGroupComponent}
              SearchBarComponent={SearchBarComponent}
              noOptionsFallback={<DefaultNoOptionsFallback label={t('modelManager.noModelsInstalled')} />}
              noMatchesFallback={<DefaultNoMatchesFallback label={t('modelManager.noMatchingModels')} />}
              ctx={ctx}
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
    const { ctx } = usePickerContext<AnyModelConfig, GroupData, PickerExtraContext>();
    const onToggleCompact = useCallback(() => {
      dispatch(compactModelPickerToggled());
    }, [dispatch]);
    return (
      <Flex flexDir="column" w="full" gap={2}>
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
        <Flex gap={2} alignItems="center">
          {ctx.basesWithModels.map((base) => (
            <ModelBaseFilterButton key={base} base={base} />
          ))}
        </Flex>
      </Flex>
    );
  })
);
SearchBarComponent.displayName = 'SearchBarComponent';

const ModelBaseFilterButton = memo(({ base }: { base: BaseModelType }) => {
  const { ctx } = usePickerContext<AnyModelConfig, GroupData, PickerExtraContext>();

  const onClick = useCallback(() => {
    ctx.toggleBaseModelTypeFilter(base);
  }, [base, ctx]);

  return (
    <Badge
      role="button"
      size="xs"
      variant="solid"
      userSelect="none"
      bg={ctx.baseModelTypeFilters[base] ? `${BASE_COLOR_MAP[base]}.300` : 'transparent'}
      color={ctx.baseModelTypeFilters[base] ? undefined : 'base.200'}
      borderColor={`${BASE_COLOR_MAP[base]}.300`}
      borderWidth={1}
      onClick={onClick}
    >
      {MODEL_TYPE_SHORT_MAP[base]}
    </Badge>
  );
});
ModelBaseFilterButton.displayName = 'ModelBaseFilterButton';

const PickerGroupComponent = memo(
  ({ group, children }: PropsWithChildren<{ group: Group<AnyModelConfig, GroupData> }>) => {
    return (
      <Flex
        flexDir="column"
        w="full"
        borderLeftColor={`${BASE_COLOR_MAP[group.data.base]}.300`}
        borderLeftWidth={4}
        ps={2}
      >
        <GroupHeader group={group} />
        <Flex flexDir="column" gap={1} w="full" py={1}>
          {children}
        </Flex>
      </Flex>
    );
  }
);
PickerGroupComponent.displayName = 'PickerGroupComponent';

const groupSx = {
  flexDir: 'column',
  flex: 1,
  ps: 2,
  pe: 4,
  py: 1,
  userSelect: 'none',
  position: 'sticky',
  top: 0,
  bg: 'base.800',
  minH: 8,
} satisfies SystemStyleObject;

const GroupHeader = memo(({ group, ...rest }: { group: Group<AnyModelConfig, GroupData> } & BoxProps) => {
  const { t } = useTranslation();

  return (
    <Flex {...rest} sx={groupSx}>
      <Flex gap={2} alignItems="center">
        <Text fontSize="sm" fontWeight="semibold" color={`${BASE_COLOR_MAP[group.data.base]}.300`}>
          {MODEL_TYPE_SHORT_MAP[group.data.base]}
        </Text>
        <Text fontSize="sm" color="base.300" noOfLines={1}>
          {t('common.model_withCount', { count: group.options.length })}
        </Text>
      </Flex>
    </Flex>
  );
});
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

const optionNameSx: SystemStyleObject = {
  fontSize: 'sm',
  noOfLines: 1,
  fontWeight: 'semibold',
  '&[data-is-compact="true"]': {
    fontWeight: 'normal',
  },
};

export const PickerOptionComponent = typedMemo(({ option, ...rest }: { option: AnyModelConfig } & BoxProps) => {
  const compactModelPicker = useAppSelector(selectCompactModelPicker);

  return (
    <Flex {...rest} sx={optionSx} data-is-compact={compactModelPicker}>
      {!compactModelPicker && <ModelImage image_url={option.cover_image} />}
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text sx={optionNameSx} data-is-compact={compactModelPicker}>
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
