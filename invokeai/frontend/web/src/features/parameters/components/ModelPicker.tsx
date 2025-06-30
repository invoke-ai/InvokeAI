import type { BoxProps, ButtonProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Flex,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $onClickGoToModelManager } from 'app/store/nanostores/onClickGoToModelManager';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { Group, PickerContextState } from 'common/components/Picker/Picker';
import { buildGroup, getRegex, Picker, usePickerContext } from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
import { typedMemo } from 'common/util/typedMemo';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { BASE_COLOR_MAP } from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { API_BASE_MODELS, MODEL_TYPE_MAP, MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { selectIsModelsTabDisabled } from 'features/system/store/configSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { filesize } from 'filesize';
import { memo, useCallback, useMemo, useRef } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiStarFill } from 'react-icons/pi';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';

// Type for models with starred field
type StarredModelConfig = AnyModelConfig & { starred?: boolean };

const getOptionId = <T extends AnyModelConfig>(modelConfig: T & { starred?: boolean }) => modelConfig.key;

const ModelManagerLink = memo((props: ButtonProps) => {
  const onClickGoToModelManager = useStore($onClickGoToModelManager);
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(setActiveTab('models'));
    setInstallModelsTabByName('launchpad');
  }, [dispatch]);

  return (
    <Button
      size="sm"
      flexGrow={0}
      variant="link"
      color="base.200"
      onClick={onClickGoToModelManager ?? onClick}
      {...props}
    />
  );
});
ModelManagerLink.displayName = 'ModelManagerLink';

const components = {
  LinkComponent: <ModelManagerLink />,
};

const NoOptionsFallback = memo(({ noOptionsText }: { noOptionsText?: string }) => {
  const { t } = useTranslation();
  const isModelsTabDisabled = useAppSelector(selectIsModelsTabDisabled);
  const onClickGoToModelManager = useStore($onClickGoToModelManager);

  return (
    <Flex flexDir="column" gap={4} alignItems="center">
      <Text color="base.200">{noOptionsText ?? t('modelManager.modelPickerFallbackNoModelsInstalled')}</Text>
      {(!isModelsTabDisabled || onClickGoToModelManager) && (
        <Text color="base.200">
          <Trans i18nKey="modelManager.modelPickerFallbackNoModelsInstalled2" components={components} />
        </Text>
      )}
    </Flex>
  );
});
NoOptionsFallback.displayName = 'NoOptionsFallback';

const getGroupIDFromModelConfig = (modelConfig: AnyModelConfig): string => {
  if (API_BASE_MODELS.includes(modelConfig.base)) {
    return 'api';
  }
  return modelConfig.base;
};

const getGroupNameFromModelConfig = (modelConfig: AnyModelConfig): string => {
  if (API_BASE_MODELS.includes(modelConfig.base)) {
    return 'External API';
  }
  return MODEL_TYPE_MAP[modelConfig.base];
};

const getGroupShortNameFromModelConfig = (modelConfig: AnyModelConfig): string => {
  if (API_BASE_MODELS.includes(modelConfig.base)) {
    return 'api';
  }
  return MODEL_TYPE_SHORT_MAP[modelConfig.base];
};

const getGroupColorSchemeFromModelConfig = (modelConfig: AnyModelConfig): string => {
  if (API_BASE_MODELS.includes(modelConfig.base)) {
    return 'pink';
  }
  return BASE_COLOR_MAP[modelConfig.base];
};

const popperModifiers = [
  {
    // Prevents the popover from "touching" the edges of the screen
    name: 'preventOverflow',
    options: { padding: 16 },
  },
];

export const ModelPicker = typedMemo(
  <T extends AnyModelConfig = AnyModelConfig>({
    modelConfigs,
    selectedModelConfig,
    onChange,
    grouped,
    relatedModelKeys = [],
    getIsOptionDisabled,
    placeholder,
    allowEmpty,
    isDisabled,
    isInvalid,
    className,
    noOptionsText,
    initialGroupStates,
  }: {
    modelConfigs: T[];
    selectedModelConfig: T | undefined;
    onChange: (modelConfig: T) => void;
    grouped?: boolean;
    relatedModelKeys?: string[];
    getIsOptionDisabled?: (model: T) => boolean;
    placeholder?: string;
    allowEmpty?: boolean;
    isDisabled?: boolean;
    isInvalid?: boolean;
    className?: string;
    noOptionsText?: string;
    initialGroupStates?: Record<string, boolean>;
  }) => {
    const { t } = useTranslation();
    const options = useMemo<(T & { starred?: boolean })[] | Group<T & { starred?: boolean }>[]>(() => {
      if (!grouped) {
        // Add starred field to model options and sort them
        const modelsWithStarred = modelConfigs.map(model => ({
          ...model,
          starred: relatedModelKeys.includes(model.key)
        }));
        
        // Sort so starred models come first
        return modelsWithStarred.sort((a, b) => {
          if (a.starred && !b.starred) return -1;
          if (!a.starred && b.starred) return 1;
          return 0;
        });
      }

      // When all groups are disabled, we show all models
      const groups: Record<string, Group<T & { starred?: boolean }>> = {};

      for (const modelConfig of modelConfigs) {
        const groupId = getGroupIDFromModelConfig(modelConfig);
        let group = groups[groupId];
        if (!group) {
          group = buildGroup<T & { starred?: boolean }>({
            id: modelConfig.base,
            color: `${getGroupColorSchemeFromModelConfig(modelConfig)}.300`,
            shortName: getGroupShortNameFromModelConfig(modelConfig),
            name: getGroupNameFromModelConfig(modelConfig),
            getOptionCountString: (count) => t('common.model_withCount', { count }),
            options: [],
          });
          groups[groupId] = group;
        }
        if (group) {
          // Add starred field to the model
          const modelWithStarred = {
            ...modelConfig,
            starred: relatedModelKeys.includes(modelConfig.key)
          };
          group.options.push(modelWithStarred);
        }
      }

      const _options: Group<T & { starred?: boolean }>[] = [];

      // Add groups in the original order
      for (const groupId of ['api', 'flux', 'cogview4', 'sdxl', 'sd-3', 'sd-2', 'sd-1']) {
        const group = groups[groupId];
        if (group) {
          // Sort options within each group so starred ones come first
          group.options.sort((a, b) => {
            if (a.starred && !b.starred) return -1;
            if (!a.starred && b.starred) return 1;
            return 0;
          });
          _options.push(group);
          delete groups[groupId];
        }
      }
      _options.push(...Object.values(groups));

      return _options;
    }, [grouped, modelConfigs, relatedModelKeys, t]);
    const popover = useDisclosure(false);
    const pickerRef = useRef<PickerContextState<T & { starred?: boolean }>>(null);

    const onClose = useCallback(() => {
      popover.close();
      pickerRef.current?.$searchTerm.set('');
    }, [popover]);

    const onSelect = useCallback(
      (model: T & { starred?: boolean }) => {
        onClose();
        // Remove the starred field before passing to onChange
        const { starred, ...modelWithoutStarred } = model;
        onChange(modelWithoutStarred as T);
      },
      [onChange, onClose]
    );

    const colorScheme = useMemo(() => {
      if (!selectedModelConfig && !allowEmpty) {
        return 'error';
      }
      if (isInvalid) {
        return 'error';
      }
      return undefined;
    }, [allowEmpty, isInvalid, selectedModelConfig]);

    return (
      <Popover
        isOpen={popover.isOpen}
        onOpen={popover.open}
        onClose={onClose}
        initialFocusRef={pickerRef.current?.inputRef}
        modifiers={popperModifiers}
      >
        <PopoverTrigger>
          <Button
            className={className}
            size="sm"
            flexGrow={1}
            variant="outline"
            colorScheme={colorScheme}
            isDisabled={isDisabled}
          >
            {selectedModelConfig?.name ?? placeholder ?? 'Select Model'}
            <Spacer />
            <PiCaretDownBold />
          </Button>
        </PopoverTrigger>
        <Portal appendToParentPortal={false}>
          <PopoverContent p={0} w={400} h={400}>
            <PopoverArrow />
            <PopoverBody p={0} w="full" h="full">
              <Picker<T & { starred?: boolean }>
                handleRef={pickerRef}
                optionsOrGroups={options}
                getOptionId={getOptionId<T>}
                onSelect={onSelect}
                selectedOption={selectedModelConfig ? { ...selectedModelConfig, starred: relatedModelKeys.includes(selectedModelConfig.key) } : undefined}
                isMatch={isMatch<T>}
                OptionComponent={PickerOptionComponent<T>}
                noOptionsFallback={<NoOptionsFallback noOptionsText={noOptionsText} />}
                noMatchesFallback={t('modelManager.noMatchingModels')}
                NextToSearchBar={<NavigateToModelManagerButton />}
                getIsOptionDisabled={getIsOptionDisabled}
                searchable
                initialGroupStates={initialGroupStates}
              />
            </PopoverBody>
          </PopoverContent>
        </Portal>
      </Popover>
    );
  }
);
ModelPicker.displayName = 'ModelPicker';

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
    px: 1,
    py: 0.5,
  },
  scrollMarginTop: '24px', // magic number, this is the height of the header
};

const optionNameSx: SystemStyleObject = {
  fontSize: 'sm',
  noOfLines: 1,
  fontWeight: 'semibold',
  '&[data-is-compact="true"]': {
    fontWeight: 'normal',
  },
};

const PickerOptionComponent = typedMemo(
  <T extends AnyModelConfig>({ option, ...rest }: { option: T & { starred?: boolean } } & BoxProps) => {
    const { $compactView } = usePickerContext<T & { starred?: boolean }>();
    const compactView = useStore($compactView);

    return (
      <Flex {...rest} sx={optionSx} data-is-compact={compactView}>
        {!compactView && option.cover_image && <ModelImage image_url={option.cover_image} />}
        <Flex flexDir="column" gap={1} flex={1}>
          <Flex gap={2} alignItems="center">
            {option.starred && (
              <PiStarFill color="yellow" size={16} />
            )}
            <Text sx={optionNameSx} data-is-compact={compactView}>
              {option.name}
            </Text>
            <Spacer />
            {option.file_size > 0 && (
              <Text variant="subtext" fontStyle="italic" noOfLines={1} flexShrink={0} overflow="visible">
                {filesize(option.file_size)}
              </Text>
            )}
            {option.usage_info && (
              <Text variant="subtext" fontStyle="italic" noOfLines={1} flexShrink={0} overflow="visible">
                {option.usage_info}
              </Text>
            )}
          </Flex>
          {option.description && !compactView && <Text color="base.200">{option.description}</Text>}
        </Flex>
      </Flex>
    );
  }
);
PickerOptionComponent.displayName = 'PickerItemComponent';

const BASE_KEYWORDS: { [key in BaseModelType]?: string[] } = {
  'sd-1': ['sd1', 'sd1.4', 'sd1.5', 'sd-1'],
  'sd-2': ['sd2', 'sd2.0', 'sd2.1', 'sd-2'],
  'sd-3': ['sd3', 'sd3.0', 'sd3.5', 'sd-3'],
};

const isMatch = <T extends AnyModelConfig>(model: T & { starred?: boolean }, searchTerm: string) => {
  const regex = getRegex(searchTerm);
  const bases = BASE_KEYWORDS[model.base] ?? [model.base];
  const testString =
    `${model.name} ${bases.join(' ')} ${model.type} ${model.description ?? ''} ${model.format}`.toLowerCase();

  if (testString.includes(searchTerm) || regex.test(testString)) {
    return true;
  }

  return false;
};
