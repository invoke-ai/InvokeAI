import type { BoxProps, ButtonProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Badge,
  Button,
  Flex,
  Icon,
  Link,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import type { Group, PickerContextState } from 'common/components/Picker/Picker';
import { buildGroup, getRegex, isGroup, Picker, usePickerContext } from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
import { typedMemo } from 'common/util/typedMemo';
import { uniq } from 'es-toolkit/compat';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { MODEL_BASE_TO_COLOR, MODEL_BASE_TO_LONG_NAME, MODEL_BASE_TO_SHORT_NAME } from 'features/modelManagerV2/models';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import type { BaseModelType } from 'features/nodes/types/common';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { parseCategoryFromName } from 'features/parameters/components/modelPickerCategory';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { filesize } from 'filesize';
import { memo, useCallback, useMemo, useRef } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiLinkSimple } from 'react-icons/pi';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';
import {
  type AnyModelConfigWithExternal,
  type ExternalApiModelConfig,
  isExternalApiModelConfig,
} from 'services/api/types';

const selectSelectedModelKeys = createMemoizedSelector(selectParamsSlice, selectLoRAsSlice, (params, loras) => {
  const keys: string[] = [];
  const main = params.model;
  const vae = params.vae;
  const refiner = params.refinerModel;
  const controlnet = params.controlLora;

  if (main) {
    keys.push(main.key);
  }
  if (vae) {
    keys.push(vae.key);
  }
  if (refiner) {
    keys.push(refiner.key);
  }
  if (controlnet) {
    keys.push(controlnet.key);
  }
  for (const { model } of loras.loras) {
    keys.push(model.key);
  }

  return uniq(keys);
});

type WithStarred<T> = T & { starred?: boolean; category?: string | null };

// Type for models with starred field
const getOptionId = <T extends AnyModelConfigWithExternal>(modelConfig: WithStarred<T>) => modelConfig.key;

const getOptionSubgroupFromCategory = <T extends AnyModelConfigWithExternal>(option: WithStarred<T>) =>
  option.category ?? null;

const ModelManagerLink = memo((props: ButtonProps) => {
  const onClick = useCallback(() => {
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('launchpad');
  }, []);

  return <Button size="sm" flexGrow={0} variant="link" color="base.200" onClick={onClick} {...props} />;
});
ModelManagerLink.displayName = 'ModelManagerLink';

const components = {
  LinkComponent: <ModelManagerLink />,
};

const NoOptionsFallback = memo(({ noOptionsText }: { noOptionsText?: string }) => {
  const { t } = useTranslation();
  const { data: setupStatus } = useGetSetupStatusQuery();
  const user = useAppSelector(selectCurrentUser);

  const isMultiuser = setupStatus?.multiuser_enabled ?? false;
  const isAdmin = !isMultiuser || (user?.is_admin ?? false);
  const adminEmail = setupStatus?.admin_email ?? null;

  if (!isAdmin) {
    const AdminEmailLink = adminEmail ? (
      <Link href={`mailto:${adminEmail}`} color="base.200">
        {adminEmail}
      </Link>
    ) : (
      <Text as="span" color="base.200">
        your administrator
      </Text>
    );

    return (
      <Flex flexDir="column" gap={4} alignItems="center">
        <Text color="base.200" textAlign="center">
          <Trans i18nKey="modelManager.modelPickerFallbackNoModelsInstalledNonAdmin" components={{ AdminEmailLink }} />
        </Text>
      </Flex>
    );
  }

  return (
    <Flex flexDir="column" gap={4} alignItems="center">
      <Text color="base.200">{noOptionsText ?? t('modelManager.modelPickerFallbackNoModelsInstalled')}</Text>
      <Text color="base.200">
        <Trans i18nKey="modelManager.modelPickerFallbackNoModelsInstalled2" components={components} />
      </Text>
    </Flex>
  );
});
NoOptionsFallback.displayName = 'NoOptionsFallback';

const getGroupIDFromModelConfig = (modelConfig: AnyModelConfigWithExternal): string => modelConfig.base;

const getGroupNameFromModelConfig = (modelConfig: AnyModelConfigWithExternal): string => {
  return MODEL_BASE_TO_LONG_NAME[modelConfig.base];
};

const getGroupShortNameFromModelConfig = (modelConfig: AnyModelConfigWithExternal): string => {
  return MODEL_BASE_TO_SHORT_NAME[modelConfig.base];
};

const getGroupColorSchemeFromModelConfig = (modelConfig: AnyModelConfigWithExternal): string => {
  return MODEL_BASE_TO_COLOR[modelConfig.base];
};

const relatedModelKeysQueryOptions = {
  selectFromResult: ({ data }) => {
    if (!data) {
      return { relatedModelKeys: EMPTY_ARRAY };
    }
    return { relatedModelKeys: data };
  },
} satisfies Parameters<typeof useGetRelatedModelIdsBatchQuery>[1];

const popperModifiers = [
  {
    // Prevents the popover from "touching" the edges of the screen
    name: 'preventOverflow',
    options: { padding: 16 },
  },
];

const removeStarred = <T,>(obj: WithStarred<T>): T => {
  const { starred: _starred, category: _category, ...rest } = obj;
  return rest as T;
};

export const ModelPicker = typedMemo(
  <T extends AnyModelConfigWithExternal = AnyModelConfigWithExternal>({
    pickerId,
    modelConfigs,
    selectedModelConfig,
    onChange,
    grouped,
    getIsOptionDisabled,
    placeholder,
    allowEmpty,
    isDisabled,
    isInvalid,
    className,
    noOptionsText,
    initialGroupStates,
  }: {
    pickerId: string;
    modelConfigs: T[];
    selectedModelConfig: T | undefined;
    onChange: (modelConfig: T) => void;
    grouped?: boolean;
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
    const selectedKeys = useAppSelector(selectSelectedModelKeys);

    const { relatedModelKeys } = useGetRelatedModelIdsBatchQuery(selectedKeys, {
      ...relatedModelKeysQueryOptions,
    });

    const options = useMemo<WithStarred<T>[] | Group<WithStarred<T>>[]>(() => {
      // Enrich each model with starred + parsed category fields. The category is parsed from a "[category]name"
      // prefix in the model's display name. Uncategorized models have category=null.
      const enriched: WithStarred<T>[] = modelConfigs.map((model) => ({
        ...model,
        starred: relatedModelKeys.includes(model.key),
        category: parseCategoryFromName(model.name).category,
      }));

      // Sort by (starred desc, name asc) within a slice that shares a category.
      const sortByStarredAndName = (a: WithStarred<T>, b: WithStarred<T>) => {
        if (a.starred && !b.starred) {
          return -1;
        }
        if (!a.starred && b.starred) {
          return 1;
        }
        return a.name.localeCompare(b.name);
      };

      if (!grouped) {
        // Flat-list picker (e.g. LoRA selection, where compatible models are already filtered to one base).
        // When at least one model carries a category, promote categories to top-level groups so the user can
        // toggle / filter by category. Uncategorized models go into a final "Uncategorized" group.
        const hasAnyCategory = enriched.some((m) => Boolean(m.category));
        if (!hasAnyCategory) {
          // Preserve previous behavior: a flat list sorted with starred models first.
          return enriched.sort((a, b) => {
            if (a.starred && !b.starred) {
              return -1;
            }
            if (!a.starred && b.starred) {
              return 1;
            }
            return 0;
          });
        }

        const categoryGroups: Record<string, Group<WithStarred<T>>> = {};
        const uncategorized: WithStarred<T>[] = [];

        for (const model of enriched) {
          if (model.category) {
            let group = categoryGroups[model.category];
            if (!group) {
              group = buildGroup<WithStarred<T>>({
                id: `__category__${model.category}`,
                name: model.category,
                shortName: model.category,
                color: 'base.300',
                options: [],
                getOptionCountString: (count) => t('common.model_withCount', { count }),
              });
              categoryGroups[model.category] = group;
            }
            group.options.push(model);
          } else {
            uncategorized.push(model);
          }
        }

        const result: Group<WithStarred<T>>[] = [];
        const sortedCategoryNames = Object.keys(categoryGroups).sort((a, b) => a.localeCompare(b));
        for (const catName of sortedCategoryNames) {
          const group = categoryGroups[catName];
          if (group) {
            group.options.sort(sortByStarredAndName);
            result.push(group);
          }
        }
        if (uncategorized.length > 0) {
          uncategorized.sort(sortByStarredAndName);
          result.push(
            buildGroup<WithStarred<T>>({
              id: '__uncategorized__',
              name: t('modelManager.uncategorized'),
              shortName: t('modelManager.uncategorized'),
              color: 'base.500',
              options: uncategorized,
              getOptionCountString: (count) => t('common.model_withCount', { count }),
            })
          );
        }
        return result;
      }

      // Grouped picker (main model selection): top-level groups by base (FLUX, SDXL, ...). Within each base group
      // we further organize options by their parsed category as inline subgroup headers. Categorized models appear
      // first (alphabetically by category); uncategorized models appear after, without a subgroup header.
      const groups: Record<string, Group<WithStarred<T>>> = {};

      for (const model of enriched) {
        const groupId = getGroupIDFromModelConfig(model);
        let group = groups[groupId];
        if (!group) {
          group = buildGroup<WithStarred<T>>({
            id: model.base,
            color: `${getGroupColorSchemeFromModelConfig(model)}.300`,
            shortName: getGroupShortNameFromModelConfig(model),
            name: getGroupNameFromModelConfig(model),
            getOptionCountString: (count) => t('common.model_withCount', { count }),
            options: [],
            getOptionSubgroup: getOptionSubgroupFromCategory,
          });
          groups[groupId] = group;
        }
        group.options.push(model);
      }

      const sortWithinBaseGroup = (a: WithStarred<T>, b: WithStarred<T>) => {
        const catA = a.category ?? null;
        const catB = b.category ?? null;
        // Categorized first; uncategorized at the end.
        if (catA && !catB) {
          return -1;
        }
        if (!catA && catB) {
          return 1;
        }
        if (catA && catB && catA !== catB) {
          return catA.localeCompare(catB);
        }
        return sortByStarredAndName(a, b);
      };

      const _options: Group<WithStarred<T>>[] = [];

      // Add groups in the original order
      for (const groupId of ['api', 'flux', 'z-image', 'qwen-image', 'cogview4', 'sdxl', 'sd-3', 'sd-2', 'sd-1']) {
        const group = groups[groupId];
        if (group) {
          group.options.sort(sortWithinBaseGroup);
          _options.push(group);
          delete groups[groupId];
        }
      }
      for (const group of Object.values(groups)) {
        group.options.sort(sortWithinBaseGroup);
        _options.push(group);
      }

      return _options;
    }, [grouped, modelConfigs, relatedModelKeys, t]);
    const popover = useDisclosure(false);
    const pickerRef = useRef<PickerContextState<WithStarred<T>>>(null);

    const selectedOption = useMemo<WithStarred<T> | undefined>(() => {
      if (!selectedModelConfig) {
        return undefined;
      }
      let _selectedOption: WithStarred<T> | undefined = undefined;

      for (const optionOrGroup of options) {
        if (isGroup(optionOrGroup)) {
          const result = optionOrGroup.options.find((o) => o.key === selectedModelConfig.key);
          if (result) {
            _selectedOption = result;
            break;
          }
        } else if (optionOrGroup.key === selectedModelConfig.key) {
          _selectedOption = optionOrGroup;
          break;
        }
      }

      return _selectedOption;
    }, [options, selectedModelConfig]);

    const onClose = useCallback(() => {
      popover.close();
      pickerRef.current?.$searchTerm.set('');
    }, [popover]);

    const onSelect = useCallback(
      (model: WithStarred<T>) => {
        onClose();
        // Remove the starred field before passing to onChange
        onChange(removeStarred(model));
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
            {selectedModelConfig
              ? parseCategoryFromName(selectedModelConfig.name).displayName
              : (placeholder ?? 'Select Model')}
            <Spacer />
            <PiCaretDownBold />
          </Button>
        </PopoverTrigger>
        <Portal appendToParentPortal={false}>
          <PopoverContent p={0} w={400} h={400}>
            <PopoverArrow />
            <PopoverBody p={0} w="full" h="full" borderWidth={1} borderColor="base.700" borderRadius="base">
              <Picker<WithStarred<T>>
                pickerId={pickerId}
                handleRef={pickerRef}
                optionsOrGroups={options}
                getOptionId={getOptionId<T>}
                onSelect={onSelect}
                selectedOption={selectedOption}
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
    bg: 'invokeBlue.300',
    color: 'base.900',
    '.extra-info': {
      color: 'base.700',
    },
    '.picker-option': {
      fontWeight: 'bold',
      '&[data-is-compact="true"]': {
        fontWeight: 'semibold',
      },
    },
    '&[data-active="true"]': {
      bg: 'invokeBlue.250',
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
  <T extends AnyModelConfigWithExternal>({ option, ...rest }: { option: WithStarred<T> } & BoxProps) => {
    const { isCompactView } = usePickerContext<WithStarred<T>>();
    const externalOption = isExternalApiModelConfig(option) ? (option as ExternalApiModelConfig) : null;
    const providerLabel = externalOption ? externalOption.provider_id.toUpperCase() : null;
    // Hide the "[category]" prefix in the picker list; the category is shown as a subgroup header instead. The
    // underlying model name (and the name shown in the model manager) is unchanged.
    const displayName = useMemo(() => parseCategoryFromName(option.name).displayName, [option.name]);

    return (
      <Flex {...rest} sx={optionSx} data-is-compact={isCompactView}>
        {!isCompactView && option.cover_image && <ModelImage image_url={option.cover_image} />}
        <Flex flexDir="column" gap={1} flex={1}>
          <Flex gap={2} alignItems="center">
            {option.starred && <Icon as={PiLinkSimple} color="invokeYellow.500" boxSize={4} />}
            <Text className="picker-option" sx={optionNameSx} data-is-compact={isCompactView}>
              {displayName}
            </Text>
            {!isCompactView && externalOption && (
              <Badge
                colorScheme={MODEL_BASE_TO_COLOR[externalOption.base as BaseModelType]}
                variant="subtle"
                flexShrink={0}
              >
                {providerLabel}
              </Badge>
            )}
            <Spacer />
            {option.file_size > 0 && (
              <Text
                className="extra-info"
                variant="subtext"
                fontStyle="italic"
                noOfLines={1}
                flexShrink={0}
                overflow="visible"
              >
                {filesize(option.file_size)}
              </Text>
            )}
          </Flex>
          {option.description && !isCompactView && (
            <Text className="extra-info" color="base.200">
              {option.description}
            </Text>
          )}
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

const isMatch = <T extends AnyModelConfigWithExternal>(model: WithStarred<T>, searchTerm: string) => {
  const regex = getRegex(searchTerm);
  const bases = BASE_KEYWORDS[model.base] ?? [model.base];
  const externalModel = isExternalApiModelConfig(model) ? (model as ExternalApiModelConfig) : null;
  const externalSearch = externalModel ? ` ${externalModel.provider_id} ${externalModel.provider_model_id}` : '';
  const categorySearch = model.category ? ` ${model.category}` : '';
  const testString =
    `${model.name} ${bases.join(' ')} ${model.type} ${model.description ?? ''} ${model.format}${externalSearch}${categorySearch}`.toLowerCase();

  if (testString.includes(searchTerm) || regex.test(testString)) {
    return true;
  }

  return false;
};
