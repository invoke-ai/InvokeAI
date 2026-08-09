/* eslint-disable react/react-compiler, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { ModelConfig, ModelTaxonomyType } from '@features/models/core/types';
import type { PickerGroup, PickerOptionState } from '@platform/ui/Picker';

import { Badge, Box, HStack, Icon, Image, Popover, Portal, Spacer, Stack, Text } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel, getModelBaseLongLabel } from '@features/models/core/baseIdentity';
import { getModelPickerGroups } from '@features/models/core/library';
import { formatBytes, getModelTypeLabel, getModelTypePluralLabel } from '@features/models/core/taxonomy';
import { getModelImageUrl } from '@features/models/data/api';
import { ensureModelsLoaded, useModelsSelector } from '@features/models/data/modelsStore';
import { ensureRelatedModelKeysLoaded, useRelatedModelKeys } from '@features/models/data/relationshipsStore';
import { useModelsUi } from '@features/models/ui/ModelsUiContext';
import { setPickerCompactView, useModelsUiSelector } from '@features/models/ui/uiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton, IconButton, Tooltip } from '@platform/ui';
import { Picker } from '@platform/ui/Picker';
import { Link } from '@tanstack/react-router';
import { dropdownContent } from '@theme/recipes';
import {
  BoxIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  LinkIcon,
  RotateCcwIcon,
  XIcon,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const EMPTY_BASES: ReadonlySet<string> = new Set();
const EMPTY_KEYS: ReadonlySet<string> = new Set();

const getOptionId = (model: ModelConfig): string => model.key;

export const ModelSelect = ({
  className,
  disabled,
  excludeKeys,
  filter,
  id,
  invalid,
  isClearable = false,
  modelTypes,
  onChange,
  placeholder,
  showManagerButton = true,
  size = 'sm',
  value,
}: {
  className?: string;
  disabled?: boolean;
  excludeKeys?: ReadonlySet<string>;
  filter?: (model: ModelConfig) => boolean;
  id?: string;
  invalid?: boolean;
  isClearable?: boolean;
  modelTypes: readonly ModelTaxonomyType[];
  onChange: (model: ModelConfig | null) => void;
  placeholder?: string;
  showManagerButton?: boolean;
  size?: 'xs' | 'sm' | 'md';
  value: string | null;
}) => {
  const { t } = useTranslation();
  const { enableModelDescriptions } = useModelsUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const loadError = useModelsSelector((snapshot) => snapshot.error);
  const loadStatus = useModelsSelector((snapshot) => snapshot.status);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedBases, setSelectedBases] = useState<ReadonlySet<string>>(EMPTY_BASES);
  const [lastDisabled, setLastDisabled] = useState(disabled);

  const pickerId = id ?? `models:${modelTypes.join('+')}`;
  const isCompact = useModelsUiSelector((snapshot) => snapshot.pickerCompactViews[pickerId] ?? false);
  const relatedKeyList = useRelatedModelKeys(isOpen ? value : null);
  const relatedKeys = useMemo<ReadonlySet<string>>(
    () => (relatedKeyList ? new Set(relatedKeyList) : EMPTY_KEYS),
    [relatedKeyList]
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  useEffect(() => {
    if (isOpen && value) {
      // A missing relationship list is not worth surfacing; the picker just
      // loses its pinning.
      ensureRelatedModelKeysLoaded(value).catch(() => {});
    }
  }, [isOpen, value]);

  if (disabled !== lastDisabled) {
    setLastDisabled(disabled);

    if (disabled) {
      setIsOpen(false);
      setSelectedBases(EMPTY_BASES);
    }
  }

  const { availableBases, candidates, groups } = useMemo(
    () =>
      isOpen
        ? getModelPickerGroups(models, {
            baseFilter: selectedBases,
            excludeKeys,
            filter,
            modelTypes,
            relatedKeys,
            searchTerm: '',
          })
        : { availableBases: [], candidates: [], groups: [] },
    [excludeKeys, filter, isOpen, modelTypes, models, relatedKeys, selectedBases]
  );

  const selectedModel = useMemo(() => models.find((model) => model.key === value) ?? null, [models, value]);
  const hasMixedTypes = useMemo(() => new Set(candidates.map((model) => model.type)).size > 1, [candidates]);
  const scopeLabel =
    modelTypes.length === 1 ? getModelTypePluralLabel(modelTypes[0] ?? 'main').toLowerCase() : t('models.scopeModels');

  const pickerGroups = useMemo<PickerGroup<ModelConfig>[]>(
    () =>
      groups.map((group) => ({
        colorPalette: getModelBaseColorPalette(group.base),
        getCountLabel: (count) => t('models.modelCount', { count }),
        id: group.key,
        name: getModelBaseLongLabel(group.base),
        options: group.models,
        shortName: getModelBaseLabel(group.base),
      })),
    [groups, t]
  );

  const closeAndReset = () => {
    setIsOpen(false);
    setSelectedBases(EMPTY_BASES);
  };
  const toggleBase = (base: string) => {
    setSelectedBases((prev) => {
      const next = new Set(prev);

      if (next.has(base)) {
        next.delete(base);
      } else {
        next.add(base);
      }

      return next;
    });
  };
  const selectModel = (model: ModelConfig) => {
    onChange(model);
    closeAndReset();
  };
  const canClear = isClearable && Boolean(value);

  const renderOption = useCallback(
    (model: ModelConfig, state: PickerOptionState) => (
      <ModelOptionContent
        enableDescription={enableModelDescriptions}
        isRelated={relatedKeys.has(model.key)}
        model={model}
        showType={hasMixedTypes}
        state={state}
      />
    ),
    [enableModelDescriptions, hasMixedTypes, relatedKeys]
  );

  return (
    <Box className={className} minW="0" w="full">
      <Popover.Root
        ids={id ? { trigger: id } : undefined}
        open={isOpen}
        positioning={{
          fitViewport: true,
          gutter: 4,
          hideWhenDetached: true,
          overflowPadding: 8,
          placement: 'bottom-start',
          sameWidth: true,
          strategy: 'fixed',
        }}
        onOpenChange={(event) => {
          if (disabled) {
            setIsOpen(false);
            return;
          }

          if (event.open) {
            setSelectedBases(EMPTY_BASES);
            setIsOpen(true);
            return;
          }

          closeAndReset();
        }}
      >
        <Box minW="0" position="relative" w="full">
          <Popover.Trigger asChild>
            <Button
              aria-invalid={invalid ? true : undefined}
              aria-haspopup="listbox"
              className={className}
              borderColor={invalid ? undefined : isOpen ? 'accent.solid' : 'border'}
              colorPalette={invalid ? 'red' : 'gray'}
              disabled={disabled}
              justifyContent="space-between"
              minW="0"
              pe={canClear ? '7' : '2'}
              ps="2"
              size={size}
              transitionDuration="fast"
              transitionProperty="border-color, background"
              variant="outline"
              w="full"
              _hover={{
                bg: 'transparent',
                borderColor: invalid ? undefined : isOpen ? 'accent.solid' : 'border.emphasized',
              }}
            >
              {selectedModel ? (
                <ModelButtonContent model={selectedModel} />
              ) : (
                <Text as="span" color="fg.subtle" fontSize="xs" minW="0" truncate>
                  {placeholder ?? t('models.scopeSelect', { scope: scopeLabel })}
                </Text>
              )}
              {canClear ? null : <Icon as={ChevronDownIcon} boxSize="3" flexShrink={0} />}
            </Button>
          </Popover.Trigger>
          {canClear ? (
            <CloseButton
              aria-label={t('models.clearSelectedModel')}
              disabled={disabled}
              insetEnd="1"
              position="absolute"
              size="2xs"
              top="50%"
              transform="translateY(-50%)"
              zIndex="1"
              onClick={(event) => {
                event.stopPropagation();
                onChange(null);
                closeAndReset();
              }}
              onMouseDown={(event) => {
                event.stopPropagation();
              }}
            >
              <XIcon />
            </CloseButton>
          ) : null}
        </Box>
        <Portal>
          <Popover.Positioner>
            <Popover.Content
              css={dropdownContent}
              maxH="min(24rem, var(--available-height))"
              maxW="min(26rem, calc(100vw - 1rem))"
              minW="min(20rem, calc(100vw - 1rem))"
              overflow="hidden"
              p="0"
            >
              <Picker<ModelConfig>
                emptyMessage={t('models.scopeNoCompatibleInstalled', { scope: scopeLabel })}
                getOptionId={getOptionId}
                groups={pickerGroups}
                isCompact={isCompact}
                isMatch={matchesModel}
                listLabel={t('models.scopeAvailable', { scope: scopeLabel })}
                noMatchesMessage={
                  selectedBases.size > 0
                    ? t('models.scopeNoMatchBases', { scope: scopeLabel })
                    : t('models.scopeNoMatchSearch', { scope: scopeLabel })
                }
                renderOption={renderOption}
                searchPlaceholder={t('models.scopeSearch', { scope: scopeLabel })}
                searchSlot={
                  <>
                    <CompactViewToggle isCompact={isCompact} pickerId={pickerId} />
                    {showManagerButton ? <ModelManagerLinkButton /> : null}
                  </>
                }
                selectedId={value}
                statusSlot={
                  loadStatus === 'idle' || loadStatus === 'loading' ? (
                    <Text color="fg.subtle" fontSize="2xs" p="2">
                      {t('models.loadingModels')}
                    </Text>
                  ) : loadStatus === 'error' ? (
                    <Stack alignItems="start" gap="1.5" p="2">
                      <Text color="fg.error" fontSize="2xs">
                        {loadError ?? t('models.failedToLoadModels')}
                      </Text>
                      <Button size="2xs" variant="outline" onClick={() => void ensureModelsLoaded()}>
                        {t('common.retry')}
                      </Button>
                    </Stack>
                  ) : candidates.length === 0 ? (
                    <Text color="fg.subtle" fontSize="2xs" p="2">
                      {t('models.scopeNoCompatibleInstalled', { scope: scopeLabel })}
                    </Text>
                  ) : undefined
                }
                toolbarSlot={
                  availableBases.length >= 2 ? (
                    <HStack alignItems="center" flexWrap="wrap" gap="1">
                      {availableBases.map((base) => (
                        <BaseChip key={base} base={base} isSelected={selectedBases.has(base)} onToggle={toggleBase} />
                      ))}
                      <Spacer />
                      <IconButton
                        aria-label={t('models.resetBaseFilters')}
                        flexShrink={0}
                        opacity={selectedBases.size === 0 ? 0.5 : undefined}
                        pointerEvents={selectedBases.size === 0 ? 'none' : undefined}
                        size="2xs"
                        variant="ghost"
                        onClick={() => setSelectedBases(EMPTY_BASES)}
                      >
                        <Icon as={RotateCcwIcon} boxSize="3" />
                      </IconButton>
                    </HStack>
                  ) : null
                }
                onSelect={selectModel}
              />
            </Popover.Content>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
    </Box>
  );
};

const matchesModel = (model: ModelConfig, searchTerm: string): boolean => {
  const terms = searchTerm.toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = `${model.name} ${model.base} ${model.type} ${model.description ?? ''}`.toLowerCase();

  return terms.every((term) => haystack.includes(term));
};

const CompactViewToggle = ({ isCompact, pickerId }: { isCompact: boolean; pickerId: string }) => {
  const { t } = useTranslation();
  const label = isCompact ? t('models.fullRows') : t('models.compactRows');
  const handleClick = useCallback(() => setPickerCompactView(pickerId, !isCompact), [isCompact, pickerId]);

  return (
    <Tooltip content={label} showArrow>
      <IconButton
        aria-label={label}
        aria-pressed={isCompact}
        flexShrink={0}
        size="xs"
        variant="ghost"
        onClick={handleClick}
      >
        <Icon as={isCompact ? ChevronsUpDownIcon : ChevronsDownUpIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};

const ModelManagerLinkButton = () => {
  const { t } = useTranslation();
  const { managerProjectId } = useModelsUi();
  const search = useMemo(() => ({ project: managerProjectId ?? undefined }), [managerProjectId]);

  return (
    <Tooltip content={t('models.manageModels')} showArrow>
      <IconButton aria-label={t('models.manageModels')} asChild flexShrink={0} size="xs" variant="ghost">
        <Link search={search} to="/models">
          <BoxIcon />
        </Link>
      </IconButton>
    </Tooltip>
  );
};

const BaseChip = ({
  base,
  isSelected,
  onToggle,
}: {
  base: string;
  isSelected: boolean;
  onToggle: (base: string) => void;
}) => (
  <Badge
    aria-pressed={isSelected}
    colorPalette={getModelBaseColorPalette(base)}
    cursor="pointer"
    fontSize="2xs"
    role="button"
    size="sm"
    tabIndex={0}
    userSelect="none"
    variant={isSelected ? 'solid' : 'surface'}
    onClick={(event) => {
      event.stopPropagation();
      onToggle(base);
    }}
    onKeyDown={(event) => {
      event.stopPropagation();

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onToggle(base);
      }
    }}
  >
    {getModelBaseLabel(base)}
  </Badge>
);

const ModelButtonContent = ({ model }: { model: ModelConfig }) => (
  <HStack as="span" flex="1" gap="2" minW="0">
    <Text as="span" fontSize="xs" minW="0" title={model.name} truncate>
      {model.name}
    </Text>
    <Badge
      colorPalette={getModelBaseColorPalette(model.base)}
      flexShrink={0}
      fontSize="2xs"
      size="sm"
      variant="surface"
    >
      {getModelBaseLabel(model.base)}
    </Badge>
  </HStack>
);

const ModelOptionContent = ({
  enableDescription,
  isRelated,
  model,
  showType,
  state,
}: {
  enableDescription: boolean;
  isRelated: boolean;
  model: ModelConfig;
  showType: boolean;
  state: PickerOptionState;
}) => {
  const { t } = useTranslation();
  const showDetail = !state.isCompact;

  return (
    <HStack gap="2" minW="0" textAlign="left" w="full">
      <Icon
        as={CheckIcon}
        aria-hidden
        boxSize="3"
        color="accent.solid"
        flexShrink={0}
        opacity={state.isSelected ? 1 : 0}
      />
      {showDetail && model.cover_image ? (
        <Image alt="" boxSize="8" flexShrink={0} objectFit="cover" rounded="sm" src={getModelImageUrl(model.key)} />
      ) : null}
      <Stack flex="1" gap="0" minW="0">
        <HStack gap="1.5" minW="0">
          {isRelated ? (
            <Tooltip content={t('models.linkedToCurrentSelection')}>
              <Icon
                as={LinkIcon}
                aria-label={t('models.linkedModel')}
                boxSize="3"
                color="accent.solid"
                flexShrink={0}
              />
            </Tooltip>
          ) : null}
          <Text fontSize="xs" minW="0" truncate>
            {model.name}
          </Text>
          {showType ? (
            <Badge colorPalette="gray" flexShrink={0} fontSize="2xs" size="xs" variant="surface">
              {getModelTypeLabel(model.type)}
            </Badge>
          ) : null}
        </HStack>
        {showDetail && enableDescription && model.description ? (
          <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
            {model.description}
          </Text>
        ) : null}
      </Stack>
      <Text color="fg.subtle" flexShrink={0} fontSize="2xs" fontStyle="italic">
        {formatBytes(model.file_size)}
      </Text>
    </HStack>
  );
};
