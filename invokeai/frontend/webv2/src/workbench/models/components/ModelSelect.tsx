/* eslint-disable react/react-compiler, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';

import {
  Badge,
  Box,
  HStack,
  Icon,
  Input,
  InputGroup,
  Popover,
  Portal,
  ScrollArea,
  Spacer,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { Button, CloseButton, IconButton, Tooltip } from '@workbench/components/ui';
import { getModelBaseColorPalette, getModelBaseLabel } from '@workbench/models/baseIdentity';
import { getModelPickerGroups } from '@workbench/models/library';
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import { formatBytes, getModelTypePluralLabel } from '@workbench/models/taxonomy';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { BoxIcon, CheckIcon, ChevronDownIcon, RotateCcwIcon, SearchIcon, XIcon } from 'lucide-react';
import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';

const EMPTY_BASES: ReadonlySet<string> = new Set();

/**
 * Universal single-model picker: a button-triggered searchable list over the
 * installed library. Scope it with `modelTypes` — one type for dedicated
 * pickers (a LoRA picker, a main-model picker) or several for cross-type
 * pickers, in which case results are grouped by type.
 */
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
  /** Hide specific models (e.g. the current model and already-linked ones). */
  excludeKeys?: ReadonlySet<string>;
  /** Extra predicate, e.g. base-architecture compatibility. */
  filter?: (model: ModelConfig) => boolean;
  id?: string;
  invalid?: boolean;
  isClearable?: boolean;
  /** The model types this instance searches. */
  modelTypes: ModelTaxonomyType[];
  onChange: (model: ModelConfig | null) => void;
  placeholder?: string;
  /** Show the shortcut to the model manager. Hide it inside the manager itself. */
  showManagerButton?: boolean;
  size?: 'xs' | 'sm' | 'md';
  /** Selected model key, or null. */
  value: string | null;
}) => {
  const enableModelDescriptions = useWorkbenchPreferenceSelector((preferences) => preferences.enableModelDescriptions);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedBases, setSelectedBases] = useState<ReadonlySet<string>>(EMPTY_BASES);
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const optionRefs = useRef<HTMLElement[]>([]);

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const { availableBases, candidates, groups } = useMemo(
    () =>
      isOpen
        ? getModelPickerGroups(models, {
            baseFilter: selectedBases,
            excludeKeys,
            filter,
            modelTypes,
            searchTerm: deferredSearchTerm,
          })
        : { availableBases: [], candidates: [], groups: [] },
    [deferredSearchTerm, excludeKeys, filter, isOpen, modelTypes, models, selectedBases]
  );

  const selectedModel = useMemo(() => models.find((model) => model.key === value) ?? null, [models, value]);

  const scopeLabel =
    modelTypes.length === 1 ? getModelTypePluralLabel(modelTypes[0] ?? 'main').toLowerCase() : 'models';
  const closeAndReset = () => {
    setIsOpen(false);
    setSearchTerm('');
    setSelectedBases(EMPTY_BASES);
  };

  useEffect(() => {
    if (disabled) {
      closeAndReset();
    }
  }, [disabled]);

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
  const resetBases = () => {
    setSelectedBases(EMPTY_BASES);
  };
  const selectModel = (model: ModelConfig) => {
    onChange(model);
    closeAndReset();
  };
  const clearSelection = () => {
    onChange(null);
    closeAndReset();
  };
  const focusSearchInput = () => {
    window.requestAnimationFrame(() => searchInputRef.current?.focus());
  };
  const focusOption = (edge: 'first' | 'last') => {
    const options = optionRefs.current.filter(Boolean);

    options[edge === 'first' ? 0 : options.length - 1]?.focus();
  };
  const focusAdjacentOption = (currentIndex: number, direction: 1 | -1) => {
    const options = optionRefs.current.filter(Boolean);
    const nextIndex = currentIndex + direction;

    if (nextIndex < 0) {
      searchInputRef.current?.focus();
      return;
    }

    options[nextIndex]?.focus();
  };

  optionRefs.current = [];
  let optionIndex = 0;
  const canClear = isClearable && Boolean(value);

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
            setSearchTerm('');
            setSelectedBases(EMPTY_BASES);
            setIsOpen(true);
            focusSearchInput();
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
              colorPalette={invalid ? 'red' : isOpen ? 'accent' : 'bg'}
              disabled={disabled}
              justifyContent="space-between"
              minW="0"
              pe={canClear ? '7' : '2'}
              ps="2"
              size={size}
              variant="outline"
              w="full"
            >
              {selectedModel ? (
                <ModelButtonContent model={selectedModel} />
              ) : (
                <Text as="span" color="fg.subtle" fontSize="xs" minW="0" truncate>
                  {placeholder ?? `Select ${scopeLabel}…`}
                </Text>
              )}
              {canClear ? null : <Icon as={ChevronDownIcon} boxSize="3" flexShrink={0} />}
            </Button>
          </Popover.Trigger>
          {canClear ? (
            <CloseButton
              aria-label="Clear selected model"
              disabled={disabled}
              insetEnd="1"
              position="absolute"
              size="2xs"
              top="50%"
              transform="translateY(-50%)"
              zIndex="1"
              onClick={(event) => {
                event.stopPropagation();
                clearSelection();
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
              bg="bg.muted"
              borderColor="border.emphasized"
              borderRadius="lg"
              borderWidth="1px"
              boxShadow="lg"
              color="fg"
              maxH="min(18rem, var(--available-height))"
              minW="min(18rem, calc(100vw - 1rem))"
              overflow="hidden"
              p="0"
            >
              <Stack gap="2" p="2">
                <HStack gap="1">
                  <InputGroup flex="1" minW="0" startElement={<Icon as={SearchIcon} size="xs" />}>
                    <Input
                      ref={searchInputRef}
                      aria-label={`Search ${scopeLabel}`}
                      placeholder={`Search ${scopeLabel}...`}
                      size="xs"
                      value={searchTerm}
                      onChange={(event) => setSearchTerm(event.currentTarget.value)}
                      onKeyDown={(event) => {
                        event.stopPropagation();

                        if (event.key === 'Escape') {
                          event.preventDefault();
                          closeAndReset();
                          return;
                        }

                        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                          event.preventDefault();
                          focusOption(event.key === 'ArrowDown' ? 'first' : 'last');
                        }
                      }}
                    />
                  </InputGroup>
                  {showManagerButton ? <ModelManagerLinkButton /> : null}
                </HStack>
                {availableBases.length >= 2 ? (
                  <HStack alignItems="center" flexWrap="wrap" gap="1">
                    {availableBases.map((base) => (
                      <BaseChip key={base} base={base} isSelected={selectedBases.has(base)} onToggle={toggleBase} />
                    ))}
                    <Spacer />
                    <IconButton
                      aria-label="Reset base filters"
                      flexShrink={0}
                      opacity={selectedBases.size === 0 ? 0.5 : undefined}
                      pointerEvents={selectedBases.size === 0 ? 'none' : undefined}
                      size="2xs"
                      variant="ghost"
                      onClick={resetBases}
                    >
                      <Icon as={RotateCcwIcon} boxSize="3" />
                    </IconButton>
                  </HStack>
                ) : null}
              </Stack>
              <HStack borderTopWidth="1px" borderColor="border.subtle" />
              <ScrollArea.Root maxH="14rem" size="xs" variant="hover" w="full">
                <ScrollArea.Viewport maxH="inherit" w="full">
                  <ScrollArea.Content py="1" role="listbox" aria-label={`Available ${scopeLabel}`}>
                    {groups.length === 0 ? (
                      <Text color="fg.subtle" fontSize="2xs" p="2">
                        {candidates.length === 0
                          ? `No compatible ${scopeLabel} installed.`
                          : !deferredSearchTerm.trim() && selectedBases.size > 0
                            ? `No ${scopeLabel} match the selected bases.`
                            : `No ${scopeLabel} match your search.`}
                      </Text>
                    ) : null}
                    {groups.map((group) => (
                      <Stack key={group.type} gap="0">
                        {modelTypes.length > 1 ? (
                          <Text
                            color="fg.subtle"
                            fontSize="2xs"
                            fontWeight="600"
                            letterSpacing="0.02em"
                            px="2"
                            py="1"
                            textTransform="uppercase"
                          >
                            {group.label}
                          </Text>
                        ) : null}
                        {group.models.map((model) => {
                          const currentOptionIndex = optionIndex;
                          optionIndex += 1;

                          return (
                            <Button
                              key={model.key}
                              ref={(element) => {
                                if (element) {
                                  optionRefs.current[currentOptionIndex] = element;
                                }
                              }}
                              aria-selected={value === model.key}
                              borderRadius="0"
                              gap="2"
                              justifyContent="start"
                              truncate
                              role="option"
                              size="2xs"
                              variant="ghost"
                              w="full"
                              _highlighted={{ bg: 'bg.subtle' }}
                              _focusVisible={{
                                outline: '2px solid',
                                outlineColor: 'accent.solid',
                                outlineOffset: '-2px',
                              }}
                              onClick={() => selectModel(model)}
                              onKeyDown={(event) => {
                                if (event.key === 'Escape') {
                                  event.preventDefault();
                                  closeAndReset();
                                  return;
                                }

                                if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                                  event.preventDefault();
                                  focusAdjacentOption(currentOptionIndex, event.key === 'ArrowDown' ? 1 : -1);
                                }
                              }}
                              transitionDuration="faster"
                            >
                              <Icon
                                as={CheckIcon}
                                aria-hidden
                                boxSize="3"
                                color="accent.solid"
                                flexShrink={0}
                                opacity={value === model.key ? 1 : 0}
                              />
                              <ModelOptionContent enableDescription={enableModelDescriptions} model={model} />
                            </Button>
                          );
                        })}
                      </Stack>
                    ))}
                  </ScrollArea.Content>
                </ScrollArea.Viewport>
                <ScrollArea.Scrollbar>
                  <ScrollArea.Thumb />
                </ScrollArea.Scrollbar>
              </ScrollArea.Root>
            </Popover.Content>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
    </Box>
  );
};

const ModelManagerLinkButton = () => {
  const projectId = useActiveProjectId();
  const search = useMemo(() => ({ project: projectId }), [projectId]);

  return (
    <Tooltip content="Manage models" showArrow>
      <IconButton aria-label="Manage models" asChild flexShrink={0} size="xs" variant="ghost">
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
    <Text as="span" fontSize="xs" minW="0" truncate>
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

const ModelOptionContent = ({ enableDescription, model }: { enableDescription: boolean; model: ModelConfig }) => (
  <HStack flex="1" gap="2" minW="0" textAlign="left" w="full">
    <Stack flex="1" gap="0" minW="0">
      <Text fontSize="xs" minW="0" truncate>
        {model.name}
      </Text>
      {enableDescription && model.description ? (
        <Text color="fg.subtle" fontSize="2xs" truncate>
          {model.description}
        </Text>
      ) : null}
    </Stack>
    <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
      {formatBytes(model.file_size)}
    </Text>
    <Badge
      colorPalette={getModelBaseColorPalette(model.base)}
      flexShrink={0}
      fontSize="2xs"
      size="xs"
      variant="surface"
    >
      {getModelBaseLabel(model.base)}
    </Badge>
  </HStack>
);
