/* eslint-disable react/react-compiler, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { AspectRatioId, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, Box, HStack, Icon, InputGroup, NumberInput, Stack, Text } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import {
  ASPECT_RATIO_MAP,
  calculateNewSize,
  clampDimension,
  MAX_DIMENSION,
  MIN_DIMENSION,
} from '@features/generation/core/settings';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { MODEL_DEFAULT_END_ELEMENT_PROPS, ModelDefaultButton } from '@platform/ui/ModelDefaultButton';
import { ArrowLeftRightIcon, LockIcon, RulerDimensionLineIcon } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useGenerationUi } from './GenerationUiContext';
import { AspectRatioChips } from './shared/AspectRatioChips';
import { AspectRatioLockButton } from './shared/AspectRatioSelect';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';

interface GenerateDimensionFieldsProps {
  settings: GenerateSettings;
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

type Dimensions = Pick<GenerateSettings, 'height' | 'width'>;

/** The ratio to enforce, preferring the stored value and falling back to the current dimensions. */
const getActiveRatio = (settings: GenerateSettings): number =>
  settings.aspectRatioValue > 0
    ? settings.aspectRatioValue
    : settings.height > 0
      ? settings.width / settings.height
      : 1;

const PREVIEW_STAGE_PX = 108;
const PREVIEW_PAD_PX = 10;

const clampToRange = (value: number): number => Math.min(MAX_DIMENSION, Math.max(MIN_DIMENSION, value));

/**
 * The preview IS a control: the solid rectangle is the current size, the dashed
 * ghost is the model-recommended size at the same scale, and the corner handle
 * resizes by direct manipulation — grid-snapped on release, ratio-lock
 * respected while dragging. Arrow keys nudge by one grid step.
 */
const SizePreview = ({
  current,
  grid,
  handleLabel,
  isRatioConstrained,
  onDraft,
  onCommitDims,
  ratio,
  recommended,
}: {
  current: Dimensions;
  grid: number;
  handleLabel: string;
  isRatioConstrained: boolean;
  onDraft: (dims: Dimensions) => void;
  onCommitDims: (dims: Dimensions) => void;
  ratio: number;
  recommended: Dimensions;
}) => {
  const dragRef = useRef<{
    scale: number;
    startHeight: number;
    startWidth: number;
    startX: number;
    startY: number;
  } | null>(null);
  const inner = PREVIEW_STAGE_PX - PREVIEW_PAD_PX * 2;
  const maxSide = Math.max(current.width, current.height, recommended.width, recommended.height, 1);
  const scale = inner / maxSide;

  const snap = (dims: Dimensions): Dimensions => {
    const width = clampDimension(dims.width, grid);

    return {
      height: isRatioConstrained && ratio > 0 ? clampDimension(width / ratio, grid) : clampDimension(dims.height, grid),
      width,
    };
  };

  // The rectangle stays centered, so the corner moves at half the size delta —
  // the factor of two keeps the handle tracking the pointer. The scale is the
  // one captured at drag start: the live scale shrinks as the rectangle grows,
  // and reading it mid-drag turns each pointer step into a larger size step — a
  // runaway feedback loop.
  const dimsFromPointer = (event: { clientX: number; clientY: number }): Dimensions | null => {
    const drag = dragRef.current;

    if (!drag) {
      return null;
    }

    const width = clampToRange(drag.startWidth + ((event.clientX - drag.startX) * 2) / drag.scale);
    const height =
      isRatioConstrained && ratio > 0
        ? clampToRange(width / ratio)
        : clampToRange(drag.startHeight + ((event.clientY - drag.startY) * 2) / drag.scale);

    return { height: Math.round(height), width: Math.round(width) };
  };

  const nudge = (dw: number, dh: number) => {
    onCommitDims(
      snap({ height: clampToRange(current.height + dh * grid), width: clampToRange(current.width + dw * grid) })
    );
  };

  return (
    <Box
      aspectRatio="1/1"
      borderColor="bg.emphasized"
      borderWidth={1}
      flexShrink="0"
      position="relative"
      rounded="sm"
      w={`${PREVIEW_STAGE_PX}px`}
    >
      {/* The recommended size, as a ghost the current rectangle can be matched against. */}
      <Box
        borderColor="border.muted"
        borderStyle="dashed"
        borderWidth="1px"
        h={`${recommended.height * scale}px`}
        left="50%"
        pointerEvents="none"
        position="absolute"
        top="50%"
        transform="translate(-50%, -50%)"
        w={`${recommended.width * scale}px`}
      />
      <Box
        bg="bg.emphasized/40"
        borderColor="border.emphasized"
        borderWidth="1.5px"
        h={`${current.height * scale}px`}
        left="50%"
        position="absolute"
        rounded="2px"
        top="50%"
        transform="translate(-50%, -50%)"
        w={`${current.width * scale}px`}
      >
        <Box
          aria-label={handleLabel}
          as="button"
          bg="fg.muted"
          bottom="-4px"
          cursor="nwse-resize"
          h="8px"
          position="absolute"
          right="-4px"
          rounded="1px"
          w="8px"
          _focusVisible={{ outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '1px' }}
          _hover={{ bg: 'fg' }}
          onKeyDown={(event) => {
            const nudges: Record<string, [number, number]> = {
              ArrowDown: [0, 1],
              ArrowLeft: [-1, 0],
              ArrowRight: [1, 0],
              ArrowUp: [0, -1],
            };
            const step = nudges[event.key];

            if (step) {
              event.preventDefault();
              nudge(step[0], step[1]);
            }
          }}
          onPointerDown={(event) => {
            event.preventDefault();
            event.currentTarget.setPointerCapture(event.pointerId);
            dragRef.current = {
              scale,
              startHeight: current.height,
              startWidth: current.width,
              startX: event.clientX,
              startY: event.clientY,
            };
          }}
          onPointerMove={(event) => {
            const dims = dimsFromPointer(event);

            if (dims) {
              onDraft(dims);
            }
          }}
          onPointerUp={(event) => {
            const dims = dimsFromPointer(event);

            dragRef.current = null;

            if (dims) {
              onCommitDims(snap(dims));
            }
          }}
        />
      </Box>
    </Box>
  );
};

export const GenerateDimensionFields = ({
  onCommit,
  projectId,
  selectedModel,
  settings,
}: GenerateDimensionFieldsProps) => {
  const { t } = useTranslation();
  const { secondsPerRun } = useGenerationUi().queueInsights;
  const [draftDimensions, setDraftDimensions] = useState<Dimensions | null>(null);
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const dimensions = getGenerationDimensions(selectedModel);
  const dimensionGrid = dimensions.grid;
  const isRatioConstrained = settings.aspectRatioId !== 'Free' || settings.aspectRatioIsLocked;
  const displayDimensions = draftDimensions ?? { height: settings.height, width: settings.width };
  const onCommitRef = useRef(onCommit);
  const pendingDimensionsRef = useRef<Dimensions | null>(null);
  const projectIdRef = useRef(projectId);
  const previousSettingsDimensionsRef = useRef<Dimensions>({ height: settings.height, width: settings.width });
  const dimensionRatio = displayDimensions.height > 0 ? displayDimensions.width / displayDimensions.height : 1;

  useEffect(() => {
    onCommitRef.current = onCommit;
  }, [onCommit]);

  const commitDimensions = useCallback((dimensions: Dimensions) => {
    pendingDimensionsRef.current = dimensions;
    onCommitRef.current(dimensions);
  }, []);

  useLayoutEffect(() => {
    if (projectIdRef.current === projectId) {
      return;
    }

    projectIdRef.current = projectId;
    pendingDimensionsRef.current = null;
    previousSettingsDimensionsRef.current = { height: settings.height, width: settings.width };
    setDraftDimensions(null);
  }, [projectId, settings.height, settings.width]);

  useEffect(() => {
    const previousSettingsDimensions = previousSettingsDimensionsRef.current;
    previousSettingsDimensionsRef.current = { height: settings.height, width: settings.width };

    if (!draftDimensions) {
      return;
    }

    if (settings.width === draftDimensions.width && settings.height === draftDimensions.height) {
      pendingDimensionsRef.current = null;
      setDraftDimensions(null);
      return;
    }

    if (settings.width === previousSettingsDimensions.width && settings.height === previousSettingsDimensions.height) {
      return;
    }

    pendingDimensionsRef.current = null;
    setDraftDimensions(null);
  }, [draftDimensions, settings.height, settings.width]);

  const getNextDimensions = (key: 'height' | 'width', value: number, shouldSnap: boolean): Dimensions => {
    const nextValue = shouldSnap ? clampDimension(value, dimensionGrid) : value;

    if (!isRatioConstrained) {
      return { ...displayDimensions, [key]: nextValue };
    }

    const ratio = getActiveRatio({ ...settings, ...displayDimensions });

    return key === 'width'
      ? { height: shouldSnap ? clampDimension(nextValue / ratio, dimensionGrid) : nextValue / ratio, width: nextValue }
      : { height: nextValue, width: shouldSnap ? clampDimension(nextValue * ratio, dimensionGrid) : nextValue * ratio };
  };

  const setDimension =
    (key: 'height' | 'width') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      const value = valueAsNumber;

      if (!Number.isFinite(value) || value <= 0) {
        return;
      }

      setDraftDimensions(getNextDimensions(key, value, false));
    };

  const commitDimension =
    (key: 'height' | 'width') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      const value = valueAsNumber;

      if (!Number.isFinite(value) || value <= 0) {
        return;
      }

      const dimensions = getNextDimensions(key, value, true);

      setDraftDimensions(dimensions);
      commitDimensions(dimensions);
    };

  const snapDimension = (key: 'height' | 'width') => () => {
    const snapped = clampDimension(displayDimensions[key], dimensionGrid);

    if (snapped !== displayDimensions[key]) {
      const dimensions = getNextDimensions(key, snapped, true);

      setDraftDimensions(dimensions);
      commitDimensions(dimensions);
    }
  };

  const setDimensionToModelDefault = (key: 'height' | 'width') => {
    if (!modelDefaults) {
      return;
    }

    const dimensions = getNextDimensions(key, modelDefaults[key], true);

    setDraftDimensions(dimensions);
    commitDimensions(dimensions);
  };

  const commitSettings = (patch: Partial<GenerateSettings>) => {
    pendingDimensionsRef.current = null;
    setDraftDimensions(null);
    onCommit(patch);
  };

  const setAspectRatioId = (id: AspectRatioId) => {
    if (id === 'Free') {
      commitSettings({
        aspectRatioId: 'Free',
        aspectRatioIsLocked: false,
        aspectRatioValue: displayDimensions.height > 0 ? displayDimensions.width / displayDimensions.height : 1,
        ...displayDimensions,
      });
      return;
    }

    const ratio = ASPECT_RATIO_MAP[id].ratio;

    commitSettings({
      aspectRatioId: id,
      aspectRatioIsLocked: true,
      aspectRatioValue: ratio,
      ...calculateNewSize(ratio, displayDimensions.width * displayDimensions.height, dimensionGrid),
    });
  };

  const toggleLock = () => {
    commitSettings({
      aspectRatioIsLocked: !settings.aspectRatioIsLocked,
      // Locking in Free mode captures the current ratio so further edits preserve it.
      aspectRatioValue:
        !settings.aspectRatioIsLocked && settings.aspectRatioId === 'Free' && displayDimensions.height > 0
          ? displayDimensions.width / displayDimensions.height
          : settings.aspectRatioValue,
      ...displayDimensions,
    });
  };

  const swapDimensions = () => {
    const inverseId: AspectRatioId =
      settings.aspectRatioId === 'Free' ? 'Free' : ASPECT_RATIO_MAP[settings.aspectRatioId].inverseId;

    commitSettings({
      aspectRatioId: inverseId,
      aspectRatioValue: settings.aspectRatioValue > 0 ? 1 / settings.aspectRatioValue : 1,
      height: displayDimensions.width,
      width: displayDimensions.height,
    });
  };

  const optimizeSize = () => {
    const optimal = dimensions.optimal;
    const ratio = isRatioConstrained
      ? getActiveRatio({ ...settings, ...displayDimensions })
      : displayDimensions.height > 0
        ? displayDimensions.width / displayDimensions.height
        : 1;

    commitSettings(calculateNewSize(ratio, optimal * optimal, dimensionGrid));
  };

  // "Recommended" is what the optimize action would produce for the live ratio:
  // the model's optimal pixel budget reshaped to the current proportions.
  const recommendedDimensions = calculateNewSize(
    dimensionRatio,
    dimensions.optimal * dimensions.optimal,
    dimensionGrid
  );
  const isAtRecommendedSize =
    displayDimensions.width === recommendedDimensions.width &&
    displayDimensions.height === recommendedDimensions.height;
  const megapixels = (displayDimensions.width * displayDimensions.height) / 1_000_000;

  const badges = (
    <>
      <Badge size="xs">
        {displayDimensions.width}x{displayDimensions.height}
      </Badge>
      {settings.aspectRatioIsLocked && (
        <Badge size="xs">
          <Icon as={LockIcon} boxSize="3" />
        </Badge>
      )}
    </>
  );

  const dimensionInput = (key: 'height' | 'width') => (
    <NumberInput.Root
      size="xs"
      allowMouseWheel
      flex="1"
      max={MAX_DIMENSION}
      min={MIN_DIMENSION}
      value={String(displayDimensions[key])}
      step={dimensionGrid}
      onBlur={snapDimension(key)}
      onValueCommit={commitDimension(key)}
      onValueChange={setDimension(key)}
    >
      <InputGroup
        endElement={
          modelDefaults && displayDimensions[key] !== modelDefaults[key] ? (
            <ModelDefaultButton
              label={
                key === 'width'
                  ? t('widgets.generate.useModelDefaultWidth')
                  : t('widgets.generate.useModelDefaultHeight')
              }
              onClick={() => setDimensionToModelDefault(key)}
            />
          ) : undefined
        }
        endElementProps={MODEL_DEFAULT_END_ELEMENT_PROPS}
        startElementProps={{ pointerEvents: 'auto' }}
        startElement={
          <NumberInput.Scrubber>
            <Icon as={RulerDimensionLineIcon} boxSize="3" rotate={key === 'height' ? '90' : undefined} />
          </NumberInput.Scrubber>
        }
      >
        <NumberInput.Input aria-label={key === 'width' ? t('widgets.generate.width') : t('widgets.generate.height')} />
      </InputGroup>
    </NumberInput.Root>
  );

  return (
    <GenerateCollapsibleSection label={t('widgets.generate.size')} badges={badges} defaultOpen sectionId="dimensions">
      <Stack gap="2" p="2">
        <HStack alignItems="stretch" gap="2">
          <Stack flex="1" gap="2" minW="0">
            {/* The lock binds width to height, so it sits between the two values it couples. */}
            <GenerateFieldContextMenu
              copyValue={() => `${displayDimensions.width}x${displayDimensions.height}`}
              isAtDefault={
                modelDefaults !== null &&
                displayDimensions.width === modelDefaults.width &&
                displayDimensions.height === modelDefaults.height
              }
              onReset={
                modelDefaults
                  ? () => {
                      setDraftDimensions(null);
                      commitDimensions({ height: modelDefaults.height, width: modelDefaults.width });
                    }
                  : undefined
              }
            >
              <HStack alignItems="center" gap="1">
                {dimensionInput('width')}
                <AspectRatioLockButton isLocked={settings.aspectRatioIsLocked} onToggle={toggleLock} />
                {dimensionInput('height')}
              </HStack>
            </GenerateFieldContextMenu>
            {/* Every preset stays visible in the run beneath the values it
                reshapes — nothing hides behind an overflow menu. */}
            <HStack alignItems="flex-start" gap="1">
              <AspectRatioChips
                fallbackRatio={dimensionRatio}
                value={settings.aspectRatioId}
                onChange={setAspectRatioId}
              />
              <Tooltip content={t('widgets.generate.swapWidthAndHeight')}>
                <IconButton
                  aria-label={t('widgets.generate.swapWidthAndHeight')}
                  flexShrink="0"
                  size="2xs"
                  variant="outline"
                  onClick={swapDimensions}
                >
                  <ArrowLeftRightIcon />
                </IconButton>
              </Tooltip>
            </HStack>
            <HStack gap="2" justify="space-between" minH="5" mt="auto">
              <Text color="fg.subtle" fontSize="2xs">
                {t('widgets.generate.megapixelsValue', { value: megapixels.toFixed(2) })}
                {isAtRecommendedSize ? ` · ${t('widgets.generate.sizeRecommended')}` : ''}
                {/* Grounded in this project's recent completed runs, never a guess. */}
                {secondsPerRun !== null
                  ? ` · ${t('widgets.generate.secondsPerRun', { value: Math.round(secondsPerRun) })}`
                  : ''}
              </Text>
              {isAtRecommendedSize ? null : (
                <Tooltip content={t('widgets.generate.setOptimalSizeDescription')}>
                  <Button color="fg.muted" size="2xs" variant="ghost" onClick={optimizeSize}>
                    {t('widgets.generate.setOptimalSize')}
                  </Button>
                </Tooltip>
              )}
            </HStack>
          </Stack>
          <SizePreview
            current={displayDimensions}
            grid={dimensionGrid}
            handleLabel={t('widgets.generate.sizePreviewHandle')}
            isRatioConstrained={isRatioConstrained}
            ratio={isRatioConstrained ? getActiveRatio({ ...settings, ...displayDimensions }) : dimensionRatio}
            recommended={recommendedDimensions}
            onCommitDims={(dims) => {
              setDraftDimensions(dims);
              commitDimensions(dims);
            }}
            onDraft={setDraftDimensions}
          />
        </HStack>
      </Stack>
    </GenerateCollapsibleSection>
  );
};
