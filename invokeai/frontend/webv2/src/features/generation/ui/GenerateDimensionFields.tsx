/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { AspectRatioId, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, Box, HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import { MAX_DIMENSION, MIN_DIMENSION } from '@features/generation/core/dimensionLimits';
import {
  ASPECT_RATIO_MAP,
  calculateNewSize,
  clampDimension,
  deriveAspectRatioId,
  isAspectRatioLocked,
} from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ArrowLeftRightIcon, CheckIcon, LockIcon, RotateCcwIcon } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { type GenerateDraft, pickGenerateSettings } from './generateDebounce';
import {
  type CanvasRenderSize,
  type GenerationQueueInsights,
  useGenerationQueueInsights,
  useGenerationUi,
} from './GenerationUiContext';
import { AspectRatioLockButton, AspectRatioSelect } from './shared/AspectRatioSelect';
import { ExternalSizeSelect } from './shared/ExternalSizeSelect';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';

interface GenerateDimensionFieldsProps {
  draft: GenerateDraft;
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
  /** Applies a Reset size choice at once; on the canvas it resizes the frame around its centre. */
  onResetSize: (patch: Partial<GenerateSettings>) => void;
}

type Dimensions = Pick<GenerateSettings, 'height' | 'width'>;

/** The ratio to enforce, preferring the stored value and falling back to the current dimensions. */
const getActiveRatio = (settings: Pick<GenerateSettings, 'aspectRatioValue' | 'height' | 'width'>): number =>
  settings.aspectRatioValue > 0
    ? settings.aspectRatioValue
    : settings.height > 0
      ? settings.width / settings.height
      : 1;

const PREVIEW_STAGE_PX = 108;
/** The scrub range covers everyday sizes; typing still reaches `MAX_DIMENSION`. */
const DIMENSION_SLIDER_MAX = 2048;
/** Vertical gap between the width and height rows; the lock bracket's geometry assumes it. */
const DIMENSION_ROW_GAP = '2';

/** Geometry assumes 28px rows, an 8px row gap, a 4px column gap, and a 24px lock button. */
const LOCK_BRACKET_PATH = 'M0 14H12a4 4 0 0 1 4 4v28a4 4 0 0 1-4 4H0';
const LOCK_BRACKET_CSS = {
  alignItems: 'center',
  alignSelf: 'stretch',
  display: 'flex',
  flexShrink: 0,
  justifyContent: 'center',
  position: 'relative',
  w: '6',
  '& [data-part="bracket"]': {
    fill: 'none',
    h: '64px',
    insetInlineStart: '-4px',
    pointerEvents: 'none',
    position: 'absolute',
    stroke: 'border',
    strokeWidth: '1px',
    top: 0,
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'stroke',
    w: '28px',
  },
  '&[data-locked] [data-part="bracket"]': { stroke: 'accent.solid' },
};
/** Same width as the lock column so the swap lines up under the lock. */
const SWAP_COLUMN_CSS = { display: 'flex', flexShrink: 0, justifyContent: 'center', w: '6' };
const PREVIEW_PAD_PX = 10;

const clampToRange = (value: number): number => Math.min(MAX_DIMENSION, Math.max(MIN_DIMENSION, value));

/** Whether the model sets a size of its own; otherwise its defaults are just the architecture's optimal size. */
const hasModelDefaultSize = (model: GenerateModelConfig | undefined): boolean => {
  const defaults = model?.default_settings;

  return Number.isFinite(defaults?.width) || Number.isFinite(defaults?.height);
};

const RESET_SIZE_MENU_POSITIONING = { placement: 'bottom-end' } as const;

interface ResetSizeChoice {
  value: string;
  label: string;
  size: Dimensions;
  /** Already applied: shown checked and disabled. */
  isCurrent: boolean;
  onSelect: () => void;
}

/** The ratio row's Reset size menu; each choice shows the size it sets. */
const ResetSizeMenu = ({ choices }: { choices: readonly ResetSizeChoice[] }) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const label = t('widgets.generate.resetSize');

  return (
    <Menu.Root ids={ids} positioning={RESET_SIZE_MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <Tooltip content={label} ids={ids}>
          <IconButton aria-label={label} size="sm" variant="outline">
            <RotateCcwIcon />
          </IconButton>
        </Tooltip>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="15rem">
            {choices.map((choice) => (
              <Menu.Item
                key={choice.value}
                aria-current={choice.isCurrent ? 'true' : undefined}
                disabled={choice.isCurrent}
                value={choice.value}
                onSelect={choice.onSelect}
              >
                <HStack gap="2" minW="0" w="full">
                  <Icon
                    as={CheckIcon}
                    aria-hidden
                    boxSize="3.5"
                    color="fg.subtle"
                    flexShrink={0}
                    opacity={choice.isCurrent ? 1 : 0}
                  />
                  <Text flex="1" fontSize="md" truncate>
                    {choice.label}
                  </Text>
                  <Text color="fg.subtle" flexShrink={0} fontSize="sm" fontVariantNumeric="tabular-nums">
                    {`${choice.size.width}×${choice.size.height}`}
                  </Text>
                </HStack>
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/** Respect ratio lock during drag and snap to the grid on release; arrow keys move by one grid step. */
const SizePreview = ({
  current,
  ghost,
  grid,
  handleLabel,
  isRatioConstrained,
  onDraft,
  onCommitDims,
  ratio,
}: {
  current: Dimensions;
  /** A dashed size to match the current rectangle against. */
  ghost: Dimensions;
  grid: number;
  handleLabel: string;
  isRatioConstrained: boolean;
  onDraft: (dims: Dimensions) => void;
  onCommitDims: (dims: Dimensions) => void;
  ratio: number;
}) => {
  const dragRef = useRef<{
    scale: number;
    startHeight: number;
    startWidth: number;
    startX: number;
    startY: number;
  } | null>(null);
  const inner = PREVIEW_STAGE_PX - PREVIEW_PAD_PX * 2;
  const maxSide = Math.max(current.width, current.height, ghost.width, ghost.height, 1);
  const scale = inner / maxSide;

  const snap = (dims: Dimensions): Dimensions => {
    const width = clampDimension(dims.width, grid);

    return {
      height: isRatioConstrained && ratio > 0 ? clampDimension(width / ratio, grid) : clampDimension(dims.height, grid),
      width,
    };
  };

  // Centered resize doubles pointer displacement; freeze the drag-start scale to avoid feedback.
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
      <Box
        borderColor="border.muted"
        borderStyle="dashed"
        borderWidth="1px"
        h={`${ghost.height * scale}px`}
        left="50%"
        pointerEvents="none"
        position="absolute"
        top="50%"
        transform="translate(-50%, -50%)"
        w={`${ghost.width * scale}px`}
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

const selectSecondsPerRun = (insights: GenerationQueueInsights) => insights.secondsPerRun;

const selectDimensionSettings = pickGenerateSettings([
  'aspectRatioId',
  'aspectRatioIsLocked',
  'aspectRatioValue',
  'externalImageSize',
  'height',
  'width',
]);

export const GenerateDimensionFields = ({
  draft,
  onCommit,
  onResetSize,
  projectId,
  selectedModel,
}: GenerateDimensionFieldsProps) => {
  const { i18n, t } = useTranslation();
  const { CanvasRenderSize: CanvasRenderSizeSlot, project } = useGenerationUi();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectDimensionSettings);
  const secondsPerRun = useGenerationQueueInsights(selectSecondsPerRun);
  const [draftDimensions, setDraftDimensions] = useState<Dimensions | null>(null);
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const dimensions = getGenerationDimensions(selectedModel);
  const dimensionGrid = dimensions.grid;
  const isRatioConstrained = isAspectRatioLocked(settings);
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
      // eslint-disable-next-line react/set-state-in-effect
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

    const ratio = getActiveRatio({ aspectRatioValue: settings.aspectRatioValue, ...displayDimensions });

    return key === 'width'
      ? { height: shouldSnap ? clampDimension(nextValue / ratio, dimensionGrid) : nextValue / ratio, width: nextValue }
      : { height: nextValue, width: shouldSnap ? clampDimension(nextValue * ratio, dimensionGrid) : nextValue * ratio };
  };

  const commitDimension = (key: 'height' | 'width') => (value: number) => {
    const dimensions = getNextDimensions(key, value, true);

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

  // Presets imply a lock; free ratios capture the current ratio. Unlocking selects Free.
  const toggleLock = () => {
    if (isRatioConstrained) {
      commitSettings({
        aspectRatioId: 'Free',
        aspectRatioIsLocked: false,
        aspectRatioValue: dimensionRatio,
        ...displayDimensions,
      });
      return;
    }

    const id = deriveAspectRatioId(displayDimensions.width, displayDimensions.height);

    commitSettings({
      aspectRatioId: id,
      aspectRatioIsLocked: true,
      aspectRatioValue: id === 'Free' ? dimensionRatio : ASPECT_RATIO_MAP[id].ratio,
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

  const resetSize = (patch: Partial<GenerateSettings>) => {
    pendingDimensionsRef.current = null;
    setDraftDimensions(null);
    onResetSize(patch);
  };

  const nativeSide = clampDimension(dimensions.optimal, dimensionGrid);
  const optimalRatio = isRatioConstrained
    ? getActiveRatio({ aspectRatioValue: settings.aspectRatioValue, ...displayDimensions })
    : dimensionRatio;
  // The model's pixel budget reshaped to the ratio the section holds.
  const optimalSize = calculateNewSize(optimalRatio, dimensions.optimal * dimensions.optimal, dimensionGrid);
  const isAtSize = (size: Dimensions) =>
    displayDimensions.width === size.width && displayDimensions.height === size.height;
  // A preset by its name and a locked free ratio as `W:1`; an unlocked free ratio is just whatever the size is now.
  const optimalLabel =
    settings.aspectRatioId !== 'Free'
      ? t('widgets.generate.resetSizeOptimalForRatio', { ratio: settings.aspectRatioId })
      : settings.aspectRatioIsLocked
        ? t('widgets.generate.resetSizeOptimalForRatio', {
            ratio: `${new Intl.NumberFormat(i18n.resolvedLanguage, { maximumFractionDigits: 2 }).format(optimalRatio)}:1`,
          })
        : t('widgets.generate.resetSizeOptimalForCurrentRatio');

  const resetToModelDefaults = () => {
    if (modelDefaults) {
      resetSize({ height: modelDefaults.height, width: modelDefaults.width });
    }
  };
  // Only a size the model sets itself is worth offering; otherwise its defaults are the native square.
  const modelDefaultSize =
    modelDefaults && hasModelDefaultSize(selectedModel)
      ? { height: modelDefaults.height, width: modelDefaults.width }
      : null;

  const resetSizeChoices: readonly ResetSizeChoice[] = [
    {
      isCurrent:
        isAtSize({ height: nativeSide, width: nativeSide }) &&
        settings.aspectRatioId === '1:1' &&
        settings.aspectRatioIsLocked,
      label: t('widgets.generate.resetSizeSquareNative'),
      // The field set choosing the 1:1 preset writes.
      onSelect: () =>
        resetSize({
          aspectRatioId: '1:1',
          aspectRatioIsLocked: true,
          aspectRatioValue: 1,
          height: nativeSide,
          width: nativeSide,
        }),
      size: { height: nativeSide, width: nativeSide },
      value: 'square-native',
    },
    // A square ratio's optimal size is the native square, which the first choice already sets.
    ...(optimalSize.width === nativeSide && optimalSize.height === nativeSide
      ? []
      : [
          {
            isCurrent: isAtSize(optimalSize),
            label: optimalLabel,
            onSelect: () => resetSize(optimalSize),
            size: optimalSize,
            value: 'optimal-for-ratio',
          },
        ]),
    ...(modelDefaultSize
      ? [
          {
            isCurrent: isAtSize(modelDefaultSize),
            label: t('widgets.generate.resetSizeModelDefaults'),
            onSelect: resetToModelDefaults,
            size: modelDefaultSize,
            value: 'model-defaults',
          },
        ]
      : []),
  ];

  // The recommendation reshapes the model's pixel budget to the live aspect ratio.
  const recommendedDimensions = calculateNewSize(
    dimensionRatio,
    dimensions.optimal * dimensions.optimal,
    dimensionGrid
  );
  const frameLabel = `${displayDimensions.width}x${displayDimensions.height}`;

  // The optimal-side stop is scalar; recommended dimensions preserve the live ratio.
  const dimensionField = (key: 'height' | 'width') => (
    <ScrubberField
      defaultValue={modelDefaults?.[key]}
      hint={key}
      inputMax={MAX_DIMENSION}
      label={key === 'width' ? t('widgets.generate.width') : t('widgets.generate.height')}
      marks={[dimensions.optimal, recommendedDimensions[key]]}
      max={DIMENSION_SLIDER_MAX}
      min={MIN_DIMENSION}
      step={dimensionGrid}
      value={displayDimensions[key]}
      onChange={commitDimension(key)}
    />
  );

  /** In canvas mode the frame is generated at the render size; elsewhere the two are the same. */
  const renderSection = (canvas: CanvasRenderSize | null) => {
    const renderSize = canvas?.size ?? displayDimensions;
    const isResized = renderSize.width !== displayDimensions.width || renderSize.height !== displayDimensions.height;
    const isAtRecommendedRenderSize =
      renderSize.width === recommendedDimensions.width && renderSize.height === recommendedDimensions.height;

    const footer = (
      <HStack gap="2" minH="5" mt="auto">
        <Text color="fg.muted" fontSize="xs">
          {isResized ? `${renderSize.width}×${renderSize.height} · ` : ''}
          {t('widgets.generate.megapixelsValue', {
            value: ((renderSize.width * renderSize.height) / 1_000_000).toFixed(2),
          })}
          {isAtRecommendedRenderSize ? ` · ${t('widgets.generate.sizeRecommended')}` : ''}
          {/* Grounded in this project's recent completed runs, never a guess. */}
          {secondsPerRun !== null
            ? ` · ${t('widgets.generate.secondsPerRun', { value: Math.round(secondsPerRun) })}`
            : ''}
        </Text>
      </HStack>
    );

    return (
      <GenerateCollapsibleSection
        label={t('widgets.generate.size')}
        badges={
          <>
            <Badge>{isResized ? `${frameLabel} → ${renderSize.width}x${renderSize.height}` : frameLabel}</Badge>
            {isRatioConstrained && (
              <Badge>
                <Icon as={LockIcon} boxSize="3" />
              </Badge>
            )}
          </>
        }
        defaultOpen
        sectionId="dimensions"
      >
        <Stack gap="2" p="2">
          <HStack alignItems="stretch" gap="2">
            <Stack flex="1" gap="2" minW="0">
              {/* The lock binds width to height, so it sits between the two values it couples. */}
              <GenerateFieldContextMenu
                copyValue={() => frameLabel}
                isAtDefault={
                  modelDefaults !== null &&
                  displayDimensions.width === modelDefaults.width &&
                  displayDimensions.height === modelDefaults.height
                }
                onReset={modelDefaults ? resetToModelDefaults : undefined}
              >
                <HStack alignItems="center" gap="1">
                  <Stack flex="1" gap={DIMENSION_ROW_GAP} minW="0">
                    {dimensionField('width')}
                    {dimensionField('height')}
                  </Stack>
                  <Box css={LOCK_BRACKET_CSS} data-locked={isRatioConstrained ? '' : undefined}>
                    <svg aria-hidden="true" data-part="bracket" viewBox="0 0 28 64">
                      <path d={LOCK_BRACKET_PATH} />
                    </svg>
                    <AspectRatioLockButton isLocked={isRatioConstrained} size="sm" onToggle={toggleLock} />
                  </Box>
                </HStack>
              </GenerateFieldContextMenu>

              {selectedModel?.type === 'external_image_generator' ? (
                <ExternalSizeSelect
                  isCanvasFrame={project.invocationSourceId === 'canvas'}
                  model={selectedModel}
                  settings={{
                    externalImageSize: settings.externalImageSize,
                    height: displayDimensions.height,
                    width: displayDimensions.width,
                  }}
                  onCommit={commitSettings}
                />
              ) : null}
              <HStack alignItems="center" gap="1">
                <AspectRatioSelect
                  fallbackRatio={dimensionRatio}
                  value={settings.aspectRatioId}
                  onChange={setAspectRatioId}
                />
                <ResetSizeMenu choices={resetSizeChoices} />
                <Box css={SWAP_COLUMN_CSS}>
                  <Tooltip content={t('widgets.generate.swapWidthAndHeight')}>
                    <IconButton
                      aria-label={t('widgets.generate.swapWidthAndHeight')}
                      size="sm"
                      variant="outline"
                      onClick={swapDimensions}
                    >
                      <ArrowLeftRightIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </HStack>
              {canvas ? null : footer}
            </Stack>
            <SizePreview
              current={displayDimensions}
              ghost={canvas ? renderSize : recommendedDimensions}
              grid={dimensionGrid}
              handleLabel={t('widgets.generate.sizePreviewHandle')}
              isRatioConstrained={isRatioConstrained}
              ratio={
                isRatioConstrained
                  ? getActiveRatio({ aspectRatioValue: settings.aspectRatioValue, ...displayDimensions })
                  : dimensionRatio
              }
              onCommitDims={(dims) => {
                setDraftDimensions(dims);
                commitDimensions(dims);
              }}
              onDraft={setDraftDimensions}
            />
          </HStack>
          {canvas ? (
            <>
              {canvas.controls}
              {footer}
            </>
          ) : null}
        </Stack>
      </GenerateCollapsibleSection>
    );
  };

  // Switching source remounts the section's subtree; that is deliberate, as its state lives in this component.
  // A hosted provider is asked for the frame itself (it buckets to its own size table), so there is no render size.
  return project.invocationSourceId === 'canvas' && selectedModel?.type !== 'external_image_generator' ? (
    <CanvasRenderSizeSlot frame={displayDimensions}>{renderSection}</CanvasRenderSizeSlot>
  ) : (
    renderSection(null)
  );
};
