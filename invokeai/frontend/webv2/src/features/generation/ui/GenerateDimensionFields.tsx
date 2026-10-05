/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { AspectRatioId, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import {
  ASPECT_RATIO_MAP,
  calculateNewSize,
  clampDimension,
  deriveAspectRatioId,
  MAX_DIMENSION,
  MIN_DIMENSION,
} from '@features/generation/core/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { ArrowLeftRightIcon, LockIcon } from 'lucide-react';
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
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { GenerateFieldContextMenu } from './shared/GenerateFieldContextMenu';

interface GenerateDimensionFieldsProps {
  draft: GenerateDraft;
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
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
  'height',
  'width',
]);

export const GenerateDimensionFields = ({
  draft,
  onCommit,
  projectId,
  selectedModel,
}: GenerateDimensionFieldsProps) => {
  const { t } = useTranslation();
  const { CanvasRenderSize: CanvasRenderSizeSlot, project } = useGenerationUi();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectDimensionSettings);
  const secondsPerRun = useGenerationQueueInsights(selectSecondsPerRun);
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

  const optimizeSize = () => {
    const optimal = dimensions.optimal;
    const ratio = isRatioConstrained
      ? getActiveRatio({ aspectRatioValue: settings.aspectRatioValue, ...displayDimensions })
      : displayDimensions.height > 0
        ? displayDimensions.width / displayDimensions.height
        : 1;

    commitSettings(calculateNewSize(ratio, optimal * optimal, dimensionGrid));
  };

  // The recommendation reshapes the model's pixel budget to the live aspect ratio.
  const recommendedDimensions = calculateNewSize(
    dimensionRatio,
    dimensions.optimal * dimensions.optimal,
    dimensionGrid
  );
  const isAtRecommendedSize =
    displayDimensions.width === recommendedDimensions.width &&
    displayDimensions.height === recommendedDimensions.height;
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
    // In canvas mode, resizing the frame only adds something while it renders at its own size: Auto has not already
    // grown it toward the optimum and Custom has not set a size of its own.
    const offersOptimalSize = !isAtRecommendedSize && !isResized;

    const footer = (
      <HStack gap="2" justify="space-between" minH="5" mt="auto">
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
        {offersOptimalSize ? (
          <Tooltip content={t('widgets.generate.setOptimalSizeDescription')}>
            <Button color="fg.muted" size="sm" variant="ghost" onClick={optimizeSize}>
              {t('widgets.generate.setOptimalSize')}
            </Button>
          </Tooltip>
        ) : null}
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

              <HStack alignItems="center" gap="1">
                <AspectRatioSelect
                  fallbackRatio={dimensionRatio}
                  value={settings.aspectRatioId}
                  onChange={setAspectRatioId}
                />
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
  return project.invocationSourceId === 'canvas' ? (
    <CanvasRenderSizeSlot frame={displayDimensions}>{renderSection}</CanvasRenderSizeSlot>
  ) : (
    renderSection(null)
  );
};
