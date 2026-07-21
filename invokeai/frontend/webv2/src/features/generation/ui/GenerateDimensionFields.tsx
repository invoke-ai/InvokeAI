/* eslint-disable react/react-compiler, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { AspectRatioId, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { Badge, Box, createListCollection, HStack, Icon, InputGroup, NumberInput, Stack, Text } from '@chakra-ui/react';
import { getDefaultGenerateSettings, getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import {
  ASPECT_RATIO_MAP,
  ASPECT_RATIO_OPTIONS,
  calculateNewSize,
  clampDimension,
  isAspectRatioId,
  MAX_DIMENSION,
  MIN_DIMENSION,
} from '@features/generation/core/settings';
import { Field, IconButton, Select, Tooltip } from '@platform/ui';
import { ArrowLeftRightIcon, LockIcon, LockOpenIcon, RulerDimensionLineIcon, ScalingIcon } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AspectRatioPreview } from './shared/AspectRatioPreview';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';

interface GenerateDimensionFieldsProps {
  settings: GenerateSettings;
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

type Dimensions = Pick<GenerateSettings, 'height' | 'width'>;
type AspectRatioOption = { id: AspectRatioId; ratio: number };

/** The ratio to enforce, preferring the stored value and falling back to the current dimensions. */
const getActiveRatio = (settings: GenerateSettings): number =>
  settings.aspectRatioValue > 0
    ? settings.aspectRatioValue
    : settings.height > 0
      ? settings.width / settings.height
      : 1;

const getAspectRatioOptionRatio = (id: AspectRatioId, fallbackRatio: number): number =>
  id === 'Free' ? fallbackRatio : ASPECT_RATIO_MAP[id].ratio;

export const GenerateDimensionFields = ({
  onCommit,
  projectId,
  selectedModel,
  settings,
}: GenerateDimensionFieldsProps) => {
  const { t } = useTranslation();
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

  const aspectRatioOptions = useMemo<AspectRatioOption[]>(
    () =>
      ASPECT_RATIO_OPTIONS.map((id) => ({
        id,
        ratio: getAspectRatioOptionRatio(id, dimensionRatio),
      })),
    [dimensionRatio]
  );

  const aspectRatioCollection = useMemo(
    () =>
      createListCollection({
        itemToString: (item: AspectRatioOption) => item.id,
        itemToValue: (item: AspectRatioOption) => item.id,
        items: aspectRatioOptions,
      }),
    [aspectRatioOptions]
  );

  const activeAspectRatioPreviewRatio = getAspectRatioOptionRatio(settings.aspectRatioId, dimensionRatio);

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

  const setAspectRatioId = ({ value }: SelectValueChangeDetails<AspectRatioOption>) => {
    const id = value[0];

    if (!isAspectRatioId(id)) {
      return;
    }

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

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.dimensions')}
      badges={badges}
      defaultOpen
      sectionId="dimensions"
    >
      <Field label={t('widgets.generate.aspectRatio')} p="2">
        <HStack gap="1">
          <Select
            collection={aspectRatioCollection}
            contentProps={{ maxH: '18rem' }}
            flex="1"
            renderItem={(option) => (
              <HStack as="span" gap="2">
                <AspectRatioPreview boxSize="6" ratio={option.ratio} />
                <Text as="span" fontSize="xs">
                  {option.id}
                </Text>
              </HStack>
            )}
            size="xs"
            value={[settings.aspectRatioId]}
            valueText={
              <HStack as="span" gap="2" minW="0">
                <AspectRatioPreview boxSize="5" ratio={activeAspectRatioPreviewRatio} />
                <Text as="span" fontSize="xs" truncate>
                  {settings.aspectRatioId}
                </Text>
              </HStack>
            }
            onValueChange={setAspectRatioId}
          />
          <Tooltip
            content={
              settings.aspectRatioIsLocked
                ? t('widgets.generate.unlockAspectRatio')
                : t('widgets.generate.lockAspectRatio')
            }
          >
            <IconButton
              aria-label={
                settings.aspectRatioIsLocked
                  ? t('widgets.generate.unlockAspectRatio')
                  : t('widgets.generate.lockAspectRatio')
              }
              size="xs"
              variant={settings.aspectRatioIsLocked ? 'solid' : 'outline'}
              onClick={toggleLock}
            >
              {settings.aspectRatioIsLocked ? <LockIcon /> : <LockOpenIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip content={t('widgets.generate.swapWidthAndHeight')}>
            <IconButton
              aria-label={t('widgets.generate.swapWidthAndHeight')}
              size="xs"
              variant="outline"
              onClick={swapDimensions}
            >
              <ArrowLeftRightIcon />
            </IconButton>
          </Tooltip>
          <Tooltip content={t('widgets.generate.setOptimalSizeDescription')}>
            <IconButton
              aria-label={t('widgets.generate.setOptimalSize')}
              size="xs"
              variant="outline"
              onClick={optimizeSize}
            >
              <ScalingIcon />
            </IconButton>
          </Tooltip>
        </HStack>
      </Field>
      <HStack alignItems="flex-start" gap="2" p="2">
        <Stack w="full">
          <Field
            label={t('widgets.generate.width')}
            helpText={t('widgets.generate.multipleOf', { value: dimensionGrid })}
          >
            <NumberInput.Root
              size="xs"
              allowMouseWheel
              max={MAX_DIMENSION}
              min={MIN_DIMENSION}
              value={String(displayDimensions.width)}
              step={dimensionGrid}
              onBlur={snapDimension('width')}
              onValueCommit={commitDimension('width')}
              onValueChange={setDimension('width')}
            >
              <InputGroup
                endElement={
                  modelDefaults && displayDimensions.width !== modelDefaults.width ? (
                    <ModelDefaultButton
                      label={t('widgets.generate.useModelDefaultWidth')}
                      onClick={() => setDimensionToModelDefault('width')}
                    />
                  ) : undefined
                }
                endElementProps={{ pointerEvents: 'auto' }}
                startElementProps={{ pointerEvents: 'auto' }}
                startElement={
                  <NumberInput.Scrubber>
                    <Icon as={RulerDimensionLineIcon} boxSize="3" />
                  </NumberInput.Scrubber>
                }
              >
                <NumberInput.Input />
              </InputGroup>
            </NumberInput.Root>
          </Field>
          <Field
            label={t('widgets.generate.height')}
            helpText={t('widgets.generate.multipleOf', { value: dimensionGrid })}
          >
            <NumberInput.Root
              size="xs"
              allowMouseWheel
              max={MAX_DIMENSION}
              min={MIN_DIMENSION}
              value={String(displayDimensions.height)}
              step={dimensionGrid}
              onBlur={snapDimension('height')}
              onValueCommit={commitDimension('height')}
              onValueChange={setDimension('height')}
            >
              <InputGroup
                endElement={
                  modelDefaults && displayDimensions.height !== modelDefaults.height ? (
                    <ModelDefaultButton
                      label={t('widgets.generate.useModelDefaultHeight')}
                      onClick={() => setDimensionToModelDefault('height')}
                    />
                  ) : undefined
                }
                endElementProps={{ pointerEvents: 'auto' }}
                startElementProps={{ pointerEvents: 'auto' }}
                startElement={
                  <NumberInput.Scrubber>
                    <Icon as={RulerDimensionLineIcon} boxSize="3" rotate="90" />
                  </NumberInput.Scrubber>
                }
              >
                <NumberInput.Input />
              </InputGroup>
            </NumberInput.Root>
          </Field>
        </Stack>
        <Box w="1/3" aspectRatio="1/1" borderColor="bg.emphasized" borderWidth={1} p={1} rounded="sm">
          <AspectRatioPreview boxSize="full" ratio={dimensionRatio} />
        </Box>
      </HStack>
    </GenerateCollapsibleSection>
  );
};
