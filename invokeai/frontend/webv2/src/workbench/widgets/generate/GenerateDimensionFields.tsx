import {
  Badge,
  Box,
  createListCollection,
  HStack,
  Icon,
  InputGroup,
  NumberInput,
  Portal,
  Select,
  Stack,
  Text,
} from '@chakra-ui/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeftRightIcon, LockIcon, LockOpenIcon, RulerDimensionLineIcon, ScalingIcon } from 'lucide-react';

import { IconButton } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { Tooltip } from '../../components/ui/Tooltip';
import {
  ASPECT_RATIO_MAP,
  ASPECT_RATIO_OPTIONS,
  calculateNewSize,
  clampDimension,
  getOptimalDimension,
  isAspectRatioId,
} from '../../generation/settings';
import type { AspectRatioId, GenerateSettings, MainModelConfig } from '../../generation/types';
import { AspectRatioPreview } from './shared/AspectRatioPreview';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateDimensionFieldsProps {
  settings: GenerateSettings;
  selectedModel: MainModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

type Dimensions = Pick<GenerateSettings, 'height' | 'width'>;
type AspectRatioOption = { id: AspectRatioId; ratio: number };

const DIMENSION_COMMIT_DEBOUNCE_MS = 150;

/** The ratio to enforce, preferring the stored value and falling back to the current dimensions. */
const getActiveRatio = (settings: GenerateSettings): number =>
  settings.aspectRatioValue > 0
    ? settings.aspectRatioValue
    : settings.height > 0
      ? settings.width / settings.height
      : 1;

const getAspectRatioOptionRatio = (id: AspectRatioId, fallbackRatio: number): number =>
  id === 'Free' ? fallbackRatio : ASPECT_RATIO_MAP[id].ratio;

export const GenerateDimensionFields = ({ onCommit, selectedModel, settings }: GenerateDimensionFieldsProps) => {
  const [draftDimensions, setDraftDimensions] = useState<Dimensions | null>(null);
  const isRatioConstrained = settings.aspectRatioId !== 'Free' || settings.aspectRatioIsLocked;
  const displayDimensions = draftDimensions ?? { height: settings.height, width: settings.width };
  const onCommitRef = useRef(onCommit);
  const pendingDimensionsRef = useRef<Dimensions | null>(null);
  const commitTimeoutRef = useRef<number | null>(null);
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

  const clearPendingDimensionCommit = useCallback(() => {
    if (commitTimeoutRef.current !== null) {
      window.clearTimeout(commitTimeoutRef.current);
      commitTimeoutRef.current = null;
    }
  }, []);

  const commitDimensions = useCallback(
    (dimensions: Dimensions) => {
      clearPendingDimensionCommit();
      pendingDimensionsRef.current = dimensions;
      onCommitRef.current(dimensions);
    },
    [clearPendingDimensionCommit]
  );

  const scheduleDimensionCommit = useCallback(
    (dimensions: Dimensions) => {
      clearPendingDimensionCommit();
      pendingDimensionsRef.current = dimensions;
      commitTimeoutRef.current = window.setTimeout(() => {
        commitTimeoutRef.current = null;

        if (pendingDimensionsRef.current) {
          onCommitRef.current(pendingDimensionsRef.current);
        }
      }, DIMENSION_COMMIT_DEBOUNCE_MS);
    },
    [clearPendingDimensionCommit]
  );

  useEffect(
    () => () => {
      clearPendingDimensionCommit();
    },
    [clearPendingDimensionCommit]
  );

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
    clearPendingDimensionCommit();
    setDraftDimensions(null);
  }, [clearPendingDimensionCommit, draftDimensions, settings.height, settings.width]);

  const getNextDimensions = (key: 'height' | 'width', value: number): Dimensions => {
    if (!isRatioConstrained) {
      return { ...displayDimensions, [key]: value };
    }

    const ratio = getActiveRatio({ ...settings, ...displayDimensions });

    return key === 'width'
      ? { height: clampDimension(value / ratio), width: value }
      : { height: value, width: clampDimension(value * ratio) };
  };

  const setDimension =
    (key: 'height' | 'width') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      const value = valueAsNumber;

      if (!Number.isFinite(value) || value <= 0) {
        return;
      }

      const dimensions = getNextDimensions(key, value);

      setDraftDimensions(dimensions);
      scheduleDimensionCommit(dimensions);
    };

  const commitDimension =
    (key: 'height' | 'width') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      const value = valueAsNumber;

      if (!Number.isFinite(value) || value <= 0) {
        return;
      }

      const dimensions = getNextDimensions(key, value);

      setDraftDimensions(dimensions);
      commitDimensions(dimensions);
    };

  const snapDimension = (key: 'height' | 'width') => () => {
    const snapped = clampDimension(displayDimensions[key]);

    if (snapped !== displayDimensions[key]) {
      const dimensions = getNextDimensions(key, snapped);

      setDraftDimensions(dimensions);
      commitDimensions(dimensions);
    }
  };

  const commitSettings = (patch: Partial<GenerateSettings>) => {
    clearPendingDimensionCommit();
    pendingDimensionsRef.current = null;
    setDraftDimensions(null);
    onCommit(patch);
  };

  const setAspectRatioId = ({ value }: Select.ValueChangeDetails<AspectRatioOption>) => {
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
      ...calculateNewSize(ratio, displayDimensions.width * displayDimensions.height),
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
    const optimal = getOptimalDimension(selectedModel?.base ?? 'sdxl');
    const ratio = isRatioConstrained
      ? getActiveRatio({ ...settings, ...displayDimensions })
      : displayDimensions.height > 0
        ? displayDimensions.width / displayDimensions.height
        : 1;

    commitSettings(calculateNewSize(ratio, optimal * optimal));
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
    <GenerateCollapsibleSection label="Dimensions" badges={badges} defaultOpen>
      <Field label="Aspect ratio" p="2">
        <HStack gap="1">
          <Select.Root
            collection={aspectRatioCollection}
            flex="1"
            size="xs"
            value={[settings.aspectRatioId]}
            onValueChange={setAspectRatioId}
          >
            <Select.HiddenSelect />
            <Select.Control>
              <Select.Trigger>
                <Select.ValueText>
                  <HStack as="span" gap="2" minW="0">
                    <AspectRatioPreview boxSize="5" ratio={activeAspectRatioPreviewRatio} />
                    <Text as="span" fontSize="xs" truncate>
                      {settings.aspectRatioId}
                    </Text>
                  </HStack>
                </Select.ValueText>
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Portal>
              <Select.Positioner>
                <Select.Content
                  bg="bg.muted"
                  borderColor="border.emphasized"
                  borderWidth="1px"
                  color="fg"
                  maxH="18rem"
                  rounded="lg"
                  shadow="lg"
                  zIndex="popover"
                >
                  {aspectRatioOptions.map((option) => (
                    <Select.Item key={option.id} item={option}>
                      <Select.ItemText>
                        <HStack as="span" gap="2">
                          <AspectRatioPreview boxSize="6" ratio={option.ratio} />
                          <Text as="span" fontSize="xs">
                            {option.id}
                          </Text>
                        </HStack>
                      </Select.ItemText>
                      <Select.ItemIndicator />
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Positioner>
            </Portal>
          </Select.Root>
          <Tooltip content={settings.aspectRatioIsLocked ? 'Unlock aspect ratio' : 'Lock aspect ratio'}>
            <IconButton
              aria-label={settings.aspectRatioIsLocked ? 'Unlock aspect ratio' : 'Lock aspect ratio'}
              size="xs"
              variant={settings.aspectRatioIsLocked ? 'solid' : 'outline'}
              onClick={toggleLock}
            >
              {settings.aspectRatioIsLocked ? <LockIcon /> : <LockOpenIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip content="Swap width and height">
            <IconButton aria-label="Swap width and height" size="xs" variant="outline" onClick={swapDimensions}>
              <ArrowLeftRightIcon />
            </IconButton>
          </Tooltip>
          <Tooltip content="Set size to the model's optimal pixel count">
            <IconButton aria-label="Set optimal size" size="xs" variant="outline" onClick={optimizeSize}>
              <ScalingIcon />
            </IconButton>
          </Tooltip>
        </HStack>
      </Field>
      <HStack alignItems="flex-start" gap="2" p="2">
        <Stack w="full">
          <Field label="Width">
            <NumberInput.Root
              size="xs"
              allowMouseWheel
              value={String(displayDimensions.width)}
              step={8}
              onBlur={snapDimension('width')}
              onValueCommit={commitDimension('width')}
              onValueChange={setDimension('width')}
            >
              <NumberInput.Control />
              <InputGroup
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
          <Field label="Height">
            <NumberInput.Root
              size="xs"
              allowMouseWheel
              value={String(displayDimensions.height)}
              step={8}
              onBlur={snapDimension('height')}
              onValueCommit={commitDimension('height')}
              onValueChange={setDimension('height')}
            >
              <NumberInput.Control />
              <InputGroup
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
