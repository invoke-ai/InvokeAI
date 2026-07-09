import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { Rect } from '@workbench/canvas-engine/types';

import { createListCollection, HStack, NumberInput, Text } from '@chakra-ui/react';
import { bboxEquals, constrainBboxToRatio, roundBbox } from '@workbench/canvas-engine/tools/bboxHitTest';
import { Select, ToggleDot } from '@workbench/components/ui';
import { useBboxGrid, useBboxOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

/** Aspect-ratio presets offered in the options bar. `null` ratio = Free (unlocked). */
interface AspectPreset {
  id: string;
  ratio: number | null;
}

const ASPECT_PRESETS: readonly AspectPreset[] = [
  { id: 'Free', ratio: null },
  { id: '1:1', ratio: 1 },
  { id: '4:3', ratio: 4 / 3 },
  { id: '3:4', ratio: 3 / 4 },
  { id: '16:9', ratio: 16 / 9 },
  { id: '9:16', ratio: 9 / 16 },
];

const ASPECT_TRIGGER_PROPS = { minW: '5.5rem' } as const;

const bboxEqualsSelected = (a: Rect | null, b: Rect | null): boolean =>
  a === b || (!!a && !!b && a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height);

/** The preset id whose ratio matches `aspectRatio` (within tolerance), or `null`. */
const matchingPresetId = (aspectRatio: number): string | null =>
  ASPECT_PRESETS.find((preset) => preset.ratio !== null && Math.abs(preset.ratio - aspectRatio) < 1e-3)?.id ?? null;

/**
 * Bbox tool options: numeric W/H and X/Y of the generation frame (document px,
 * snap-rounded), an aspect-ratio preset select, and an aspect lock toggle. Frame
 * edits commit through the engine's structural history (shared canvas undo stack
 * with drags); the aspect lock/ratio lives in the engine's transient store.
 */
export const BboxOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const bbox = useActiveProjectSelector((project) => project.canvas.document.bbox, bboxEqualsSelected);
  const options = useBboxOptions(engine);
  const grid = useBboxGrid(engine);

  const collection = useMemo(
    () =>
      createListCollection({
        itemToString: (item: AspectPreset) => item.id,
        itemToValue: (item: AspectPreset) => item.id,
        items: [...ASPECT_PRESETS],
      }),
    []
  );

  const selectValue = useMemo(
    () => [options.aspectLocked ? (matchingPresetId(options.aspectRatio) ?? 'Free') : 'Free'],
    [options.aspectLocked, options.aspectRatio]
  );

  const commitBbox = useCallback(
    (next: Rect) => {
      const rounded = roundBbox(next);
      if (bboxEquals(rounded, bbox)) {
        return;
      }
      engine.commitStructural(
        t('widgets.canvas.toolOptions.setFrame'),
        { bbox: rounded, type: 'setCanvasBbox' },
        { bbox: roundBbox(bbox), type: 'setCanvasBbox' }
      );
    },
    [bbox, engine, t]
  );

  const setWidth = useCallback(
    (value: number) => {
      const width = Math.max(1, Math.round(value / grid) * grid);
      const height =
        options.aspectLocked && options.aspectRatio > 0
          ? Math.max(1, Math.round(width / options.aspectRatio))
          : bbox.height;
      commitBbox({ height, width, x: bbox.x, y: bbox.y });
    },
    [bbox, commitBbox, grid, options.aspectLocked, options.aspectRatio]
  );

  const setHeight = useCallback(
    (value: number) => {
      const height = Math.max(1, Math.round(value / grid) * grid);
      const width =
        options.aspectLocked && options.aspectRatio > 0
          ? Math.max(1, Math.round(height * options.aspectRatio))
          : bbox.width;
      commitBbox({ height, width, x: bbox.x, y: bbox.y });
    },
    [bbox, commitBbox, grid, options.aspectLocked, options.aspectRatio]
  );

  const setX = useCallback(
    (value: number) => commitBbox({ ...bbox, x: Math.round(value / grid) * grid }),
    [bbox, commitBbox, grid]
  );

  const setY = useCallback(
    (value: number) => commitBbox({ ...bbox, y: Math.round(value / grid) * grid }),
    [bbox, commitBbox, grid]
  );

  const onAspectPresetChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const id = value[0];
      const preset = ASPECT_PRESETS.find((entry) => entry.id === id);
      if (!preset) {
        return;
      }
      if (preset.ratio === null) {
        engine.stores.bboxOptions.set({ ...options, aspectLocked: false });
        return;
      }
      engine.stores.bboxOptions.set({ aspectLocked: true, aspectRatio: preset.ratio });
      commitBbox(constrainBboxToRatio(bbox, preset.ratio, grid));
    },
    [bbox, commitBbox, engine, grid, options]
  );

  const onLockToggle = useCallback(
    (checked: boolean) => {
      const aspectRatio =
        checked && bbox.height > 0 && matchingPresetId(options.aspectRatio) === null
          ? bbox.width / bbox.height
          : options.aspectRatio > 0
            ? options.aspectRatio
            : 1;
      engine.stores.bboxOptions.set({ aspectLocked: checked, aspectRatio });
    },
    [bbox, engine, options.aspectRatio]
  );

  const onWidthChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setWidth(valueAsNumber);
      }
    },
    [setWidth]
  );
  const onHeightChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setHeight(valueAsNumber);
      }
    },
    [setHeight]
  );
  const onXChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setX(valueAsNumber);
      }
    },
    [setX]
  );
  const onYChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setY(valueAsNumber);
      }
    },
    [setY]
  );

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.frameWidth')}
        </Text>
        <NumberInput.Root min={1} size="xs" value={String(bbox.width)} w="5rem" onValueChange={onWidthChange}>
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.frameWidth')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.frameHeight')}
        </Text>
        <NumberInput.Root min={1} size="xs" value={String(bbox.height)} w="5rem" onValueChange={onHeightChange}>
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.frameHeight')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionX')}
        </Text>
        <NumberInput.Root size="xs" value={String(bbox.x)} w="5rem" onValueChange={onXChange}>
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionX')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionY')}
        </Text>
        <NumberInput.Root size="xs" value={String(bbox.y)} w="5rem" onValueChange={onYChange}>
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionY')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <Select
        aria-label={t('widgets.canvas.toolOptions.aspectRatio')}
        collection={collection}
        size="xs"
        triggerProps={ASPECT_TRIGGER_PROPS}
        value={selectValue}
        valueText={selectValue[0]}
        onValueChange={onAspectPresetChange}
      />
      <ToggleDot
        checked={options.aspectLocked}
        label={t('widgets.canvas.toolOptions.lockAspect')}
        onCheckedChange={onLockToggle}
      />
    </HStack>
  );
};
