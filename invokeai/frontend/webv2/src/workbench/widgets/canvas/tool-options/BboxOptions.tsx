import type { SelectValueChangeDetails } from '@chakra-ui/react';

import { createListCollection, HStack } from '@chakra-ui/react';
import { Select, ToggleDot } from '@platform/ui';
import { constrainBboxToRatio } from '@workbench/canvas-engine/api';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { useBboxEditor } from './useBboxEditor';

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

/** The preset id whose ratio matches `aspectRatio` (within tolerance), or `null`. */
const matchingPresetId = (aspectRatio: number): string | null =>
  ASPECT_PRESETS.find((preset) => preset.ratio !== null && Math.abs(preset.ratio - aspectRatio) < 1e-3)?.id ?? null;

/**
 * Bbox tool options: aspect-ratio preset select and aspect lock toggle. Numeric
 * frame details live in the separate bbox details bar.
 */
export const BboxOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const { bbox, commitBbox, grid, options } = useBboxEditor(engine);

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

  const onAspectPresetChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const id = value[0];
      const preset = ASPECT_PRESETS.find((entry) => entry.id === id);
      if (!preset) {
        return;
      }
      if (preset.ratio === null) {
        engine.interaction.set('bboxOptions', { ...options, aspectLocked: false });
        return;
      }
      engine.interaction.set('bboxOptions', { aspectLocked: true, aspectRatio: preset.ratio });
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
      engine.interaction.set('bboxOptions', { aspectLocked: checked, aspectRatio });
    },
    [bbox, engine, options.aspectRatio]
  );

  return (
    <HStack align="center" gap="3">
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
