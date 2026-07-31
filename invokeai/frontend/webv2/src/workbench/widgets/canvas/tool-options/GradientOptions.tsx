import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasLayerSourceContract, GradientStop, GradientToolOptions } from '@workbench/canvas-engine/api';

import { createListCollection, HStack, NumberInput, Text } from '@chakra-ui/react';
import { ColorPicker, Select } from '@platform/ui';
import { useGradientOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;
type GradientKind = GradientToolOptions['kind'];

interface SelectedGradient {
  id: string;
  source: GradientSource;
}

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const SELECT_TRIGGER_PROPS = { minW: '6rem' } as const;

/**
 * Gradient tool options: kind (linear/radial), angle (degrees), and a MINIMAL
 * two-stop editor — start/end color, each carrying its own alpha. Edits set
 * defaults for the next created gradient AND apply to a selected gradient layer
 * (colors commit one history entry on interaction end; discrete edits commit at
 * once). A full multi-stop editor is a follow-up.
 */
export const GradientOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useGradientOptions(engine);
  const sampleColor = useColorSampler(engine);

  const selected = useActiveProjectSelector(
    (project): SelectedGradient | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId
        ? document.layers.find((entry) => entry.id === document.selectedLayerId)
        : undefined;
      if (layer && layer.type === 'raster' && layer.source.type === 'gradient') {
        return { id: layer.id, source: layer.source };
      }
      return null;
    },
    (a, b) => a?.id === b?.id && a?.source === b?.source
  );

  const kind: GradientKind = selected ? selected.source.kind : options.kind;
  const angle = selected ? selected.source.angle : options.angle;
  const stops = selected ? selected.source.stops : options.stops;
  const start = stops[0] ?? { color: '#000000ff', offset: 0 };
  const end = stops[stops.length - 1] ?? { color: '#ffffffff', offset: 1 };

  const kindCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: GradientKind }>({
        items: [
          { label: t('widgets.canvas.toolOptions.gradientLinear'), value: 'linear' },
          { label: t('widgets.canvas.toolOptions.gradientRadial'), value: 'radial' },
        ],
      }),
    [t]
  );
  const kindValue = useMemo(() => [kind], [kind]);

  const applyGradient = useCallback(
    (next: { kind: GradientKind; angle: number; stops: GradientStop[] }, commit: boolean) => {
      engine.interaction.set('gradientOptions', { angle: next.angle, kind: next.kind, stops: next.stops });
      if (selected && commit) {
        const before = selected.source;
        const after: GradientSource = { ...before, angle: next.angle, kind: next.kind, stops: next.stops };
        engine.layers.commitStructural(
          t('widgets.canvas.toolOptions.gradientEdit'),
          { id: selected.id, source: after, type: 'updateCanvasLayerSource' },
          { id: selected.id, source: before, type: 'updateCanvasLayerSource' }
        );
      }
    },
    [engine, selected, t]
  );

  const setStopColor = useCallback(
    (index: number, color: string, commit: boolean) => {
      const nextStops = stops.map((stop, i) => (i === index ? { ...stop, color } : stop));
      applyGradient({ angle, kind, stops: nextStops }, commit);
    },
    [angle, applyGradient, kind, stops]
  );

  const lastIndex = stops.length - 1;

  const onKindChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: GradientKind }>) => {
      const next = value[0] as GradientKind | undefined;
      if (next && next !== kind) {
        applyGradient({ angle, kind: next, stops: [...stops] }, true);
      }
    },
    [angle, applyGradient, kind, stops]
  );

  const onAngleChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        applyGradient({ angle: Math.round(valueAsNumber), kind, stops: [...stops] }, true);
      }
    },
    [applyGradient, kind, stops]
  );

  const onStartColorChange = useCallback((color: string) => setStopColor(0, color, false), [setStopColor]);
  const onStartColorEnd = useCallback((color: string) => setStopColor(0, color, true), [setStopColor]);

  const onEndColorChange = useCallback(
    (color: string) => setStopColor(lastIndex, color, false),
    [lastIndex, setStopColor]
  );
  const onEndColorEnd = useCallback((color: string) => setStopColor(lastIndex, color, true), [lastIndex, setStopColor]);

  return (
    <HStack align="center" gap="3">
      <Select
        aria-label={t('widgets.canvas.toolOptions.gradientKind')}
        collection={kindCollection}
        positioning={SELECT_POSITIONING}
        size="xs"
        triggerProps={SELECT_TRIGGER_PROPS}
        value={kindValue}
        valueText={t(
          kind === 'radial' ? 'widgets.canvas.toolOptions.gradientRadial' : 'widgets.canvas.toolOptions.gradientLinear'
        )}
        onValueChange={onKindChange}
      />

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientAngle')}
        </Text>
        <NumberInput.Root
          disabled={kind === 'radial'}
          max={360}
          min={-360}
          size="xs"
          value={String(Math.round(angle))}
          w="4.5rem"
          onValueChange={onAngleChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.gradientAngle')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientStart')}
        </Text>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.gradientStart')}
          value={start.color}
          withAlpha
          onSampleColor={sampleColor}
          onValueChange={onStartColorChange}
          onValueChangeEnd={onStartColorEnd}
        />
      </HStack>

      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.gradientEnd')}
        </Text>
        <ColorPicker
          aria-label={t('widgets.canvas.toolOptions.gradientEnd')}
          value={end.color}
          withAlpha
          onSampleColor={sampleColor}
          onValueChange={onEndColorChange}
          onValueChangeEnd={onEndColorEnd}
        />
      </HStack>
    </HStack>
  );
};
