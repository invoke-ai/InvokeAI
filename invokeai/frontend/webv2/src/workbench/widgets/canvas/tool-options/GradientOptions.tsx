import type { CanvasLayerSourceContract, GradientStop, GradientToolOptions } from '@workbench/canvas-engine/api';
import type { ToolFormProps, ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { Box, chakra, Flex, IconButton as ChakraIconButton, Icon } from '@chakra-ui/react';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { getDocumentLayer } from '@workbench/canvas-engine/api';
import { useActiveColorPair } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useGradientOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import {
  FormNumberField,
  FormSlider,
  useNumberCommit,
  useSliderGesture,
} from '@workbench/widgets/canvas/tool-presentation/FormControls';
import {
  EditTargetChip,
  PropertyControlRow,
  PropertySegmentedRow,
  PropertySwitchRow,
} from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { PlusIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { insertStopAt, moveStop, recolorStop, removeStop, stopsToCssGradient } from './gradientStops';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;
type GradientKind = GradientToolOptions['kind'];

interface SelectedGradient {
  id: string;
  name: string;
  source: GradientSource;
}

/**
 * Edit selected gradients or creation defaults; FG-to-BG follows the live pair until explicit stop edits switch to
 * custom colors. Stop gestures commit once on release.
 */
const useGradientEditor = (engine: ToolFormProps['engine']) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const options = useGradientOptions(engine);
  const pair = useActiveColorPair();
  const selected = useActiveProjectSelector(
    (project): SelectedGradient | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId ? getDocumentLayer(document, document.selectedLayerId) : undefined;
      return layer && layer.type === 'raster' && layer.source.type === 'gradient'
        ? { id: layer.id, name: layer.name, source: layer.source }
        : null;
    },
    (a, b) => a?.id === b?.id && a?.name === b?.name && a?.source === b?.source
  );
  const kind: GradientKind = selected ? selected.source.kind : options.kind;
  const angle = selected ? selected.source.angle : options.angle;
  const pairStops = useMemo<GradientStop[]>(
    () => [
      { color: `${pair.foreground}ff`, offset: 0 },
      { color: `${pair.background}ff`, offset: 1 },
    ],
    [pair.background, pair.foreground]
  );
  const stops = selected ? selected.source.stops : options.preset === 'pair' ? pairStops : options.stops;

  const apply = useCallback(
    (next: { angle: number; kind: GradientKind; stops: GradientStop[] }, commit: boolean) => {
      engine.interaction.set('gradientOptions', { ...options, angle: next.angle, kind: next.kind });
      if (selected && commit) {
        const after: GradientSource = { ...selected.source, ...next };
        commitPrepared(t('widgets.canvas.toolOptions.gradientEdit'), (model) =>
          model.prepare({ id: selected.id, source: after, type: 'patch-source' })
        );
      }
    },
    [commitPrepared, engine, options, selected, t]
  );
  const setCustomStops = useCallback(
    (nextStops: GradientStop[]) => {
      engine.interaction.set('gradientOptions', { ...options, preset: 'custom', stops: nextStops });
    },
    [engine, options]
  );
  const setPreset = useCallback(
    (preset: 'pair' | 'custom') => {
      // Leaving the preset keeps what the strip currently shows as the custom set.
      engine.interaction.set('gradientOptions', {
        ...options,
        preset,
        stops:
          preset === 'custom' && options.preset === 'pair' ? pairStops.map((stop) => ({ ...stop })) : options.stops,
      });
    },
    [engine, options, pairStops]
  );
  /** Routes a settled stop list to its owner: the selected layer (one entry) or the custom defaults. */
  const commitStops = useCallback(
    (nextStops: GradientStop[]) => {
      if (selected) {
        apply({ angle, kind, stops: nextStops }, true);
        return;
      }
      setCustomStops(nextStops);
    },
    [angle, apply, kind, selected, setCustomStops]
  );
  return {
    angle,
    apply,
    commitStops,
    kind,
    preset: selected ? null : options.preset,
    selectedName: selected?.name ?? null,
    setPreset,
    stops,
  };
};

const STRIP_CHECKER = 'repeating-conic-gradient(#00000022 0% 25%, transparent 0% 50%) 0 0 / 10px 10px';

/** Add stops from ramp colors and move by drag/keyboard; preview locally and commit one history entry on release. */
const GradientStopStrip = ({
  onCommit,
  onSelect,
  selectedIndex,
  stops,
}: {
  onCommit: (stops: GradientStop[]) => void;
  onSelect: (index: number) => void;
  selectedIndex: number;
  stops: readonly GradientStop[];
}) => {
  const { t } = useTranslation();
  const rampRef = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState<GradientStop[] | null>(null);
  const dragIndex = useRef<number>(-1);
  const shown = draft ?? stops;

  const offsetFromClientX = useCallback((clientX: number): number => {
    const rect = rampRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) {
      return 0;
    }
    return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  }, []);

  const onRampPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.target !== rampRef.current) {
        return;
      }
      const { index, stops: next } = insertStopAt(stops, offsetFromClientX(event.clientX));
      onCommit(next);
      onSelect(index);
    },
    [offsetFromClientX, onCommit, onSelect, stops]
  );

  const onDragStart = useCallback((index: number) => {
    dragIndex.current = index;
  }, []);
  const onDragMove = useCallback(
    (clientX: number) => {
      if (dragIndex.current < 0) {
        return;
      }
      // The base index stays fixed while the gesture runs; only the draft shows the reorder.
      setDraft(moveStop(stops, dragIndex.current, offsetFromClientX(clientX)).stops);
    },
    [offsetFromClientX, stops]
  );
  const onDragEnd = useCallback(
    (clientX: number) => {
      if (dragIndex.current < 0) {
        return;
      }
      const settled = moveStop(stops, dragIndex.current, offsetFromClientX(clientX));
      dragIndex.current = -1;
      setDraft(null);
      onCommit(settled.stops);
      onSelect(settled.index);
    },
    [offsetFromClientX, onCommit, onSelect, stops]
  );
  const rampStyle = useMemo(() => ({ background: `${stopsToCssGradient(shown)}, ${STRIP_CHECKER}` }), [shown]);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const pendingFocus = useRef<number>(-1);
  // A key-move can reorder past a neighbour, so the handle at the OLD index now
  // renders a different stop; refocus the moved stop at its new index.
  useEffect(() => {
    if (pendingFocus.current < 0) {
      return;
    }
    const handle = rootRef.current?.querySelector<HTMLButtonElement>(`[data-stop-index="${pendingFocus.current}"]`);
    pendingFocus.current = -1;
    handle?.focus();
  });
  const onKeyMove = useCallback(
    (index: number, direction: -1 | 1, big: boolean) => {
      const target = stops[index];
      if (!target) {
        return;
      }
      const moved = moveStop(stops, index, target.offset + direction * (big ? 0.1 : 0.01));
      pendingFocus.current = moved.index;
      onCommit(moved.stops);
      onSelect(moved.index);
    },
    [onCommit, onSelect, stops]
  );
  const onDragCancel = useCallback(() => {
    dragIndex.current = -1;
    setDraft(null);
  }, []);

  return (
    <Box ref={rootRef} position="relative" py="1" w="full">
      <Box
        ref={rampRef}
        aria-label={t('widgets.canvas.toolOptions.gradientStops')}
        cursor="copy"
        h="6"
        role="img"
        rounded="sm"
        style={rampStyle}
        w="full"
        onPointerDown={onRampPointerDown}
      />
      {shown.map((stop, index) => (
        <StopHandle
          // Offsets can collide mid-drag, so position in the list is the identity.
          // oxlint-disable-next-line react/no-array-index-key
          key={index}
          index={index}
          selected={index === selectedIndex}
          stop={stop}
          onDragCancel={onDragCancel}
          onDragEnd={onDragEnd}
          onDragMove={onDragMove}
          onDragStart={onDragStart}
          onKeyMove={onKeyMove}
          onSelect={onSelect}
        />
      ))}
    </Box>
  );
};

const StopHandle = ({
  index,
  onDragCancel,
  onDragEnd,
  onDragMove,
  onDragStart,
  onKeyMove,
  onSelect,
  selected,
  stop,
}: {
  index: number;
  onDragCancel: () => void;
  onDragEnd: (clientX: number) => void;
  onDragMove: (clientX: number) => void;
  onDragStart: (index: number) => void;
  onKeyMove: (index: number, direction: -1 | 1, big: boolean) => void;
  onSelect: (index: number) => void;
  selected: boolean;
  stop: GradientStop;
}) => {
  const { t } = useTranslation();
  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      onDragStart(index);
      onSelect(index);
    },
    [index, onDragStart, onSelect]
  );
  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => onDragMove(event.clientX),
    [onDragMove]
  );
  const onPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => onDragEnd(event.clientX),
    [onDragEnd]
  );
  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      const direction = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (direction === 0) {
        return;
      }
      event.preventDefault();
      onKeyMove(index, direction, event.shiftKey);
    },
    [index, onKeyMove]
  );
  return (
    <chakra.button
      aria-label={t('widgets.canvas.toolOptions.gradientStopAt', { percent: Math.round(stop.offset * 100) })}
      aria-pressed={selected}
      bg={stop.color}
      borderColor={selected ? 'fg' : 'border.emphasized'}
      borderWidth="2px"
      boxSize="3.5"
      cursor="ew-resize"
      left={`calc(${(stop.offset * 100).toFixed(2)}% - 7px)`}
      position="absolute"
      rounded="full"
      top="50%"
      transform="translateY(-50%)"
      type="button"
      data-stop-index={index}
      onKeyDown={onKeyDown}
      onPointerCancel={onDragCancel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  );
};

const GradientSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const editor = useGradientEditor(engine);
  const kindOptions = useMemo(
    () => [
      { label: t('widgets.canvas.toolOptions.gradientLinear'), value: 'linear' as const },
      { label: t('widgets.canvas.toolOptions.gradientRadial'), value: 'radial' as const },
    ],
    [t]
  );
  const setKind = useCallback(
    (next: GradientKind) => editor.apply({ angle: editor.angle, kind: next, stops: [...editor.stops] }, true),
    [editor]
  );
  // Ticks preview through the options store; ONE document commit lands on release.
  const previewAngle = useCallback(
    (value: number) => editor.apply({ angle: Math.round(value), kind: editor.kind, stops: [...editor.stops] }, false),
    [editor]
  );
  const setAngle = useCallback(
    (value: number) => editor.apply({ angle: Math.round(value), kind: editor.kind, stops: [...editor.stops] }, true),
    [editor]
  );
  const angleGesture = useSliderGesture(Math.round(editor.angle), setAngle, previewAngle);
  const onAngleCommit = useNumberCommit(setAngle);
  return (
    <>
      <EditTargetChip layerName={editor.selectedName} />
      <PropertySegmentedRow
        label={t('widgets.properties.rows.kind')}
        options={kindOptions}
        value={editor.kind}
        onValueChange={setKind}
      />
      <PropertyControlRow label={t('widgets.properties.rows.angle')}>
        <FormSlider
          aria-label={t('widgets.canvas.toolOptions.gradientAngle')}
          disabled={editor.kind === 'radial'}
          max={360}
          min={-360}
          value={angleGesture.value}
          onValueChange={angleGesture.onChange}
          onValueChangeEnd={angleGesture.onChangeEnd}
        />
        <FormNumberField
          aria-label={t('widgets.canvas.toolOptions.gradientAngle')}
          disabled={editor.kind === 'radial'}
          max={360}
          min={-360}
          suffix="°"
          value={String(Math.round(editor.angle))}
          onValueCommit={onAngleCommit}
        />
      </PropertyControlRow>
    </>
  );
};

const GradientStopsSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const editor = useGradientEditor(engine);
  const sampleColor = useColorSampler(engine);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const stopCount = editor.stops.length;
  const stop = editor.stops[Math.min(selectedIndex, stopCount - 1)] ?? null;
  const stopIndex = Math.min(selectedIndex, stopCount - 1);

  const onPresetToggle = useCallback((checked: boolean) => editor.setPreset(checked ? 'pair' : 'custom'), [editor]);
  const onColorChange = useCallback(
    (hex: string) => {
      // Live color edits settle on release; the defaults' custom set is its own owner.
      if (editor.preset !== null) {
        editor.commitStops(recolorStop(editor.stops, stopIndex, hex));
      }
    },
    [editor, stopIndex]
  );
  const onColorChangeEnd = useCallback(
    (hex: string) => editor.commitStops(recolorStop(editor.stops, stopIndex, hex)),
    [editor, stopIndex]
  );
  const setOffset = useCallback(
    (value: number) => {
      const moved = moveStop(editor.stops, stopIndex, value / 100);
      editor.commitStops(moved.stops);
      setSelectedIndex(moved.index);
    },
    [editor, stopIndex]
  );
  const onOffsetCommit = useNumberCommit(setOffset);
  const onRemove = useCallback(() => {
    editor.commitStops(removeStop(editor.stops, stopIndex));
    setSelectedIndex(Math.max(0, stopIndex - 1));
  }, [editor, stopIndex]);
  // The keyboard path to what a ramp click does: a new stop halfway to the
  // next one (or back toward the previous from the last stop).
  const onAdd = useCallback(() => {
    const current = editor.stops[stopIndex];
    if (!current) {
      return;
    }
    const neighbour = editor.stops[stopIndex + 1] ?? editor.stops[stopIndex - 1];
    const at = neighbour ? (current.offset + neighbour.offset) / 2 : 0.5;
    const inserted = insertStopAt(editor.stops, at);
    editor.commitStops(inserted.stops);
    setSelectedIndex(inserted.index);
  }, [editor, stopIndex]);

  return (
    <>
      {editor.preset !== null ? (
        <PropertySwitchRow
          checked={editor.preset === 'pair'}
          label={t('widgets.canvas.toolOptions.gradientPairPreset')}
          onCheckedChange={onPresetToggle}
        />
      ) : null}
      <GradientStopStrip
        selectedIndex={stopIndex}
        stops={editor.stops}
        onCommit={editor.commitStops}
        onSelect={setSelectedIndex}
      />
      {stop ? (
        <PropertyControlRow label={t('widgets.properties.rows.stop')}>
          <Flex align="center" gap="2" minW="0">
            <ColorPicker
              aria-label={t('widgets.canvas.toolOptions.gradientStopColor')}
              value={stop.color}
              withAlpha
              onSampleColor={sampleColor}
              onValueChange={onColorChange}
              onValueChangeEnd={onColorChangeEnd}
            />
            <FormNumberField
              aria-label={t('widgets.properties.rows.offset')}
              max={100}
              min={0}
              suffix="%"
              value={String(Math.round(stop.offset * 100))}
              onValueCommit={onOffsetCommit}
            />
          </Flex>
          <Flex gap="0.5">
            <ChakraIconButton
              aria-label={t('widgets.canvas.toolOptions.gradientAddStop')}
              size="2xs"
              variant="ghost"
              onClick={onAdd}
            >
              <Icon as={PlusIcon} boxSize="3.5" />
            </ChakraIconButton>
            <ChakraIconButton
              aria-label={t('widgets.canvas.toolOptions.gradientRemoveStop')}
              disabled={stopCount <= 2}
              size="2xs"
              variant="ghost"
              onClick={onRemove}
            >
              <Icon as={Trash2Icon} boxSize="3.5" />
            </ChakraIconButton>
          </Flex>
        </PropertyControlRow>
      ) : null}
    </>
  );
};

export const gradientForm: ToolPropertyForm = {
  groups: [
    { body: GradientSettings, id: 'gradient', labelKey: 'widgets.properties.groups.gradient' },
    { body: GradientStopsSettings, id: 'gradient-stops', labelKey: 'widgets.properties.groups.stops' },
  ],
  id: 'gradient',
  paintsLeaf: true,
};
