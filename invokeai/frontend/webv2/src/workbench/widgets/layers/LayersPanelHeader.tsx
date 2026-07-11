import type {
  NumberInput as ChakraNumberInput,
  SelectValueChangeDetails,
  SliderValueChangeDetails,
} from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasBlendMode, CanvasLayerContract, CanvasMaskFillContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { Box, createListCollection, Flex, HStack, NumberInput, Stack } from '@chakra-ui/react';
import { ColorPicker, Field, Select, Slider } from '@workbench/components/ui';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import {
  CANVAS_DENOISING_STRENGTH_KEY,
  clampCanvasDenoisingStrength,
  MAX_CANVAS_DENOISING_STRENGTH,
  MIN_CANVAS_DENOISING_STRENGTH,
  readCanvasDenoisingStrength,
} from '@workbench/widgets/canvas/invoke/canvasStrength';
import { useCanvasEngine } from '@workbench/widgets/canvas/useCanvasEngine';
import { useRegisterGenerateDraftFlusher } from '@workbench/widgets/generate/generateDraftRegistry';
import { useDebouncedDraftValue } from '@workbench/widgets/generate/useDebouncedDraftValue';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { DenoisingStrengthWave } from './DenoisingStrengthWave';
import { applyStructural, applyStructuralPreview, CANVAS_BLEND_MODES } from './layerOps';

const STRENGTH_DEBOUNCE_MS = 250;
const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;

const formatStrengthPercent = (value: number): string => `${Math.round(value * 100)}%`;
const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

/** A mask layer whose fill colour the header swatch edits (inpaint mask / region). */
type MaskLayer = Extract<CanvasLayerContract, { type: 'inpaint_mask' | 'regional_guidance' }>;

const isMaskLayer = (layer: CanvasLayerContract | null): layer is MaskLayer =>
  layer !== null && (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance');

const selectSelectedLayer = (project: {
  canvas: { document: { layers: readonly CanvasLayerContract[]; selectedLayerId: string | null } };
}): CanvasLayerContract | null => {
  const { layers, selectedLayerId } = project.canvas.document;
  return layers.find((layer) => layer.id === selectedLayerId) ?? null;
};

export const isSameSelection = (left: CanvasLayerContract | null, right: CanvasLayerContract | null): boolean => {
  if (left?.id !== right?.id || left?.opacity !== right?.opacity || left?.blendMode !== right?.blendMode) {
    return false;
  }
  const leftFill = isMaskLayer(left) ? left.mask.fill.color : null;
  const rightFill = isMaskLayer(right) ? right.mask.fill.color : null;
  return leftFill === rightFill;
};

export const isLayerEditingDisabled = (layer: CanvasLayerContract | null, editingLocked: boolean): boolean =>
  !layer || editingLocked;

/**
 * The slimmed Photoshop-style region above the layer groups (round 3): exactly two
 * rows — the global canvas denoising-strength slider + a per-selection Opacity row
 * (percent stepper + a mask-fill colour swatch for mask layers). Blend mode and the
 * per-type settings that used to stack here now live in each row's properties
 * popover, matching the legacy layout.
 */
export const LayersPanelHeader = () => {
  const engine = useCanvasEngine();
  const dispatch = useWorkbenchDispatch();
  const layer = useActiveProjectSelector(selectSelectedLayer, isSameSelection);
  const editingLocked = useCanvasDocumentEditingLocked(engine);

  return (
    <Stack gap="0">
      <Box borderBottomWidth={1} px="1.5" py="1">
        <DenoisingStrengthControl />
      </Box>
      <Box borderBottomWidth={1} px="1.5" py="1">
        <Flex align="center" gap="2">
          <BlendModeControl dispatch={dispatch} editingLocked={editingLocked} engine={engine} layer={layer} />
          <OpacityRow dispatch={dispatch} editingLocked={editingLocked} engine={engine} layer={layer} />
        </Flex>
      </Box>
    </Stack>
  );
};

interface BlendModeOption {
  label: string;
  value: CanvasBlendMode;
}

const BlendModeControl = ({
  dispatch,
  editingLocked,
  engine,
  layer,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  editingLocked: boolean;
  engine: CanvasEngine | null;
  layer: CanvasLayerContract | null;
}) => {
  const { t } = useTranslation();
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  const blendMode = layer?.blendMode ?? 'normal';
  const blendCollection = useMemo(
    () =>
      createListCollection<BlendModeOption>({
        items: CANVAS_BLEND_MODES.map((mode) => ({ label: t(`widgets.layers.blendModes.${mode}`), value: mode })),
      }),
    [t]
  );
  const blendValue = useMemo(() => [blendMode], [blendMode]);

  const handleBlendChange = useCallback(
    ({ value }: SelectValueChangeDetails<BlendModeOption>) => {
      const mode = value[0] as CanvasBlendMode | undefined;
      if (!layer || !mode || mode === layer.blendMode) {
        return;
      }
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.actions.blendMode'),
        { id: layer.id, patch: { blendMode: mode }, type: 'updateCanvasLayer' },
        { id: layer.id, patch: { blendMode: layer.blendMode }, type: 'updateCanvasLayer' }
      );
    },
    [dispatch, engine, layer, t]
  );

  return (
    <Field disabled={disabled} flex="1.5" label={t('widgets.layers.actions.blendMode')} orientation="horizontal">
      <Select
        aria-label={t('widgets.layers.actions.blendMode')}
        collection={blendCollection}
        disabled={disabled}
        minW="7rem"
        positioning={SELECT_POSITIONING}
        size="xs"
        value={blendValue}
        valueText={t(`widgets.layers.blendModes.${blendMode}`)}
        w="full"
        onValueChange={handleBlendChange}
      />
    </Field>
  );
};

const OpacityRow = ({
  dispatch,
  editingLocked,
  engine,
  layer,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  editingLocked: boolean;
  engine: CanvasEngine | null;
  layer: CanvasLayerContract | null;
}) => {
  const { t } = useTranslation();
  // The uncommitted opacity edit: captured once per gesture. `before` is the
  // pre-gesture value (the undo target); `latest` tracks the live value because
  // React may not have re-rendered between the live dispatch and the commit
  // trigger (both can fire inside one browser event), so `layer.opacity` from the
  // render closure can be stale at commit time.
  const pendingRef = useRef<{ id: string; before: number; latest: number } | null>(null);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  const opacityPercent = useMemo(() => String(Math.round((layer?.opacity ?? 1) * 100)), [layer?.opacity]);

  // Records ONE history entry spanning the pending gesture (a spinner press,
  // an arrow-key press, or a typed value committed via Enter/blur).
  const commitPending = useCallback(() => {
    const pending = pendingRef.current;
    pendingRef.current = null;
    if (!pending || pending.before === pending.latest) {
      return;
    }
    applyStructural(
      engine,
      dispatch,
      t('widgets.layers.actions.opacity'),
      { id: pending.id, patch: { opacity: pending.latest }, type: 'updateCanvasLayer' },
      { id: pending.id, patch: { opacity: pending.before }, type: 'updateCanvasLayer' }
    );
  }, [dispatch, engine, t]);

  const handleOpacityChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (!layer || !Number.isFinite(valueAsNumber)) {
        return;
      }
      // If a pending edit belongs to a previously selected layer, flush it first
      // so its history entry is never attributed to the new layer.
      if (pendingRef.current && pendingRef.current.id !== layer.id) {
        commitPending();
      }
      const next = clamp01(valueAsNumber / 100);
      if (
        !applyStructuralPreview(engine, dispatch, {
          id: layer.id,
          patch: { opacity: next },
          type: 'updateCanvasLayer',
        })
      ) {
        return;
      }
      if (pendingRef.current === null) {
        pendingRef.current = { before: layer.opacity, id: layer.id, latest: next };
      } else {
        pendingRef.current.latest = next;
      }
    },
    [commitPending, dispatch, engine, layer]
  );

  // Commit per completed interaction: each spinner click (fires on release, so a
  // press-and-hold repeat is one gesture), each arrow/paging key release, Enter,
  // and blur (typed values).
  const handleInputKeyUp = useCallback(
    (event: { key: string }) => {
      if (['ArrowDown', 'ArrowUp', 'End', 'Enter', 'Home', 'PageDown', 'PageUp'].includes(event.key)) {
        commitPending();
      }
    },
    [commitPending]
  );

  // Flush a still-pending edit if the row unmounts mid-gesture (e.g. the panel
  // closes right after a spinner click) so the edit is never lost to history.
  const flushOnUnmountRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (node) {
        return () => commitPending();
      }
      return undefined;
    },
    [commitPending]
  );

  return (
    <Field disabled={disabled} label={t('widgets.layers.actions.opacity')} orientation="horizontal">
      <HStack ref={flushOnUnmountRef} gap="2">
        <NumberInput.Root
          disabled={disabled}
          max={100}
          min={0}
          size="xs"
          step={1}
          value={opacityPercent}
          w="20"
          onValueChange={handleOpacityChange}
        >
          <NumberInput.Control onClick={commitPending} />
          <NumberInput.Input
            aria-label={t('widgets.layers.actions.opacity')}
            onBlur={commitPending}
            onKeyUp={handleInputKeyUp}
          />
        </NumberInput.Root>
        {isMaskLayer(layer) ? (
          <MaskFillSwatch disabled={editingLocked} dispatch={dispatch} engine={engine} layer={layer} />
        ) : null}
      </HStack>
    </Field>
  );
};

/**
 * The selected mask layer's fill-colour swatch (legacy `ActionBarFill`): a colour
 * swatch that opens the picker. Live edits during the drag are un-recorded; the
 * final colour lands as one undoable history entry, mirroring the slider pattern.
 */
const MaskFillSwatch = ({
  dispatch,
  disabled,
  engine,
  layer,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  disabled: boolean;
  engine: CanvasEngine | null;
  layer: MaskLayer;
}) => {
  const { t } = useTranslation();
  const fillBeforeRef = useRef<CanvasMaskFillContract | null>(null);
  const fill = layer.mask.fill;

  const patchFill = useCallback(
    (next: CanvasMaskFillContract, before: CanvasMaskFillContract) => {
      const config =
        layer.type === 'inpaint_mask'
          ? ({ layerType: 'inpaint_mask', mask: { fill: next } } as const)
          : ({ layerType: 'regional_guidance', mask: { fill: next } } as const);
      const inverseConfig =
        layer.type === 'inpaint_mask'
          ? ({ layerType: 'inpaint_mask', mask: { fill: before } } as const)
          : ({ layerType: 'regional_guidance', mask: { fill: before } } as const);
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.maskFill.fill'),
        { config, id: layer.id, type: 'updateCanvasLayerConfig' },
        { config: inverseConfig, id: layer.id, type: 'updateCanvasLayerConfig' }
      );
    },
    [dispatch, engine, layer.id, layer.type, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      const next = { ...fill, color: hex };
      const config =
        layer.type === 'inpaint_mask'
          ? ({ layerType: 'inpaint_mask', mask: { fill: next } } as const)
          : ({ layerType: 'regional_guidance', mask: { fill: next } } as const);
      if (!applyStructuralPreview(engine, dispatch, { config, id: layer.id, type: 'updateCanvasLayerConfig' })) {
        return;
      }
      if (fillBeforeRef.current === null) {
        fillBeforeRef.current = fill;
      }
    },
    [dispatch, engine, fill, layer.id, layer.type]
  );

  const handleColorChangeEnd = useCallback(
    (hex: string) => {
      const before = fillBeforeRef.current ?? fill;
      fillBeforeRef.current = null;
      patchFill({ ...before, color: hex }, before);
    },
    [fill, patchFill]
  );

  return (
    <Box aria-disabled={disabled} inert={disabled} opacity={disabled ? 0.5 : 1}>
      <ColorPicker
        aria-label={t('widgets.layers.maskFill.color')}
        value={fill.color}
        onValueChange={handleColorChange}
        onValueChangeEnd={handleColorChangeEnd}
      />
    </Box>
  );
};

const selectCanvasStrength = (project: Parameters<typeof getProjectWidgetValues>[0]): number =>
  readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas'));

const DenoisingStrengthControl = () => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const projectId = useActiveProjectSelector((project) => project.id);
  const strength = useActiveProjectSelector(selectCanvasStrength);

  const commitStrength = useCallback(
    (value: number) => {
      dispatch({
        type: 'patchWidgetValues',
        values: { [CANVAS_DENOISING_STRENGTH_KEY]: clampCanvasDenoisingStrength(value) },
        widgetId: 'canvas',
      });
    },
    [dispatch]
  );

  const {
    draftValue: draftStrength,
    flushDraftValue,
    setDraftValue: setStrength,
  } = useDebouncedDraftValue({
    delayMs: STRENGTH_DEBOUNCE_MS,
    onCommit: commitStrength,
    resetKey: projectId,
    value: strength,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  const strengthAriaLabel = useMemo(() => [t('widgets.layers.denoisingStrength')], [t]);
  const strengthSliderValue = useMemo(() => [draftStrength], [draftStrength]);
  const strengthNumberValue = useMemo(() => draftStrength.toFixed(2), [draftStrength]);
  const strengthWave = useMemo(() => <DenoisingStrengthWave value={draftStrength} />, [draftStrength]);

  const onSliderChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next !== undefined && Number.isFinite(next)) {
        setStrength(next);
      }
    },
    [setStrength]
  );

  const onNumberChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setStrength(valueAsNumber);
      }
    },
    [setStrength]
  );

  return (
    <Field label={t('widgets.layers.denoisingStrength')} labelEnd={strengthWave} orientation="horizontal">
      <Flex direction="row" gap="2" align="center">
        <Slider
          aria-label={strengthAriaLabel}
          flex="1"
          formatValue={formatStrengthPercent}
          max={MAX_CANVAS_DENOISING_STRENGTH}
          min={MIN_CANVAS_DENOISING_STRENGTH}
          minW="0"
          size="sm"
          step={0.01}
          value={strengthSliderValue}
          withThumbTooltip
          onValueChange={onSliderChange}
          ms="2"
        />
        <NumberInput.Root
          max={MAX_CANVAS_DENOISING_STRENGTH}
          min={MIN_CANVAS_DENOISING_STRENGTH}
          size="xs"
          step={0.05}
          value={strengthNumberValue}
          w="20"
          onValueChange={onNumberChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.layers.denoisingStrength')} />
        </NumberInput.Root>
      </Flex>
    </Field>
  );
};
