import type {
  NumberInput as ChakraNumberInput,
  SelectHighlightChangeDetails,
  SelectOpenChangeDetails,
  SelectValueChangeDetails,
} from '@chakra-ui/react';
import type {
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasLayerPreviewMutation,
  CanvasNodeContract,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { KeyboardEvent } from 'react';

import {
  chakra,
  createListCollection,
  Flex,
  HStack,
  Icon,
  InputGroup,
  NumberInput,
  Popover,
  Portal,
} from '@chakra-ui/react';
import { Select } from '@platform/ui';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Slider } from '@platform/ui/Slider';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { getDocumentIndex, isGroupNode } from '@workbench/canvas-engine/api';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { baselinePatch, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { CANVAS_BLEND_MODES } from '@workbench/widgets/layers/layerOps';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { BlendIcon, ChevronDownIcon, OpacityIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type LayerBlendRowEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'layers' | 'projectId'>;

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;
const BLEND_TRIGGER_PROPS = { fontSize: 'md', h: 'control.md', minH: 'control.md' } as const;
const OPACITY_INPUT_PROPS = { fontSize: 'md', h: 'control.md' } as const;
const OPACITY_ICON_SLOT_PROPS = { color: 'fg.muted', ps: '2' } as const;
const SLIDER_TRIGGER_SLOT_PROPS = { pe: '0.5', pointerEvents: 'auto' } as const;
const SLIDER_POPOVER_POSITIONING = { placement: 'bottom-end' } as const;
// Lucide has no dedicated opacity glyph; two overlapping translucent circles read as opacity, and the Workflow
// editor's node-opacity control uses the same one.
const OPACITY_ICON = <Icon aria-hidden as={OpacityIcon} boxSize="3.5" />;
// Keys that step the slider; their release ends the gesture.
const SLIDER_STEP_KEYS = new Set([
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'End',
  'Home',
  'PageDown',
  'PageUp',
]);
// Keys Zag's number input steps or commits on (Home/End only without Ctrl/Alt/Meta). Any other key, a caret move
// included, leaves a typed value pending.
const FIELD_COMMIT_KEYS = new Set(['ArrowDown', 'ArrowUp', 'Enter']);
const FIELD_BOUND_KEYS = new Set(['End', 'Home']);

const isFieldCommitKey = (event: { altKey: boolean; ctrlKey: boolean; key: string; metaKey: boolean }): boolean =>
  FIELD_COMMIT_KEYS.has(event.key) ||
  (FIELD_BOUND_KEYS.has(event.key) && !event.ctrlKey && !event.altKey && !event.metaKey);

const getPercentValueText = ({ value }: { value: number }): string => `${value}%`;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

// Reference equality is exact: the document index hands back the same node
// object until the node itself changes. A raster-stack GROUP is a valid target
// (opacity/blend on its isolated composite); overlay-stack groups are not.
export const selectBlendTarget = (project: {
  canvas: { document: Pick<CanvasDocumentContractV3, 'stacks' | 'selectedLayerId'> };
}): CanvasNodeContract | null => {
  const document = project.canvas.document;
  if (!document.selectedLayerId) {
    return null;
  }
  const entry = getDocumentIndex(document).byId.get(document.selectedLayerId);
  if (!entry) {
    return null;
  }
  return isGroupNode(entry.node) && entry.stack !== 'raster' ? null : entry.node;
};

export const isLayerEditingDisabled = (layer: CanvasNodeContract | null, editingLocked: boolean): boolean =>
  !layer || editingLocked;

export const LayerBlendRow = ({ engine }: { engine: LayerBlendRowEngine | null }) => {
  const layer = useActiveProjectSelector(selectBlendTarget);
  const editingLocked = useCanvasDocumentEditingLocked(engine);

  return (
    <Flex align="center" flexShrink={0} gap="1.5" mx="1.5">
      {/* Keyed by layer: a selection change closes the menu and ends the outgoing layer's preview. */}
      <BlendModeControl key={layer?.id ?? ''} editingLocked={editingLocked} engine={engine} layer={layer} />
      <OpacityRow editingLocked={editingLocked} engine={engine} layer={layer} />
    </Flex>
  );
};

interface BlendModeOption {
  label: string;
  value: CanvasBlendMode;
}

/** An open menu's preview target: the layer the menu opened for. */
interface BlendPreview {
  readonly id: string;
}

const blendModeOf = (layer: CanvasNodeContract | null): CanvasBlendMode => layer?.blendMode ?? 'normal';

/** The mode `layer`'s open preview replaced, while one is open on it. */
const pinnedBlendMode = (
  baseline: CanvasLayerPreviewMutation | null,
  layer: CanvasNodeContract | null
): CanvasBlendMode | undefined =>
  layer && baseline?.id === layer.id ? baselinePatch(baseline, { blendMode: 'normal' })?.blendMode : undefined;

/**
 * Highlighting an option (hover or arrow keys) previews its mode unrecorded. Choosing one records a single step from
 * the mode the preview started on; closing without a choice or unmounting restores that mode. An undo or an edit
 * from elsewhere while the menu is open ends the preview in the engine, and a choice then records from the live
 * mode.
 */
const BlendModeControl = ({
  editingLocked,
  engine,
  layer,
}: {
  editingLocked: boolean;
  engine: LayerBlendRowEngine | null;
  layer: CanvasNodeContract | null;
}) => {
  const {
    baseline: previewBaseline,
    cancel: cancelPreview,
    commit: commitPrepared,
    preview: previewStructural,
  } = useStructuralPreview(engine);
  const { t } = useTranslation();
  // The ref serves handlers within one event: the layer the menu opened for and the mode it opened on.
  const previewRef = useRef<BlendPreview | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  // While the menu is open the document carries the previewed mode and the control keeps naming the one the preview
  // replaced; every preview re-renders through the layer. Once the engine ended the preview (an undo landed under
  // the menu) the document's mode is the committed one again, and a choice records from it.
  const shownMode = (isOpen ? pinnedBlendMode(previewBaseline(), layer) : undefined) ?? blendModeOf(layer);
  const blendCollection = useMemo(
    () =>
      createListCollection<BlendModeOption>({
        items: CANVAS_BLEND_MODES.map((mode) => ({ label: t(`widgets.layers.blendModes.${mode}`), value: mode })),
      }),
    [t]
  );
  const blendValue = useMemo(() => [shownMode], [shownMode]);
  const blendValueText = useMemo(
    () => (
      <HStack as="span" display="inline-flex" gap="1.5" maxW="full" minW="0" verticalAlign="middle">
        <Icon aria-hidden as={BlendIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
        <chakra.span minW="0" truncate>
          {t(`widgets.layers.blendModes.${shownMode}`)}
        </chakra.span>
      </HStack>
    ),
    [shownMode, t]
  );

  const endPreview = useCallback(() => {
    previewRef.current = null;
    cancelPreview();
  }, [cancelPreview]);

  const handleOpenChange = useCallback(
    ({ open }: SelectOpenChangeDetails) => {
      endPreview();
      if (open && layer) {
        previewRef.current = { id: layer.id };
      }
      setIsOpen(open && layer !== null);
    },
    [endPreview, layer]
  );

  // Leaving every option clears the highlight, which shows the mode the open preview replaced again. Once the engine
  // ended the preview, that is the document's own mode, so there is nothing to preview.
  const handleHighlightChange = useCallback(
    ({ highlightedValue }: SelectHighlightChangeDetails<BlendModeOption>) => {
      const session = previewRef.current;
      if (!session) {
        return;
      }
      const baseline = previewBaseline();
      const original = pinnedBlendMode(baseline, layer) ?? blendModeOf(layer);
      const mode = (highlightedValue as CanvasBlendMode | null) ?? original;
      if (mode === original && baseline === null) {
        return;
      }
      previewStructural({ id: session.id, patch: { blendMode: mode }, type: 'updateCanvasLayer' });
    },
    [layer, previewBaseline, previewStructural]
  );

  const handleBlendChange = useCallback(
    ({ value }: SelectValueChangeDetails<BlendModeOption>) => {
      const mode = value[0] as CanvasBlendMode | undefined;
      const id = previewRef.current?.id ?? layer?.id;
      previewRef.current = null;
      if (!id || !mode) {
        cancelPreview();
        return;
      }
      // Choosing the mode the preview started on prepares nothing, which drops the previews.
      const patch = { blendMode: mode };
      commitPrepared(t('widgets.layers.actions.blendMode'), (model, baseline) =>
        model.prepare({ before: baselinePatch(baseline, patch), id, patch, type: 'patch' })
      );
    },
    [cancelPreview, commitPrepared, layer, t]
  );

  const endPreviewOnUnmount = useCallback(
    (node: HTMLDivElement | null) => (node ? endPreview : undefined),
    [endPreview]
  );

  return (
    <Flex ref={endPreviewOnUnmount} flex="1" minW="0">
      <Select
        aria-label={t('widgets.layers.actions.blendMode')}
        collection={blendCollection}
        disabled={disabled}
        flex="1"
        itemsMaxH="16rem"
        minW="0"
        positioning={SELECT_POSITIONING}
        triggerProps={BLEND_TRIGGER_PROPS}
        value={blendValue}
        valueText={blendValueText}
        onHighlightChange={handleHighlightChange}
        onOpenChange={handleOpenChange}
        onValueChange={handleBlendChange}
      />
    </Flex>
  );
};

/**
 * Typed entry, arrow keys, and a slider in a popover at the field's end. Every change previews on the layer; each
 * completed gesture records one history entry: a typed value on Enter or blur, a held key on its release, a slider
 * drag when the pointer lifts. The field shows the slider's live value, so the thumb carries no tooltip to cover the
 * trigger.
 */
const OpacityRow = ({
  editingLocked,
  engine,
  layer,
}: {
  editingLocked: boolean;
  engine: LayerBlendRowEngine | null;
  layer: CanvasNodeContract | null;
}) => {
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const { t } = useTranslation();
  const sliderTriggerIds = useTooltipTriggerIds();
  // Track latest writes outside render closures so same-event commits record current values.
  const pendingRef = useRef<{ id: string; latest: number } | null>(null);
  // A slider key settles on its release, as in the field, so a held key records once.
  const sliderKeyHeldRef = useRef(false);
  // The closing popover's slider can keep tracking a drag through its exit; closing already recorded the gesture, so
  // nothing the slider reports after it may preview or become pending.
  const sliderOpenRef = useRef(false);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  const percent = Math.round((layer?.opacity ?? 1) * 100);
  const opacityPercent = String(percent);
  const sliderValue = useMemo(() => [percent], [percent]);
  const sliderLabel = useMemo(() => [t('widgets.layers.actions.opacity')], [t]);

  // Record one history entry per completed opacity gesture.
  const commitPending = useCallback(() => {
    const pending = pendingRef.current;
    pendingRef.current = null;
    if (!pending) {
      return;
    }
    const patch = { opacity: pending.latest };
    commitPrepared(t('widgets.layers.actions.opacity'), (model, baseline) =>
      model.prepare({ before: baselinePatch(baseline, patch), id: pending.id, patch, type: 'patch' })
    );
  }, [commitPrepared, t]);

  const previewOpacity = useCallback(
    (nextPercent: number) => {
      if (!layer || !Number.isFinite(nextPercent)) {
        return;
      }
      // If a pending edit belongs to a previously selected layer, flush it first
      // so its history entry is never attributed to the new layer.
      if (pendingRef.current && pendingRef.current.id !== layer.id) {
        commitPending();
      }
      const next = clamp01(nextPercent / 100);
      if (!previewStructural({ id: layer.id, patch: { opacity: next }, type: 'updateCanvasLayer' })) {
        return;
      }
      pendingRef.current = { id: layer.id, latest: next };
    },
    [commitPending, previewStructural, layer]
  );

  const handleInputChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => previewOpacity(valueAsNumber),
    [previewOpacity]
  );

  // Commit on a step key's release, Enter, or typed-value blur.
  const handleInputKeyUp = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (isFieldCommitKey(event)) {
        commitPending();
      }
    },
    [commitPending]
  );

  const handleSliderChange = useCallback(
    ({ value: [next] }: { value: number[] }) => {
      if (next !== undefined && sliderOpenRef.current) {
        previewOpacity(next);
      }
    },
    [previewOpacity]
  );

  // Zag ends a value change on every key press, from the thumb's own keydown; the capture-phase key flag lets a held
  // key settle on its release instead. A drag ends once, when the pointer lifts.
  const handleSliderChangeEnd = useCallback(() => {
    if (!sliderKeyHeldRef.current) {
      commitPending();
    }
  }, [commitPending]);

  const handleSliderKeyDown = useCallback((event: { key: string }) => {
    if (SLIDER_STEP_KEYS.has(event.key)) {
      sliderKeyHeldRef.current = true;
    }
  }, []);

  const handleSliderKeyUp = useCallback(
    (event: { key: string }) => {
      if (SLIDER_STEP_KEYS.has(event.key)) {
        sliderKeyHeldRef.current = false;
        commitPending();
      }
    },
    [commitPending]
  );

  const handlePopoverOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      sliderOpenRef.current = open;
      sliderKeyHeldRef.current = false;
      commitPending();
    },
    [commitPending]
  );

  // Flush a still-pending edit if the row unmounts mid-gesture (e.g. the panel
  // closes mid-drag or right after an arrow key) so the edit is never lost to history.
  const flushOnUnmountRef = useCallback(
    (node: HTMLDivElement | null) => (node ? commitPending : undefined),
    [commitPending]
  );

  const sliderTrigger = useMemo(
    () => (
      <Popover.Root
        ids={sliderTriggerIds}
        lazyMount
        positioning={SLIDER_POPOVER_POSITIONING}
        unmountOnExit
        onOpenChange={handlePopoverOpenChange}
      >
        <Tooltip content={t('widgets.layers.actions.opacitySlider')} ids={sliderTriggerIds}>
          <Popover.Trigger asChild>
            <IconButton
              aria-label={t('widgets.layers.actions.opacitySlider')}
              color="fg.muted"
              disabled={disabled}
              minW="5"
              size="sm"
              variant="ghost"
              w="5"
            >
              <Icon as={ChevronDownIcon} boxSize="3" />
            </IconButton>
          </Popover.Trigger>
        </Tooltip>
        <Portal>
          <Popover.Positioner>
            <PopoverContent aria-label={t('widgets.layers.actions.opacity')} w="auto">
              <Popover.Body px="3" py="2.5">
                <Slider
                  aria-label={sliderLabel}
                  disabled={disabled}
                  getAriaValueText={getPercentValueText}
                  max={100}
                  min={0}
                  step={1}
                  value={sliderValue}
                  w="36"
                  withThumbTooltip={false}
                  onKeyDownCapture={handleSliderKeyDown}
                  onKeyUp={handleSliderKeyUp}
                  onValueChange={handleSliderChange}
                  onValueChangeEnd={handleSliderChangeEnd}
                />
              </Popover.Body>
            </PopoverContent>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
    ),
    [
      disabled,
      handlePopoverOpenChange,
      handleSliderChange,
      handleSliderChangeEnd,
      handleSliderKeyDown,
      handleSliderKeyUp,
      sliderLabel,
      sliderTriggerIds,
      sliderValue,
      t,
    ]
  );

  return (
    <HStack ref={flushOnUnmountRef} flexShrink={0} gap="2">
      <NumberInput.Root
        disabled={disabled}
        max={100}
        min={0}
        size="lg"
        step={1}
        value={opacityPercent}
        w="24"
        onValueChange={handleInputChange}
      >
        <InputGroup
          endElement={sliderTrigger}
          endElementProps={SLIDER_TRIGGER_SLOT_PROPS}
          startElement={OPACITY_ICON}
          startElementProps={OPACITY_ICON_SLOT_PROPS}
        >
          <NumberInput.Input
            aria-label={t('widgets.layers.actions.opacity')}
            css={OPACITY_INPUT_PROPS}
            fontVariantNumeric="tabular-nums"
            onBlur={commitPending}
            onKeyUp={handleInputKeyUp}
          />
        </InputGroup>
      </NumberInput.Root>
    </HStack>
  );
};
