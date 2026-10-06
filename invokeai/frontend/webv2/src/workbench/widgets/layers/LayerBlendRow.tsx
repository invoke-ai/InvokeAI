import type {
  NumberInput as ChakraNumberInput,
  SelectHighlightChangeDetails,
  SelectOpenChangeDetails,
  SelectValueChangeDetails,
} from '@chakra-ui/react';
import type { CanvasBlendMode, CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { createListCollection, Flex, HStack, NumberInput } from '@chakra-ui/react';
import { Select } from '@platform/ui';
import { getDocumentIndex, isGroupNode } from '@workbench/canvas-engine/api';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { CANVAS_BLEND_MODES } from '@workbench/widgets/layers/layerOps';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type LayerBlendRowEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'layers' | 'projectId'>;

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;
const BLEND_TRIGGER_PROPS = { fontSize: 'md', h: 'control.md', minH: 'control.md' } as const;
const OPACITY_INPUT_PROPS = { fontSize: 'md', h: 'control.md' } as const;

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

/** An open menu's preview: its layer, the mode the menu opened on, and whether a preview has been applied. */
interface BlendPreview {
  readonly id: string;
  readonly original: CanvasBlendMode;
  previewed: boolean;
}

const blendModeOf = (layer: CanvasNodeContract | null): CanvasBlendMode => layer?.blendMode ?? 'normal';

/**
 * Highlighting an option (hover or arrow keys) previews its mode unrecorded. Choosing one records a single step from
 * the mode the menu opened on; closing without a choice or unmounting restores that mode.
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
  const { cancel: cancelPreview, commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const { t } = useTranslation();
  // The ref serves handlers within one event; the state pins what the closed trigger and checked item show.
  const previewRef = useRef<BlendPreview | null>(null);
  const [openedOn, setOpenedOn] = useState<CanvasBlendMode | null>(null);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  // While open the document carries the previewed mode; the control keeps naming the committed one.
  const shownMode = openedOn ?? blendModeOf(layer);
  const blendCollection = useMemo(
    () =>
      createListCollection<BlendModeOption>({
        items: CANVAS_BLEND_MODES.map((mode) => ({ label: t(`widgets.layers.blendModes.${mode}`), value: mode })),
      }),
    [t]
  );
  const blendValue = useMemo(() => [shownMode], [shownMode]);

  const endPreview = useCallback(() => {
    const session = previewRef.current;
    previewRef.current = null;
    if (session?.previewed) {
      cancelPreview({ id: session.id, patch: { blendMode: session.original }, type: 'updateCanvasLayer' });
    }
  }, [cancelPreview]);

  const handleOpenChange = useCallback(
    ({ open }: SelectOpenChangeDetails) => {
      endPreview();
      if (open && layer) {
        previewRef.current = { id: layer.id, original: blendModeOf(layer), previewed: false };
      }
      setOpenedOn(open && layer ? blendModeOf(layer) : null);
    },
    [endPreview, layer]
  );

  // Leaving every option clears the highlight, which shows the original again.
  const handleHighlightChange = useCallback(
    ({ highlightedValue }: SelectHighlightChangeDetails<BlendModeOption>) => {
      const session = previewRef.current;
      if (!session) {
        return;
      }
      const mode = (highlightedValue as CanvasBlendMode | null) ?? session.original;
      if (!session.previewed && mode === session.original) {
        return;
      }
      if (previewStructural({ id: session.id, patch: { blendMode: mode }, type: 'updateCanvasLayer' })) {
        session.previewed = true;
      }
    },
    [previewStructural]
  );

  const handleBlendChange = useCallback(
    ({ value }: SelectValueChangeDetails<BlendModeOption>) => {
      const mode = value[0] as CanvasBlendMode | undefined;
      const session = previewRef.current;
      const id = session?.id ?? layer?.id;
      const original = session ? session.original : blendModeOf(layer);
      if (!id || !mode || mode === original) {
        endPreview();
        return;
      }
      previewRef.current = null;
      commitPrepared(t('widgets.layers.actions.blendMode'), (model) =>
        model.prepare({ before: { blendMode: original }, id, patch: { blendMode: mode }, type: 'patch' })
      );
    },
    [commitPrepared, endPreview, layer, t]
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
        valueText={t(`widgets.layers.blendModes.${shownMode}`)}
        onHighlightChange={handleHighlightChange}
        onOpenChange={handleOpenChange}
        onValueChange={handleBlendChange}
      />
    </Flex>
  );
};

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
  // Capture original opacity once and track latest writes outside render closures so same-event commits record
  // current values.
  const pendingRef = useRef<{ id: string; before: number; latest: number } | null>(null);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  const opacityPercent = useMemo(() => String(Math.round((layer?.opacity ?? 1) * 100)), [layer?.opacity]);

  // Record one history entry per completed opacity gesture.
  const commitPending = useCallback(() => {
    const pending = pendingRef.current;
    pendingRef.current = null;
    if (!pending || pending.before === pending.latest) {
      return;
    }
    commitPrepared(t('widgets.layers.actions.opacity'), (model) =>
      model.prepare({
        before: { opacity: pending.before },
        id: pending.id,
        patch: { opacity: pending.latest },
        type: 'patch',
      })
    );
  }, [commitPrepared, t]);

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
        !previewStructural({
          id: layer.id,
          patch: { opacity: next },
          type: 'updateCanvasLayer',
        })
      ) {
        return;
      }
      if (pendingRef.current === null) {
        pendingRef.current = { before: layer.opacity ?? 1, id: layer.id, latest: next };
      } else {
        pendingRef.current.latest = next;
      }
    },
    [commitPending, previewStructural, layer]
  );

  // Commit on spinner release, arrow/page-key release, Enter, or typed-value blur.
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
    <HStack ref={flushOnUnmountRef} flexShrink={0} gap="2">
      <NumberInput.Root
        disabled={disabled}
        max={100}
        min={0}
        size="lg"
        step={1}
        value={opacityPercent}
        w="16"
        onValueChange={handleOpacityChange}
      >
        <NumberInput.Control onClick={commitPending} />
        <NumberInput.Input
          aria-label={t('widgets.layers.actions.opacity')}
          css={OPACITY_INPUT_PROPS}
          onBlur={commitPending}
          onKeyUp={handleInputKeyUp}
        />
      </NumberInput.Root>
    </HStack>
  );
};
