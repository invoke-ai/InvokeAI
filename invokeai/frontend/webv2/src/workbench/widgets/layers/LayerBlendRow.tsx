import type { SelectHighlightChangeDetails, SelectOpenChangeDetails, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasBlendMode, CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { createListCollection, Flex } from '@chakra-ui/react';
import { Select } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
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
// The controls share a row only when each keeps this width: a narrower scrubber would run its thumb through its
// label and value, so typical panel widths give opacity its own full-width row.
const CONTROL_FLEX = '1 1 11rem';

const formatPercent = (value: number): string => `${value}%`;

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
    <Flex align="center" columnGap="1.5" flexShrink={0} flexWrap="wrap" mx="1.5" rowGap="1">
      {/* Keyed by layer: a selection change closes the menu and ends the outgoing layer's preview. */}
      <BlendModeControl key={layer?.id ?? ''} editingLocked={editingLocked} engine={engine} layer={layer} />
      <OpacityField editingLocked={editingLocked} engine={engine} layer={layer} />
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
    <Flex ref={endPreviewOnUnmount} flex={CONTROL_FLEX} minW="0">
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

/** Where an opacity gesture started and the latest value it previewed; it records one step when it ends. */
interface OpacityGesture {
  readonly id: string;
  readonly before: number;
  latest: number;
}

const OpacityField = ({
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
  // Outside render closures: a drag keeps the handlers it started with.
  const gestureRef = useRef<OpacityGesture | null>(null);
  const disabled = isLayerEditingDisabled(layer, editingLocked);
  const opacityPercent = Math.round((layer?.opacity ?? 1) * 100);

  const settleGesture = useCallback(() => {
    const gesture = gestureRef.current;
    gestureRef.current = null;
    if (!gesture) {
      return;
    }
    if (gesture.before === gesture.latest) {
      cancelPreview({ id: gesture.id, patch: { opacity: gesture.before }, type: 'updateCanvasLayer' });
      return;
    }
    commitPrepared(t('widgets.layers.actions.opacity'), (model) =>
      model.prepare({
        before: { opacity: gesture.before },
        id: gesture.id,
        patch: { opacity: gesture.latest },
        type: 'patch',
      })
    );
  }, [cancelPreview, commitPrepared, t]);

  const handleOpacityChange = useCallback(
    (percent: number) => {
      if (!layer) {
        return;
      }
      // A gesture still open on a previously selected layer lands there before this one starts.
      if (gestureRef.current && gestureRef.current.id !== layer.id) {
        settleGesture();
      }
      const next = percent / 100;
      if (!previewStructural({ id: layer.id, patch: { opacity: next }, type: 'updateCanvasLayer' })) {
        return;
      }
      if (gestureRef.current === null) {
        gestureRef.current = { before: layer.opacity ?? 1, id: layer.id, latest: next };
      } else {
        gestureRef.current.latest = next;
      }
    },
    [layer, previewStructural, settleGesture]
  );

  return (
    <ScrubberField
      defaultValue={100}
      disabled={disabled}
      flex={CONTROL_FLEX}
      formatValue={formatPercent}
      label={t('widgets.layers.actions.opacity')}
      max={100}
      min={0}
      step={1}
      value={opacityPercent}
      onChange={handleOpacityChange}
      onChangeEnd={settleGesture}
    />
  );
};
