import type { CanvasDocumentContractV3, CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Stack, Switch, Text } from '@chakra-ui/react';
import { getDocumentNode } from '@workbench/canvas-engine/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { AdjustmentSettings } from '@workbench/widgets/layers/AdjustmentSettings';
import { ControlLayerSettings } from '@workbench/widgets/layers/ControlLayerSettings';
import { InpaintMaskSettings } from '@workbench/widgets/layers/InpaintMaskSettings';
import {
  LAYER_REGION_ITEM_ID,
  adjustmentChildKind,
  childRowNameKey,
  MASK_DENOISE_ITEM_ID,
  MASK_NOISE_ITEM_ID,
  type LayerChildRowKind,
} from '@workbench/widgets/layers/layerChildRows';
import { useLayerChildSelection } from '@workbench/widgets/layers/layerChildSelection';
import { LayerRegionSettings } from '@workbench/widgets/layers/LayerRegionSettings';
import { MaskModifierSettings } from '@workbench/widgets/layers/MaskModifierSettings';
import { RasterLayerFilterSection } from '@workbench/widgets/layers/RasterLayerFilterSection';
import { ReferenceImageSettings } from '@workbench/widgets/layers/ReferenceImageSettings';
import { RegionalGuidanceSettings } from '@workbench/widgets/layers/RegionalGuidanceSettings';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { GroupSelectedNotice } from './GroupSelectedNotice';
import { PropertiesSection } from './PropertiesSection';

type LayerSectionEngine = Pick<
  CanvasEngineHandle,
  'document' | 'exports' | 'interaction' | 'layers' | 'projectId' | 'tools'
>;

// Reference equality is exact: the document index hands back the same node
// object until the node itself changes, and the section renders the whole
// node, so a narrower comparison would serve stale views of it.
const selectSelectedNode = (project: {
  canvas: { document: Pick<CanvasDocumentContractV3, 'stacks' | 'selectedLayerId'> };
}): CanvasNodeContract | null => getDocumentNode(project.canvas.document, project.canvas.document.selectedLayerId);

/**
 * Render selected-layer editors here; blend/opacity remain in LayerBlendRow and all edits retain their document
 * seams.
 */
export const LayerSection = ({ disabled }: { disabled: boolean }) => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const node = useActiveProjectSelector(selectSelectedNode);
  const layer = node && node.type !== 'group' ? node : null;
  const documentRevision = useActiveProjectSelector((project) => project.canvas.documentRevision);
  const projectId = useActiveProjectId();
  const childSelection = useLayerChildSelection();
  const child = resolveChildEditor(node, childSelection?.projectId === projectId ? childSelection : null, t);

  return (
    <PropertiesSection
      disabled={disabled}
      subtitle={child ? child.subtitle : (node?.name ?? t('widgets.transform.noSelection'))}
      title={t('widgets.properties.sections.layer')}
    >
      {child && node ? (
        <ChildEditor key={`${node.id}:${child.itemId}`} child={child} engine={engine} node={node} />
      ) : layer ? (
        <LayerTypeSettings documentRevision={documentRevision} engine={engine} layer={layer} />
      ) : (
        <GroupSelectedNotice />
      )}
    </PropertiesSection>
  );
};

interface ChildEditorTarget {
  readonly kind: LayerChildRowKind;
  readonly itemId: string;
  readonly subtitle: string;
}

/** What the sub-selection edits, or `null` when it does not belong to `layer`. */
const resolveChildEditor = (
  layer: CanvasNodeContract | null,
  selection: { layerId: string; itemId: string } | null,
  t: (key: string) => string
): ChildEditorTarget | null => {
  if (!layer || !selection || selection.layerId !== layer.id) {
    return null;
  }
  if (layer.type === 'regional_guidance') {
    const index = layer.referenceImages.findIndex((ref) => ref.id === selection.itemId);
    return index >= 0
      ? {
          itemId: selection.itemId,
          kind: 'reference-image',
          subtitle: `${t('widgets.layers.regionalGuidance.referenceImage')} ${index + 1}`,
        }
      : null;
  }
  if (layer.type === 'inpaint_mask') {
    if (selection.itemId === MASK_NOISE_ITEM_ID && layer.noise) {
      return { itemId: selection.itemId, kind: 'mask-noise', subtitle: t('widgets.layers.modifiers.noise') };
    }
    if (selection.itemId === MASK_DENOISE_ITEM_ID && layer.denoise) {
      return { itemId: selection.itemId, kind: 'mask-denoise', subtitle: t('widgets.layers.modifiers.denoise') };
    }
  }
  if (layer.type === 'raster' && selection.itemId === LAYER_REGION_ITEM_ID && layer.inpaint) {
    return {
      itemId: selection.itemId,
      kind: 'layer-region',
      subtitle: layer.inpaint.name ?? t(childRowNameKey('layer-region')),
    };
  }
  if (layer.type === 'raster' || layer.type === 'group') {
    const entry = layer.adjustments?.find((candidate) => candidate.id === selection.itemId);
    if (entry) {
      const kind = adjustmentChildKind(entry.type);
      return { itemId: selection.itemId, kind, subtitle: entry.name ?? t(childRowNameKey(kind)) };
    }
  }
  return null;
};

const ChildEditor = ({
  child,
  engine,
  node,
}: {
  child: ChildEditorTarget;
  engine: LayerSectionEngine | null;
  node: CanvasNodeContract;
}) => {
  if (child.kind === 'reference-image' && node.type === 'regional_guidance') {
    return <ReferenceImageSettings engine={engine} layer={node} refId={child.itemId} />;
  }
  if ((child.kind === 'mask-noise' || child.kind === 'mask-denoise') && node.type === 'inpaint_mask') {
    return <MaskModifierSettings engine={engine} kind={child.kind} layer={node} />;
  }
  if (child.kind === 'layer-region' && node.type === 'raster') {
    return <LayerRegionSettings engine={engine} layer={node} />;
  }
  if (child.kind.startsWith('adjustment-') && (node.type === 'raster' || node.type === 'group')) {
    return <AdjustmentSettings engine={engine} entryId={child.itemId} layer={node} />;
  }
  return null;
};

/** Dispatches to the correct per-type settings block for the layer. */
const LayerTypeSettings = ({
  documentRevision,
  engine,
  layer,
}: {
  documentRevision: number;
  engine: LayerSectionEngine | null;
  layer: CanvasLayerContract;
}) => {
  switch (layer.type) {
    case 'inpaint_mask':
      return <InpaintMaskSettings key={layer.id} engine={engine} layer={layer} />;
    case 'regional_guidance':
      return <RegionalGuidanceSettings key={layer.id} engine={engine} layer={layer} />;
    case 'control':
      return <ControlLayerSettings key={layer.id} engine={engine} layer={layer} onOperationStarted={noop} />;
    case 'raster':
      return (
        <RasterLayerSettings
          key={`${engine?.projectId ?? 'none'}-${layer.id}-${documentRevision}`}
          engine={engine}
          layer={layer}
        />
      );
  }
};

const noop = (): void => undefined;

/** Raster-layer properties: transparency lock + non-destructive adjustments. */
const RasterLayerSettings = ({
  engine,
  layer,
}: {
  engine: LayerSectionEngine | null;
  layer: Extract<CanvasLayerContract, { type: 'raster' }>;
}) => {
  const commitPrepared = usePreparedCommit(engine);
  const { t } = useTranslation();
  const isLocked = layer.isTransparencyLocked === true;

  const handleTransparencyLock = useCallback(
    (details: { checked: boolean }) => {
      commitPrepared(t('widgets.layers.adjustments.transparencyLock'), (model) =>
        model.prepare({
          before: { isTransparencyLocked: isLocked, layerType: 'raster' },
          config: { isTransparencyLocked: details.checked, layerType: 'raster' },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, isLocked, layer.id, t]
  );

  return (
    <Stack gap="2">
      <Switch.Root checked={isLocked} onCheckedChange={handleTransparencyLock}>
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="md">{t('widgets.layers.adjustments.transparencyLock')}</Text>
        </Switch.Label>
      </Switch.Root>
      <RasterLayerFilterSection engine={engine} layer={layer} onOperationStarted={noop} />
    </Stack>
  );
};
