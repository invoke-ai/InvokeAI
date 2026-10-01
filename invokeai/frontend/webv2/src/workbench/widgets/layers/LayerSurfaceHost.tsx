import type { CanvasDocumentContractV3, LayerStackKind } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Dispatch } from 'react';

import { getDocumentLeaves, getDocumentNode } from '@workbench/canvas-engine/api';
import { useCallback, useMemo } from 'react';

import type { LayerRowCommands, LayerSurfaceAnchor } from './layerRowCommands';

import { LayerChildMenu } from './LayerChildMenu';
import { getLayerChildItem, type ProjectedChildRow } from './layerChildRows';
import { CanvasLayerContextMenu, type LayerContextMenuEngine } from './LayerContextMenu';
import { LayerGroupContextMenu, type LayerGroupContextMenuEngine } from './LayerGroupContextMenu';
import { LayerStackMenu } from './LayerStackMenu';

export type LayerSurfaceEngine = LayerContextMenuEngine &
  LayerGroupContextMenuEngine &
  Pick<CanvasEngineHandle, 'document' | 'exports' | 'projectId'>;

/** What the panel currently shows beside a row, addressed by node id rather than owned by the row. */
export type LayerSurfaceRequest =
  | { readonly kind: 'menu'; readonly id: string; readonly anchor: LayerSurfaceAnchor }
  | { readonly kind: 'stack-menu'; readonly stack: LayerStackKind; readonly anchor: LayerSurfaceAnchor }
  | { readonly kind: 'child-menu'; readonly child: ProjectedChildRow; readonly anchor: LayerSurfaceAnchor };

interface LayerSurfaceHostProps {
  commands: LayerRowCommands;
  dispatch: Dispatch<CanvasProjectMutation>;
  document: CanvasDocumentContractV3;
  editingLocked: boolean;
  engine: LayerSurfaceEngine | null;
  surface: LayerSurfaceRequest | null;
  onClose: () => void;
}

/**
 * The one menu and one properties popover the whole panel shares. A row asks for a surface by id
 * and anchor; nothing heavier than a button lives in the row. A surface whose node has gone
 * closes itself.
 */
export const LayerSurfaceHost = ({
  commands,
  dispatch,
  document,
  editingLocked,
  engine,
  surface,
  onClose,
}: LayerSurfaceHostProps) => {
  const node = surface?.kind === 'menu' ? getDocumentNode(document, surface.id) : null;
  const menuTarget = useMemo(
    () =>
      surface?.kind === 'menu' && node && node.type !== 'group'
        ? { layerId: node.id, x: surface.anchor.x, y: surface.anchor.y + surface.anchor.height }
        : null,
    [node, surface]
  );
  const handleMenuClose = useCallback(() => onClose(), [onClose]);
  const liveChild = useMemo(() => {
    if (surface?.kind !== 'child-menu') {
      return null;
    }
    const { child } = surface;
    const item = getLayerChildItem(document, child.layerId, child.itemId);
    if (!item) {
      return null;
    }
    return item.isEnabled === child.isEnabled ? child : { ...child, isEnabled: item.isEnabled };
  }, [document, surface]);
  const moveTargets = useMemo(() => {
    if (surface?.kind !== 'child-menu' || surface.child.kind !== 'reference-image') {
      return [];
    }
    const { child } = surface;
    return getDocumentLeaves(document)
      .filter(
        (leaf) =>
          leaf.type === 'regional_guidance' &&
          leaf.id !== child.layerId &&
          !leaf.referenceImages.some((ref) => ref.id === child.itemId)
      )
      .map((leaf) => ({ id: leaf.id, name: leaf.name }));
  }, [document, surface]);
  if (surface?.kind === 'stack-menu') {
    return (
      <LayerStackMenu
        anchor={surface.anchor}
        document={document}
        editingLocked={editingLocked}
        engine={engine}
        stack={surface.stack}
        onClose={onClose}
      />
    );
  }
  if (surface?.kind === 'child-menu') {
    if (!liveChild) {
      return null;
    }
    return (
      <LayerChildMenu
        anchor={surface.anchor}
        child={liveChild}
        commands={commands}
        editingLocked={editingLocked}
        moveTargets={moveTargets}
        onClose={onClose}
      />
    );
  }
  if (surface && !node) {
    return null;
  }
  return (
    <>
      <CanvasLayerContextMenu dispatch={dispatch} engine={engine} target={menuTarget} onClose={handleMenuClose} />
      {surface?.kind === 'menu' && node?.type === 'group' && engine ? (
        <LayerGroupContextMenu
          key={node.id}
          anchor={surface.anchor}
          editingLocked={editingLocked}
          engine={engine}
          group={node}
          stack={engine.document.model()?.getEntry(node.id)?.stack ?? 'raster'}
          onClose={onClose}
        />
      ) : null}
    </>
  );
};
