import { MenuDivider, MenuGroup, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type {
  CanvasInpaintMaskState,
  CanvasObjectState,
  CanvasRasterLayerState,
} from 'features/controlLayers/store/types';
import {
  buildClosedPathLassoObjects,
  buildClosedPathPolygonObjects,
  isFillableBezierPath,
} from 'features/controlLayers/util/vectorLayerMaterialization';
import { buildVectorTraceObject } from 'features/controlLayers/util/vectorLayerTrace';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowSquareOutBold,
  PiCopyFill,
  PiFrameCornersBold,
  PiPaintBucketBold,
  PiSelectionAllBold,
  PiTrashSimpleBold,
  PiWaveSineBold,
} from 'react-icons/pi';

export const VectorPathMenuItems = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext('vector_layer');
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);
  const isBusy = useCanvasIsBusy();
  const isEditingThisLayer =
    editSession?.entityIdentifier.id === entityIdentifier.id &&
    editSession.entityIdentifier.type === entityIdentifier.type;

  const activePath =
    adapter?.state.type === 'vector_layer'
      ? (adapter.state.paths.find((path) => path.id === editSession?.activePathId) ?? null)
      : null;
  const canMutate = Boolean(
    isEditingThisLayer && !isBusy && activePath && canvasManager.tool.tools.path.getCanMutateEditSession()
  );
  const canMaterializeClosedPath = Boolean(canMutate && activePath && isFillableBezierPath(activePath));

  const onTrace = useCallback(() => {
    if (!activePath || !canMutate || !adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    const color = canvasManager.stateApi.getCurrentColor();
    const { brushWidth, traceTaperEnds, traceTaper } = canvasManager.stateApi.getSettings();
    const object = buildVectorTraceObject(activePath, brushWidth, color, traceTaperEnds, traceTaper);
    if (!object) {
      return;
    }

    canvasManager.stateApi.addRasterLayer({
      isSelected: false,
      overrides: {
        position: { ...adapter.state.position },
        objects: [object] satisfies CanvasObjectState[],
      } satisfies Partial<CanvasRasterLayerState>,
    });
  }, [activePath, adapter, canMutate, canvasManager]);

  const onFill = useCallback(() => {
    if (!activePath || !canMaterializeClosedPath || !adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    const objects = buildClosedPathPolygonObjects([activePath], canvasManager.stateApi.getCurrentColor());
    if (objects.length === 0) {
      return;
    }

    canvasManager.stateApi.addRasterLayer({
      isSelected: false,
      overrides: {
        position: { ...adapter.state.position },
        objects,
      } satisfies Partial<CanvasRasterLayerState>,
    });
  }, [activePath, adapter, canMaterializeClosedPath, canvasManager]);

  const onCreateInpaintMask = useCallback(() => {
    if (!activePath || !canMaterializeClosedPath || !adapter || adapter.state.type !== 'vector_layer') {
      return;
    }

    const objects = buildClosedPathLassoObjects([activePath]);
    if (objects.length === 0) {
      return;
    }

    canvasManager.stateApi.addInpaintMask({
      isSelected: false,
      overrides: {
        position: { ...adapter.state.position },
        objects,
      } satisfies Partial<CanvasInpaintMaskState>,
    });
  }, [activePath, adapter, canMaterializeClosedPath, canvasManager]);

  const onCopy = useCallback(() => {
    canvasManager.tool.tools.path.copyActivePath();
  }, [canvasManager.tool.tools.path]);

  const onDelete = useCallback(() => {
    canvasManager.tool.tools.path.deleteActivePath();
  }, [canvasManager.tool.tools.path]);

  const onExtract = useCallback(() => {
    canvasManager.tool.tools.path.extractActivePath();
  }, [canvasManager.tool.tools.path]);

  const onTransform = useCallback(() => {
    void canvasManager.tool.tools.path.startTransformActivePath();
  }, [canvasManager.tool.tools.path]);

  if (!isEditingThisLayer) {
    return null;
  }

  return (
    <>
      <MenuGroup title={t('controlLayers.vectorEdit.pathTitle')}>
        <MenuItem onClick={onTransform} icon={<PiFrameCornersBold />} isDisabled={!canMutate}>
          {t('controlLayers.vectorEdit.transformPath')}
        </MenuItem>
        <MenuItem onClick={onTrace} icon={<PiWaveSineBold />} isDisabled={!canMutate}>
          {t('controlLayers.vectorEdit.tracePath')}
        </MenuItem>
        <MenuItem onClick={onFill} icon={<PiPaintBucketBold />} isDisabled={!canMaterializeClosedPath}>
          {t('controlLayers.vectorEdit.fillPath')}
        </MenuItem>
        <MenuItem onClick={onCreateInpaintMask} icon={<PiSelectionAllBold />} isDisabled={!canMaterializeClosedPath}>
          {t('controlLayers.vectorEdit.createInpaintMaskFromPath')}
        </MenuItem>
        <MenuItem onClick={onExtract} icon={<PiArrowSquareOutBold />} isDisabled={!canMutate}>
          {t('controlLayers.vectorEdit.extractPath')}
        </MenuItem>
        <MenuItem onClick={onCopy} icon={<PiCopyFill />} isDisabled={!canMutate}>
          {t('controlLayers.vectorEdit.copyPath')}
        </MenuItem>
        <MenuItem onClick={onDelete} icon={<PiTrashSimpleBold />} isDestructive isDisabled={!canMutate}>
          {t('controlLayers.vectorEdit.deletePath')}
        </MenuItem>
      </MenuGroup>
      <MenuDivider />
    </>
  );
});

VectorPathMenuItems.displayName = 'VectorPathMenuItems';
