import { Flex, Icon, Stack, Text } from '@chakra-ui/react';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCanvasEngine } from '@workbench/widgets/canvas/useCanvasEngine';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { publishLayerPanelSelection, readLayerPanelSelection } from '@workbench/workbenchStore';
import { LayersIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerGroupKey, LayerSelectionModifiers } from './layerGroups';

import { groupLayers, reconcileLayerPanelSelection, selectLayerInPanel } from './layerGroups';
import { LayerGroupSection } from './LayerGroupSection';
import { LayerMultiSelectionActions } from './LayerMultiSelectionActions';
import { isLayerPropertiesGroupRequested, useCurrentLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { LayersPanelHeader } from './LayersPanelHeader';

/**
 * The layers panel: a fixed Photoshop-style header region (selected layer's
 * opacity + blend mode, global denoising strength) above the layer list, which
 * is split into type groups (inpaint masks / regional guidance / control /
 * raster — legacy display order). Each group is a within-group drag-to-reorder
 * list mapped onto the single global z-ordered `layers` array.
 */
export const LayersWidgetView = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const dispatch = useCanvasProjectMutationDispatch();
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const propertiesRequest = useCurrentLayerPropertiesRequest();
  const { layers, selectedLayerId } = useActiveProjectSelector(
    (project) => ({
      layers: project.canvas.document.layers,
      selectedLayerId: project.canvas.document.selectedLayerId,
    }),
    (left, right) => left.layers === right.layers && left.selectedLayerId === right.selectedLayerId
  );

  const groups = useMemo(() => groupLayers(layers), [layers]);
  // Collapse is transient panel UI state (not part of the canvas document / undo
  // history): a set of collapsed group keys, defaulting to expanded.
  const [collapsedGroups, setCollapsedGroups] = useState<Partial<Record<LayerGroupKey, boolean>>>({});
  const allLayerIds = useMemo(() => groups.flatMap((group) => group.layers.map((layer) => layer.id)), [groups]);
  const visibleLayerIds = groups.flatMap((group) =>
    collapsedGroups[group.key] === true && !isLayerPropertiesGroupRequested(propertiesRequest, group.layers)
      ? []
      : group.layers.map((layer) => layer.id)
  );
  const [storedPanelSelection, setPanelSelection] = useState(() => {
    const shared = readLayerPanelSelection(projectId, selectedLayerId);
    return { ...shared, anchorId: shared.primaryId };
  });
  const selection = useMemo(() => {
    const shared = readLayerPanelSelection(projectId, selectedLayerId);
    const source =
      storedPanelSelection.projectId === projectId && storedPanelSelection.primaryId === selectedLayerId
        ? storedPanelSelection
        : { ...shared, anchorId: shared.primaryId };
    return reconcileLayerPanelSelection(source, projectId, allLayerIds, selectedLayerId);
  }, [allLayerIds, projectId, selectedLayerId, storedPanelSelection]);
  if (selection !== storedPanelSelection) {
    setPanelSelection(selection);
  }
  const selectedIds = selection.selectedIds;

  const handleSelectLayer = useCallback(
    (layerId: string, modifiers: LayerSelectionModifiers) => {
      const next = selectLayerInPanel(selection, layerId, modifiers.range ? visibleLayerIds : allLayerIds, modifiers);
      setPanelSelection(next);
      publishLayerPanelSelection(next);
      if (next.primaryId !== selectedLayerId) {
        dispatch({ id: next.primaryId, type: 'setCanvasSelectedLayer' });
      }
    },
    [allLayerIds, dispatch, selectedLayerId, selection, visibleLayerIds]
  );

  const handleToggleCollapse = useCallback((groupKey: LayerGroupKey) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupKey]: !prev[groupKey] }));
  }, []);

  return (
    <Stack h="full">
      <LayersPanelHeader />
      {selectedIds.length > 1 ? (
        <LayerMultiSelectionActions
          dispatch={dispatch}
          editingLocked={editingLocked}
          engine={engine}
          layers={layers}
          projectId={projectId}
          selectedIds={selectedIds}
          selectedLayerId={selectedLayerId}
        />
      ) : null}
      {groups.length === 0 ? (
        <Flex
          align="center"
          borderColor="border.subtle"
          borderStyle="dashed"
          borderWidth="1px"
          color="fg.subtle"
          direction="column"
          gap="2"
          justify="center"
          minH="8rem"
          p="4"
          rounded="md"
        >
          <Icon as={LayersIcon} boxSize="6" />
          <Text fontSize="2xs" textAlign="center">
            {t('widgets.layers.empty')}
          </Text>
        </Flex>
      ) : (
        <Stack gap="3">
          {groups.map((group) => (
            <LayerGroupSection
              key={group.key}
              dispatch={dispatch}
              engine={engine}
              groupKey={group.key}
              groupLayers={group.layers}
              isCollapsed={
                collapsedGroups[group.key] === true && !isLayerPropertiesGroupRequested(propertiesRequest, group.layers)
              }
              layers={layers}
              onSelectLayer={handleSelectLayer}
              onToggleCollapse={handleToggleCollapse}
              selectedIds={selectedIds}
              selectedLayerId={selectedLayerId}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
