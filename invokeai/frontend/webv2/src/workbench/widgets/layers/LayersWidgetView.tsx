import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Flex, Icon, Stack, Text } from '@chakra-ui/react';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { getDocumentIndex } from '@workbench/canvas-engine/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { setLayerPanelFilter, useLayerPanelState } from '@workbench/layerPanelState';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { LayersIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerSurfaceAnchor } from './layerRowCommands';
import type { LayerColorPaneLayout, LayerEditorPaneLayout, LayerTreeTabId } from './panes/editorPaneLayout';

import { LayerBlendRow } from './LayerBlendRow';
import { reconcileLayerChildSelection } from './layerChildSelection';
import { LAYER_PANEL_DEGRADE_THRESHOLD } from './layerPanelRows';
import { useCurrentLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { anchorFromPoint } from './layerRowCommands';
import { AddLayerContextMenu, LayersHeaderActions } from './LayersHeaderActions';
import { LayersPanelFooter } from './LayersPanelFooter';
import { LayersTree } from './LayersTree';
import { buildLayerStackRows } from './layerTreeRows';
import {
  areColorPaneLayoutsEqual,
  areLayerEditorPaneLayoutsEqual,
  readColorPaneLayout,
  readLayerEditorPaneLayout,
  readLayerTreeTab,
} from './panes/editorPaneLayout';
import { HistoryPane } from './panes/HistoryPane';
import { LayerColorPane, LayerEditorPanes } from './panes/LayerEditorPanes';
import { useCanvasHistoryHotkeys } from './useCanvasHistoryHotkeys';
import { useLayerSelectionCommands } from './useLayerSelectionCommands';

const TREE_TABS_ID_BASE = 'layer-tree';
const TREE_TABS: ReadonlyArray<{ id: LayerTreeTabId; labelKey: string }> = [
  { id: 'layers', labelKey: 'widgets.labels.layers' },
  { id: 'history', labelKey: 'widgets.labels.history' },
];

export const LayersWidgetView = ({ runtime }: WidgetViewProps) => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const dispatch = useCanvasProjectMutationDispatch();
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const document = useActiveProjectSelector((project) => project.canvas.document);
  const { selectedLayerId } = document;

  const panel = useLayerPanelState(projectId, selectedLayerId);
  useEffect(() => reconcileLayerChildSelection(projectId, document), [document, projectId]);
  const expandedGroupIds = useMemo(() => new Set(panel.expandedGroupIds), [panel.expandedGroupIds]);
  const stacks = useMemo(
    () => buildLayerStackRows(document.stacks, expandedGroupIds, panel.filter),
    [document.stacks, expandedGroupIds, panel.filter]
  );
  const nodeCount = getDocumentIndex(document).nodes.length;
  const degraded = nodeCount > LAYER_PANEL_DEGRADE_THRESHOLD;

  const selectionCommands = useLayerSelectionCommands(engine, projectId, panel.selectedIds, editingLocked);
  useCanvasHistoryHotkeys(runtime, engine);
  const handleFilter = useCallback(
    (filter: string) => setLayerPanelFilter(projectId, selectedLayerId, filter),
    [projectId, selectedLayerId]
  );
  const paneLayout = useActiveProjectSelector(
    (project) => readLayerEditorPaneLayout(project.widgetInstances[runtime.instanceId]?.state.values ?? {}),
    areLayerEditorPaneLayoutsEqual
  );
  const handlePaneLayout = useCallback(
    (next: LayerEditorPaneLayout) => runtime.state.patch({ editorPanes: next }),
    [runtime.state]
  );
  const colorPaneLayout = useActiveProjectSelector(
    (project) => readColorPaneLayout(project.widgetInstances[runtime.instanceId]?.state.values ?? {}),
    areColorPaneLayoutsEqual
  );
  const handleColorPaneLayout = useCallback(
    (next: LayerColorPaneLayout) => runtime.state.patch({ colorPane: next }),
    [runtime.state]
  );
  const treeTab = useActiveProjectSelector((project) =>
    readLayerTreeTab(project.widgetInstances[runtime.instanceId]?.state.values ?? {})
  );
  const handleTreeTab = useCallback((tab: LayerTreeTabId) => runtime.state.patch({ treeTab: tab }), [runtime.state]);
  const treeTabs = useMemo(() => TREE_TABS.map(({ id, labelKey }) => ({ id, label: t(labelKey) })), [t]);
  const addLayerButton = useMemo(() => <LayersHeaderActions />, []);
  const propertiesRequest = useCurrentLayerPropertiesRequest();
  useEffect(() => {
    if (propertiesRequest && treeTab === 'history') {
      runtime.state.patch({ treeTab: 'layers' });
    }
  }, [propertiesRequest, runtime.state, treeTab]);
  const revealProperties = useCallback(
    (layerId: string) => {
      dispatch({ id: layerId, type: 'setCanvasSelectedLayer' });
      runtime.state.patch({ editorPanes: { ...paneLayout, activePane: 'properties', isCollapsed: false } });
    },
    [dispatch, paneLayout, runtime.state]
  );
  const [addMenuAnchor, setAddMenuAnchor] = useState<LayerSurfaceAnchor | null>(null);
  const handleEmptyAreaContextMenu = useCallback((event: MouseEvent<HTMLElement>) => {
    if (event.defaultPrevented || (event.target as HTMLElement).closest('[role="treeitem"]')) {
      return;
    }
    event.preventDefault();
    setAddMenuAnchor(anchorFromPoint(event.clientX, event.clientY));
  }, []);
  const closeAddMenu = useCallback(() => setAddMenuAnchor(null), []);

  return (
    <Stack gap="0" h="full" minH="0">
      <LayerColorPane layout={colorPaneLayout} onLayoutChange={handleColorPaneLayout} />
      <SegmentTabs
        activeId={treeTab}
        ariaLabel={t('widgets.layers.treeTabs')}
        idBase={TREE_TABS_ID_BASE}
        tabs={treeTabs}
        trailing={addLayerButton}
        onSelect={handleTreeTab}
      />
      <Flex
        aria-labelledby={segmentTabsTabId(TREE_TABS_ID_BASE, treeTab)}
        direction="column"
        flex="1"
        gap="0.5"
        id={segmentTabsPanelId(TREE_TABS_ID_BASE)}
        minH="13rem"
        overflow="hidden"
        role="tabpanel"
      >
        {treeTab === 'history' ? (
          <HistoryPane />
        ) : (
          <>
            <LayerBlendRow engine={engine} />
            {nodeCount === 0 ? (
              <Flex
                align="center"
                borderColor="border.subtle"
                borderStyle="dashed"
                borderWidth="1px"
                color="fg.subtle"
                direction="column"
                flex="1"
                gap="2"
                justify="center"
                minH="8rem"
                mx="2"
                px="4"
                rounded="md"
                onContextMenu={handleEmptyAreaContextMenu}
              >
                <Icon as={LayersIcon} boxSize="6" />
                <Text fontSize="xs" textAlign="center">
                  {t('widgets.layers.empty')}
                </Text>
              </Flex>
            ) : (
              <Flex direction="column" flex="1" minH="8rem" onContextMenu={handleEmptyAreaContextMenu}>
                <LayersTree
                  degraded={degraded}
                  dispatch={dispatch}
                  document={document}
                  editingLocked={editingLocked}
                  engine={engine}
                  panel={panel}
                  projectId={projectId}
                  stacks={stacks}
                  onRevealProperties={revealProperties}
                />
              </Flex>
            )}
            <LayersPanelFooter
              commands={selectionCommands}
              degraded={degraded}
              filter={panel.filter}
              onFilterChange={handleFilter}
            />
            {addMenuAnchor ? <AddLayerContextMenu anchor={addMenuAnchor} onClose={closeAddMenu} /> : null}
          </>
        )}
      </Flex>
      <LayerEditorPanes layout={paneLayout} onLayoutChange={handlePaneLayout} />
    </Stack>
  );
};
