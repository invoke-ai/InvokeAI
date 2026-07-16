import { Flex, Icon, Stack, Text } from '@chakra-ui/react';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useCanvasEngine } from '@workbench/widgets/canvas/useCanvasEngine';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { LayersIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerGroupKey } from './layerGroups';

import { groupLayers } from './layerGroups';
import { LayerGroupSection } from './LayerGroupSection';
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
  const dispatch = useCanvasProjectMutationDispatch();
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
  const handleToggleCollapse = useCallback((groupKey: LayerGroupKey) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupKey]: !prev[groupKey] }));
  }, []);

  return (
    <Stack gap="3" h="full">
      <LayersPanelHeader />
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
              onToggleCollapse={handleToggleCollapse}
              selectedLayerId={selectedLayerId}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
