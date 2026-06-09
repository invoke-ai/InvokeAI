import { Icon, IconButton, Menu, Portal, Text } from '@chakra-ui/react';
import { useState } from 'react';
import { PiDotsThreeBold, PiGraphBold, PiTargetBold } from 'react-icons/pi';

import { useWorkbench } from '../WorkbenchContext';
import type { GraphBearingSurfaceContract, GraphContract, WidgetId } from '../types';
import { GraphPreviewDialog } from './GraphPreviewDialog';

interface GraphSurfaceActionsProps {
  surface: GraphBearingSurfaceContract;
}

const getPreviewGraph = (
  widgetGraphs: Partial<Record<WidgetId, GraphContract>>,
  surface: GraphBearingSurfaceContract
) => widgetGraphs[surface.widgetId] ?? null;

export const GraphSurfaceActions = ({ surface }: GraphSurfaceActionsProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const isActiveSource = activeProject.invocation.sourceId === surface.sourceId;
  const previewGraph = getPreviewGraph(activeProject.widgetGraphs, surface);

  return (
    <>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label={`${surface.label} graph actions`} color="fg.muted" size="2xs" variant="ghost">
            <PiDotsThreeBold />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content
              bg="bg.surfaceRaised"
              borderWidth="1px"
              borderColor="border.emphasis"
              color="fg.default"
              minW="11rem"
              rounded="lg"
              shadow="lg"
            >
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  Graph
                </Menu.ItemGroupLabel>
                <Menu.Item
                  value="set-source"
                  disabled={isActiveSource || !surface.canSetSource}
                  _disabled={{ opacity: 0.4 }}
                  onClick={() => dispatch({ sourceId: surface.sourceId, type: 'setInvocationSource' })}
                >
                  <Icon as={PiTargetBold} boxSize="3.5" />
                  <Menu.ItemText>Set Source</Menu.ItemText>
                  {isActiveSource ? (
                    <Text color="fg.subtle" fontSize="2xs" ms="auto">
                      Active
                    </Text>
                  ) : null}
                </Menu.Item>
                <Menu.Item
                  value="view-graph"
                  disabled={!surface.canPreviewGraph}
                  onClick={() => setIsPreviewOpen(true)}
                >
                  <Icon as={PiGraphBold} boxSize="3.5" />
                  <Menu.ItemText>View Graph</Menu.ItemText>
                </Menu.Item>
              </Menu.ItemGroup>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <GraphPreviewDialog
        graph={previewGraph}
        graphId={surface.graphId}
        isOpen={isPreviewOpen}
        sourceId={surface.sourceId}
        title={surface.label}
        onOpenChange={setIsPreviewOpen}
      />
    </>
  );
};
