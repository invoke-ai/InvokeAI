import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { useState } from 'react';
import { GitBranchIcon, TargetIcon, MoreHorizontalIcon } from 'lucide-react';

import { useWorkbench } from '../WorkbenchContext';
import type { GraphBearingSurfaceContract, GraphContract, WidgetId } from '../types';
import { GraphPreviewDialog } from './GraphPreviewDialog';
import { IconButton } from './ui/Button';

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
            <MoreHorizontalIcon />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content minW="11rem">
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
                  <Icon as={TargetIcon} boxSize="3.5" />
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
                  <Icon as={GitBranchIcon} boxSize="3.5" />
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
