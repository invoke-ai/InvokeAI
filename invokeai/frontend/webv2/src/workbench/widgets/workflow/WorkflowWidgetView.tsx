import { Box, HStack, Icon, Text } from '@chakra-ui/react';
import { WorkflowIcon } from 'lucide-react';

import { useWorkbench } from '../../WorkbenchContext';
import type { WidgetViewProps } from '../../types';
import { WorkflowEditorView } from './editor/WorkflowEditorView';
import { WorkflowLinearPanel } from './linear/WorkflowLinearPanel';

/**
 * The workflow widget's surfaces, per its manifest: the center region is the
 * node editor over the project graph; the left region is the Linear UI panel
 * bound to the same document; the bottom region is a status-bar entry whose
 * expanded panel is the same node editor — handy while another widget (e.g.
 * the canvas and its staging area) holds the center.
 */

const WorkflowStatusBarItem = () => {
  const { activeProject } = useWorkbench();
  const isRunning = activeProject.queue.items.some(
    (item) => item.snapshot.sourceId === 'project-graph' && (item.status === 'running' || item.status === 'pending')
  );

  return (
    <HStack gap="1" maxW="14rem" minW="0">
      <Icon as={WorkflowIcon} boxSize="3" color={isRunning ? 'brand.solid' : undefined} flexShrink={0} />
      <Text fontSize="2xs" minW="0" truncate>
        {activeProject.projectGraph.name || 'Untitled Workflow'}
      </Text>
      {isRunning ? <Box bg="brand.solid" boxSize="1.5" flexShrink={0} rounded="full" /> : null}
    </HStack>
  );
};

export const WorkflowWidgetView = ({ presentation, region }: WidgetViewProps) => {
  if (region === 'bottom') {
    return presentation === 'expanded' ? <WorkflowEditorView /> : <WorkflowStatusBarItem />;
  }

  if (region === 'left') {
    return <WorkflowLinearPanel />;
  }

  return <WorkflowEditorView />;
};
