import { Box, HStack, Icon, VisuallyHidden } from '@chakra-ui/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { WorkflowIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetViewProps } from './contracts';

import { WorkflowEditorView } from './editor/WorkflowEditorView';
import { WorkflowLinearPanel } from './linear/WorkflowLinearPanel';
import { useWorkflowProjectSelector } from './WorkflowUiContext';

const WorkflowStatusBarItem = () => {
  const { t } = useTranslation();
  const workflowName = useWorkflowProjectSelector((project) => project.projectGraph.name);
  const isRunning = useWorkflowProjectSelector((project) => project.isWorkflowRunning);

  return (
    <HStack gap="1" maxW="14rem" minW="0" px="2">
      {/* The status-bar chip is named by its content; the name alone would not say which widget it is. */}
      <VisuallyHidden>{t('widgets.labels.workflow')}</VisuallyHidden>
      <Icon as={WorkflowIcon} boxSize="3" color={isRunning ? 'brand.solid' : undefined} flexShrink={0} />
      <MiddleTruncate fontSize="xs" minW="0" text={workflowName || 'Untitled Workflow'} />
      {isRunning ? <Box bg="brand.solid" boxSize="1.5" flexShrink={0} rounded="full" /> : null}
    </HStack>
  );
};

export const WorkflowWidgetView = ({ presentation, region, runtime }: WorkflowWidgetViewProps) => {
  if (region === 'bottom') {
    return presentation === 'expanded' ? <WorkflowEditorView runtime={runtime} /> : <WorkflowStatusBarItem />;
  }

  if (region === 'left') {
    return <WorkflowLinearPanel />;
  }

  return <WorkflowEditorView runtime={runtime} />;
};
