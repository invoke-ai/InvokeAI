import type { WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState as NodeExecutionState } from '@features/workflow/ui/contracts';

import { Stack, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import type { InvocationNodeTemplateView } from './flowAdapters';

import {
  getExecutionStatusText,
  getFieldIssueMessage,
  getNodeExecutionError,
  type InvocationNodeFieldIssue,
} from './nodeStatusText';

/** The header panels' bodies, loaded on a node's first panel open: most nodes never open one. */

/** The status panel: the run, what blocks the next one, and the version, each when it applies. */
export const NodeStatusPanel = ({
  execution,
  issues,
  node,
  updateText,
}: {
  execution: NodeExecutionState | null;
  issues: readonly InvocationNodeFieldIssue[];
  node: WorkflowInvocationNode;
  updateText: string | null;
}) => {
  const { t } = useTranslation();
  const error = execution?.status === 'failed' ? getNodeExecutionError(node, execution.error, t) : null;

  return (
    <Stack gap="1" maxW="20rem">
      {execution ? (
        <Stack gap="0.5">
          <Text fontWeight="600">{getExecutionStatusText(t, execution)}</Text>
          {execution.status === 'running' && execution.progressMessage ? (
            <Text color="fg.muted">{execution.progressMessage}</Text>
          ) : null}
          {error ? <Text overflowWrap="anywhere">{error}</Text> : null}
        </Stack>
      ) : null}
      {issues.length > 0 ? (
        <Stack as="ul" gap="0.5" listStyleType="none">
          {issues.map((fieldIssue) => (
            <Text key={fieldIssue.template.name} as="li">
              {getFieldIssueMessage(t, fieldIssue)}
            </Text>
          ))}
        </Stack>
      ) : null}
      {updateText ? <Text>{updateText}</Text> : null}
    </Stack>
  );
};

/** Legacy's info panel: what the node is, where it comes from, and the author's notes. */
export const NodeInfoPanel = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => {
  const { t } = useTranslation();
  const title = node.data.label ? `${node.data.label} (${template.title})` : template.title;
  const nodePack = node.data.nodePack || template.nodePack;

  return (
    <Stack gap="1" maxW="20rem">
      <Text fontWeight="600">{title}</Text>
      <Text color="fg.muted">{t('nodes.nodeType', { type: template.type })}</Text>
      <Text color="fg.muted">{t('nodes.nodePackLabel', { name: nodePack })}</Text>
      <Text color="fg.muted">{t('nodes.nodeVersion', { version: node.data.version })}</Text>
      {template.description ? <Text fontStyle="italic">{template.description}</Text> : null}
      {node.data.notes ? <Text whiteSpace="pre-wrap">{node.data.notes}</Text> : null}
    </Stack>
  );
};
