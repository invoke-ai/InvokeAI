import type { FieldInputTemplate, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState as NodeExecutionState } from '@features/workflow/ui/contracts';
import type { TFunction } from 'i18next';

import { isModelFieldType, type WorkflowFieldIssue } from '@features/workflow/utility';

/** A node's status wording, shared by its header (labels) and its status panel, which loads on first open. */

/** A field that keeps its node from running, named as the node shows it. */
export interface InvocationNodeFieldIssue {
  issue: WorkflowFieldIssue;
  label: string;
  /** The rows the issue marks: its own, or both HiDiffusion ratios for their order. */
  markedFieldNames: readonly string[];
  template: FieldInputTemplate;
}

export const getNodeExecutionError = (
  node: WorkflowInvocationNode,
  error: string | null | undefined,
  translate: (key: string, options: { error: string }) => string
): string | null | undefined => {
  if (!error || node.data.type !== 'call_saved_workflow') {
    return error;
  }

  return translate('nodes.childWorkflowError', { error });
};

const getProgressPercent = (execution: NodeExecutionState): number | null =>
  execution.progress === null ? null : Math.round(execution.progress * 100);

/** What a field needs, in words: a model to pick, a value, a connection, or what its value breaks. */
export const getFieldIssueMessage = (t: TFunction, { issue, label, template }: InvocationNodeFieldIssue): string =>
  issue.kind === 'missingValue'
    ? t(isModelFieldType(template.type) ? 'nodes.status.pickModel' : 'nodes.status.fieldRequired', { field: label })
    : issue.kind === 'missingConnection'
      ? t('nodes.status.needsConnection', { field: label })
      : t('nodes.status.fieldInvalid', { field: label, reason: issue.reason });

export const getExecutionStatusText = (t: TFunction, execution: NodeExecutionState): string => {
  if (execution.status === 'running') {
    const percent = getProgressPercent(execution);

    return percent === null ? t('nodes.status.running') : t('nodes.status.runningProgress', { percent });
  }

  return execution.status === 'completed' ? t('nodes.executionCompleted') : t('nodes.executionFailed');
};
