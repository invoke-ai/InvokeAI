import { localizeForLoopValidationReason, type ForLoopValidationReason } from './forLoops';

/**
 * Why a workflow cannot run, kept structured so surfaces word it in the active language and lead to the node (and
 * input) that needs the fix. Each kind's English text below is the same text the locale holds under
 * `nodes.readiness.<kind>`; a test holds the two together.
 */
export const WORKFLOW_READINESS_MESSAGES = {
  batchCap: 'This batch would queue {{sessions}} sessions; the queue accepts at most {{max}}.',
  batchEmpty: 'Batch node "{{node}}" has an empty collection.',
  batchGroupMixedSizes: 'Batch group "{{group}}" mixes collection sizes ({{sizes}}).',
  batchNotConnected: 'Batch node "{{node}}" is not connected to any node.',
  batchOutputUnsupported: 'Batch node "{{node}}" can only batch its image output; width and height are not supported.',
  batchSourceNotGenerator: 'Batch node "{{node}}" can only receive its collection from a generator.',
  batchTargetClaimed: '"{{field}}" on "{{node}}" receives values from more than one batch node.',
  batchTargetUnsupported: '"{{source}}" cannot supply batch values to "{{field}}" on "{{node}}".',
  batchTooLarge: 'Batch node "{{node}}" has more than {{max}} items.',
  cycle: 'The workflow contains a cycle.',
  imageGeneratorNoBoard: 'Image generator "{{node}}" has no board selected.',
  invalidInput: '"{{node}}" has invalid input "{{field}}".',
  missingConnection: '"{{node}}" is missing a connection for "{{field}}".',
  missingInput: '"{{node}}" is missing required input "{{field}}".',
  noNodes: 'The workflow has no nodes. Add nodes in the workflow editor.',
  savedWorkflowLoading: '"{{node}}" is still loading its saved workflow\'s inputs.',
  savedWorkflowRequired: '"{{node}}" requires a saved workflow.',
  savedWorkflowUnavailable: 'The saved workflow "{{node}}" calls is unavailable or incompatible.',
  templatesFailed: 'Node definitions failed to load from the backend.',
  templatesLoading: 'Node definitions are still loading.',
  unknownNodeType: 'Unknown node type "{{type}}".',
} as const;

export type WorkflowReadinessIssueKind = keyof typeof WORKFLOW_READINESS_MESSAGES;

export interface WorkflowReadinessIssue {
  /** English, for logs and thrown errors. */
  message: string;
  /** The same text in the locale. */
  messageKey: string;
  messageValues?: Readonly<Record<string, number | string>>;
  /** The node the issue is about, when one: surfaces select it to show where the fix goes. */
  nodeId?: string;
  /** The node's input the issue is about, when known. */
  fieldName?: string;
}

export interface WorkflowIssueReason {
  workflowIssue: WorkflowReadinessIssue;
}

export type WorkflowReadinessReason = ForLoopValidationReason | WorkflowIssueReason;

const formatEnglishValue = (value: number | string): string =>
  typeof value === 'number' ? value.toLocaleString('en-US') : value;

export const createWorkflowIssue = (
  kind: WorkflowReadinessIssueKind,
  values: Readonly<Record<string, number | string>> = {},
  target: { fieldName?: string; nodeId?: string } = {}
): WorkflowIssueReason => ({
  workflowIssue: {
    message: WORKFLOW_READINESS_MESSAGES[kind].replace(/\{\{(\w+)\}\}/g, (match, name: string) =>
      Object.hasOwn(values, name) ? formatEnglishValue(values[name] as number | string) : match
    ),
    messageKey: `nodes.readiness.${kind}`,
    ...(Object.keys(values).length > 0 ? { messageValues: values } : {}),
    ...(target.nodeId === undefined ? {} : { nodeId: target.nodeId }),
    ...(target.fieldName === undefined ? {} : { fieldName: target.fieldName }),
  },
});

export const isWorkflowIssueReason = (reason: unknown): reason is WorkflowIssueReason =>
  typeof reason === 'object' && reason !== null && 'workflowIssue' in reason;

/** Words a readiness reason in the active language; plain strings from other sources pass through as they are. */
export const localizeWorkflowReadinessReason = (
  reason: string | WorkflowReadinessReason,
  translate: (key: string, values?: Record<string, unknown>) => string
): string =>
  isWorkflowIssueReason(reason)
    ? translate(reason.workflowIssue.messageKey, { ...reason.workflowIssue.messageValues })
    : localizeForLoopValidationReason(reason, translate);

/** The English text of a readiness reason, for errors and logs. */
export const formatWorkflowReadinessReason = (reason: string | WorkflowReadinessReason): string =>
  typeof reason === 'string'
    ? reason
    : isWorkflowIssueReason(reason)
      ? reason.workflowIssue.message
      : `For loop validation failed: ${reason.key}.`;
