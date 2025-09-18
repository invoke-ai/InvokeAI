import { deepClone } from 'common/util/deepClone';
import { forEach, get } from 'es-toolkit/compat';
import { $templates } from 'features/nodes/store/nodesSlice';
import { WorkflowMigrationError, WorkflowVersionError } from 'features/nodes/types/error';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { zSemVer } from 'features/nodes/types/semver';
import { FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING } from 'features/nodes/types/v1/fieldTypeMap';
import type { WorkflowV1 } from 'features/nodes/types/v1/workflowV1';
import { zWorkflowV1 } from 'features/nodes/types/v1/workflowV1';
import type { StatelessFieldType } from 'features/nodes/types/v2/field';
import type { WorkflowV2 } from 'features/nodes/types/v2/workflow';
import { zWorkflowV2 } from 'features/nodes/types/v2/workflow';
import type { WorkflowOutputField, WorkflowV3, WorkflowV4 } from 'features/nodes/types/workflow';
import { zWorkflowV3, zWorkflowV4 } from 'features/nodes/types/workflow';
import { t } from 'i18next';
import { z } from 'zod';

/**
 * Helper schema to extract the version from a workflow.
 *
 * All properties except for the version are ignored in this schema.
 */
const zWorkflowMetaVersion = z.object({
  meta: z.object({ version: zSemVer }),
});

/**
 * Migrates a workflow from V1 to V2.
 *
 * Changes include:
 * - Field types are now structured
 * - Invocation node pack is now saved in the node data
 * - Workflow schema version bumped to 2.0.0
 */
const migrateV1toV2 = (workflowToMigrate: WorkflowV1): WorkflowV2 => {
  const templates = $templates.get();

  workflowToMigrate.nodes.forEach((node) => {
    if (node.type === 'invocation') {
      // Migrate field types
      forEach(node.data.inputs, (input) => {
        const newFieldType = FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[input.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(t('nodes.unknownFieldType', { type: input.type }));
        }
        (input.type as unknown as StatelessFieldType) = newFieldType;
      });
      forEach(node.data.outputs, (output) => {
        const newFieldType = FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[output.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(t('nodes.unknownFieldType', { type: output.type }));
        }
        (output.type as unknown as StatelessFieldType) = newFieldType;
      });
      // Add node pack
      const invocationTemplate = templates[node.data.type];
      const nodePack = invocationTemplate ? invocationTemplate.nodePack : t('common.unknown');

      (node.data as unknown as InvocationNodeData).nodePack = nodePack;
      // Fallback to 1.0.0 if not specified - this matches the behavior of the backend
      node.data.version ||= '1.0.0';
    }
  });
  // Bump version
  (workflowToMigrate as unknown as WorkflowV2).meta.version = '2.0.0';
  // Add category - should always be 'user', 'default' workflows are only created by the backend
  (workflowToMigrate as unknown as WorkflowV2).meta.category = 'user';
  // Parsing strips out any extra properties not in the latest version
  return zWorkflowV2.parse(workflowToMigrate);
};

const migrateV2toV3 = (workflowToMigrate: WorkflowV2): WorkflowV3 => {
  // Bump version
  (workflowToMigrate as unknown as WorkflowV3).meta.version = '3.0.0';
  // Parsing strips out any extra properties not in the latest version
  return zWorkflowV3.parse(workflowToMigrate);
};

const normalizeOutputField = (field: unknown): WorkflowOutputField | null => {
  if (!field || typeof field !== 'object') {
    return null;
  }
  const maybeField = field as Record<string, unknown>;
  const nodeId =
    typeof maybeField.nodeId === 'string'
      ? maybeField.nodeId
      : typeof maybeField.node_id === 'string'
        ? maybeField.node_id
        : null;
  const fieldName =
    typeof maybeField.fieldName === 'string'
      ? maybeField.fieldName
      : typeof maybeField.field_name === 'string'
        ? maybeField.field_name
        : null;
  if (!nodeId || !fieldName) {
    return null;
  }
  const userLabel =
    typeof maybeField.userLabel === 'string'
      ? maybeField.userLabel
      : typeof maybeField.userLabel === 'object'
        ? null
        : typeof maybeField.user_label === 'string'
          ? maybeField.user_label
          : null;

  return {
    kind: 'output',
    nodeId,
    fieldName,
    userLabel,
    node_id: nodeId,
    field_name: fieldName,
    user_label: userLabel,
  } satisfies WorkflowOutputField;
};

const migrateV3toV4 = (workflowToMigrate: WorkflowV3 | Record<string, unknown>): WorkflowV4 => {
  const rawOutputs = Array.isArray((workflowToMigrate as Record<string, unknown>).output_fields)
    ? ((workflowToMigrate as Record<string, unknown>).output_fields as unknown[])
    : [];

  const normalizedOutputs = rawOutputs
    .map(normalizeOutputField)
    .filter((field): field is WorkflowOutputField => field !== null)
    .slice(0, 1);

  const migrated = {
    ...(workflowToMigrate as Record<string, unknown>),
    output_fields: normalizedOutputs,
  } as WorkflowV4;

  migrated.meta.version = '4.0.0';

  return zWorkflowV4.parse(migrated);
};

/**
 * Parses a workflow and migrates it to the latest version if necessary.
 *
 * This function will return a new workflow object, so the original workflow is not modified.
 */
export const parseAndMigrateWorkflow = (data: unknown): WorkflowV4 => {
  const workflowVersionResult = zWorkflowMetaVersion.safeParse(data);

  if (!workflowVersionResult.success) {
    throw new WorkflowVersionError(t('nodes.unableToGetWorkflowVersion'));
  }

  let workflow = deepClone(data);

  if (get(workflow, 'meta.version') === '1.0.0') {
    const v1 = zWorkflowV1.parse(workflow);
    workflow = migrateV1toV2(v1);
  }

  if (get(workflow, 'meta.version') === '2.0.0') {
    const v2 = zWorkflowV2.parse(workflow);
    workflow = migrateV2toV3(v2);
  }

  if (get(workflow, 'meta.version') === '3.0.0') {
    const v3 = zWorkflowV3.parse(workflow);
    workflow = migrateV3toV4(v3);
  }

  // We should now have a V4 workflow
  const migratedWorkflow = zWorkflowV4.parse(workflow);

  return migratedWorkflow;
};
