import { $store } from 'app/store/nanostores/store';
import { WorkflowMigrationError, WorkflowVersionError } from 'features/nodes/types/error';
import type { FieldType } from 'features/nodes/types/field';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { zSemVer } from 'features/nodes/types/semver';
import { FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING } from 'features/nodes/types/v1/fieldTypeMap';
import type { WorkflowV1 } from 'features/nodes/types/v1/workflowV1';
import { zWorkflowV1 } from 'features/nodes/types/v1/workflowV1';
import type { WorkflowV2 } from 'features/nodes/types/v2/workflow';
import { zWorkflowV2 } from 'features/nodes/types/v2/workflow';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import { t } from 'i18next';
import { forEach } from 'lodash-es';
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
  const invocationTemplates = $store.get()?.getState().nodes.templates;

  if (!invocationTemplates) {
    throw new Error(t('app.storeNotInitialized'));
  }

  workflowToMigrate.nodes.forEach((node) => {
    if (node.type === 'invocation') {
      // Migrate field types
      forEach(node.data.inputs, (input) => {
        const newFieldType = FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[input.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(t('nodes.unknownFieldType', { type: input.type }));
        }
        (input.type as unknown as FieldType) = newFieldType;
      });
      forEach(node.data.outputs, (output) => {
        const newFieldType = FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[output.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(t('nodes.unknownFieldType', { type: output.type }));
        }
        (output.type as unknown as FieldType) = newFieldType;
      });
      // Add node pack
      const invocationTemplate = invocationTemplates[node.data.type];
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

/**
 * Parses a workflow and migrates it to the latest version if necessary.
 */
export const parseAndMigrateWorkflow = (data: unknown): WorkflowV3 => {
  const workflowVersionResult = zWorkflowMetaVersion.safeParse(data);

  if (!workflowVersionResult.success) {
    throw new WorkflowVersionError(t('nodes.unableToGetWorkflowVersion'));
  }

  let workflow = data as WorkflowV1 | WorkflowV2 | WorkflowV3;

  if (workflow.meta.version === '1.0.0') {
    const v1 = zWorkflowV1.parse(workflow);
    workflow = migrateV1toV2(v1);
  }

  if (workflow.meta.version === '2.0.0') {
    const v2 = zWorkflowV2.parse(workflow);
    workflow = migrateV2toV3(v2);
  }

  return workflow as WorkflowV3;
};
