import { $store } from 'app/store/nanostores/store';
import { RootState } from 'app/store/store';
import { FieldType } from 'features/nodes/types/field';
import { InvocationNodeData } from 'features/nodes/types/invocation';
import { t } from 'i18next';
import { forEach } from 'lodash-es';
import { z } from 'zod';
import {
  WorkflowMigrationError,
  WorkflowVersionError,
} from 'features/nodes/types/error';
import { zSemVer } from 'features/nodes/types/semver';
import { FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING } from 'features/nodes/types/v1/fieldTypeMap';
import { WorkflowV1, zWorkflowV1 } from 'features/nodes/types/v1/workflowV1';
import { WorkflowV2, zWorkflowV2 } from 'features/nodes/types/workflow';

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
 */
const migrateV1toV2 = (workflowToMigrate: WorkflowV1): WorkflowV2 => {
  const invocationTemplates = ($store.get()?.getState() as RootState).nodes
    .nodeTemplates;
  workflowToMigrate.nodes.forEach((node) => {
    if (node.type === 'invocation') {
      // Migrate field types
      forEach(node.data.inputs, (input) => {
        const newFieldType = FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[input.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(
            t('nodes.unknownFieldType', { type: input.type })
          );
        }
        // Cast as the V2 type
        (input.type as unknown as FieldType) = newFieldType;
      });
      forEach(node.data.outputs, (output) => {
        const newFieldType =
          FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[output.type];
        if (!newFieldType) {
          throw new WorkflowMigrationError(
            t('nodes.unknownFieldType', { type: output.type })
          );
        }
        // Cast as the V2 type
        (output.type as unknown as FieldType) = newFieldType;
      });
      // Migrate nodePack
      const invocationTemplate = invocationTemplates[node.data.type];
      const nodePack = invocationTemplate
        ? invocationTemplate.nodePack
        : t('common.unknown');
      // Cast as the V2 type
      (node.data as unknown as InvocationNodeData).nodePack = nodePack;
    }
  });
  (workflowToMigrate.meta.version as WorkflowV2['meta']['version']) = '2.0.0';
  return zWorkflowV2.parse(workflowToMigrate);
};

/**
 * Parses a workflow and migrates it to the latest version if necessary.
 */
export const parseAndMigrateWorkflow = (data: unknown): WorkflowV2 => {
  const workflowVersionResult = zWorkflowMetaVersion.safeParse(data);

  if (!workflowVersionResult.success) {
    console.log(data);
    throw new WorkflowVersionError(t('nodes.unableToGetWorkflowVersion'));
  }

  const { version } = workflowVersionResult.data.meta;

  if (version === '1.0.0') {
    const v1 = zWorkflowV1.parse(data);
    return migrateV1toV2(v1);
  }

  if (version === '2.0.0') {
    return zWorkflowV2.parse(data);
  }

  throw new WorkflowVersionError(
    t('nodes.unrecognizedWorkflowVersion', { version })
  );
};
