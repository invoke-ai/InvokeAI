import { t } from 'i18next';
import { forEach, isString } from 'lodash-es';
import { z } from 'zod';
import { WorkflowVersionError } from '../../types/error';
import { zSemVer } from '../../types/semver';
import { FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING } from '../../types/v1/fieldTypeMap';
import { WorkflowV1, zWorkflowV1 } from '../../types/v1/workflowV1';
import { WorkflowV2, zWorkflowV2 } from '../../types/workflow';

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
  workflowToMigrate.nodes.forEach((node) => {
    if (node.type === 'invocation') {
      forEach(node.data.inputs, (input) => {
        if (!isString(input.type)) {
          return;
        }
        (input.type as unknown) =
          FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[input.type];
      });
      forEach(node.data.outputs, (output) => {
        if (!isString(output.type)) {
          return;
        }
        (output.type as unknown) =
          FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING[output.type];
      });
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
