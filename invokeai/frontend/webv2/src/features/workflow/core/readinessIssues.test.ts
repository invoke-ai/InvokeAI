import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import { createForLoopValidationReason } from './forLoops';
import {
  createWorkflowIssue,
  formatWorkflowReadinessReason,
  localizeWorkflowReadinessReason,
  WORKFLOW_READINESS_MESSAGES,
  type WorkflowReadinessIssueKind,
} from './readinessIssues';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const instance = i18n.createInstance();
const t = (key: string, values?: Record<string, unknown>) => instance.t(key, values);

beforeAll(async () => {
  await instance.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation: englishCatalog } },
  });
});

const SAMPLE_VALUES = {
  field: 'Image',
  group: 'Group 1',
  max: 10_000,
  node: 'Resize',
  sessions: 10_002,
  sizes: 'a: 2, b: 3',
  source: 'Batch',
  type: 'custom_node',
};

describe('workflow readiness issues', () => {
  it('words every kind the same in English and in the locale', () => {
    for (const kind of Object.keys(WORKFLOW_READINESS_MESSAGES) as WorkflowReadinessIssueKind[]) {
      const reason = createWorkflowIssue(kind, SAMPLE_VALUES);

      expect(localizeWorkflowReadinessReason(reason, t), kind).toBe(formatWorkflowReadinessReason(reason));
      expect(reason.workflowIssue.message, kind).not.toMatch(/\{\{/);
    }
  });

  it('formats counts for English text and keeps the node and field it names', () => {
    expect(createWorkflowIssue('batchCap', { max: 10_000, sessions: 10_002 })).toEqual({
      workflowIssue: {
        message: 'This batch would queue 10,002 sessions; the queue accepts at most 10,000.',
        messageKey: 'nodes.readiness.batchCap',
        messageValues: { max: 10_000, sessions: 10_002 },
      },
    });
    expect(
      createWorkflowIssue('missingInput', { field: 'Image', node: 'Resize' }, { fieldName: 'image', nodeId: 'n1' })
        .workflowIssue
    ).toMatchObject({ fieldName: 'image', nodeId: 'n1' });
  });

  it('passes other sources’ plain reasons through and words for-loop reasons from the locale', () => {
    expect(localizeWorkflowReadinessReason('Canvas generation frame must have a positive area.', t)).toBe(
      'Canvas generation frame must have a positive area.'
    );
    expect(localizeWorkflowReadinessReason(createForLoopValidationReason('nodes.forReturnOwnership'), t)).toBe(
      'For loop validation failed: Each ForReturn must belong to exactly one For.'
    );
  });
});
