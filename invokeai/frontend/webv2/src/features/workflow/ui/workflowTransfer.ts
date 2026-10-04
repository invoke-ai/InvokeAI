import type { ProjectGraphState } from '@features/workflow/contracts';

import { serializeWorkflowJson } from '@features/workflow/utility';
import { downloadText } from '@platform/browser/downloadBlob';

/** Export-side helpers shared by the workflow menu items and panels. */

export const downloadWorkflowJson = (document: ProjectGraphState): void => {
  downloadText(
    JSON.stringify(serializeWorkflowJson(document), null, 2),
    `${document.name.trim() || 'workflow'}.json`,
    'application/json'
  );
};

export const getWorkflowJsonText = (document: ProjectGraphState): string =>
  JSON.stringify(serializeWorkflowJson(document), null, 2);

export const copyWorkflowJson = (document: ProjectGraphState): Promise<void> =>
  navigator.clipboard.writeText(getWorkflowJsonText(document));
