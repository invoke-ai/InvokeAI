import type { ProjectGraphState } from '@features/workflow/contracts';

import { serializeWorkflowJson } from '@features/workflow/utility';

/** Export-side helpers shared by the workflow menu items and panels. */

export const downloadWorkflowJson = (document: ProjectGraphState): void => {
  const blob = new Blob([JSON.stringify(serializeWorkflowJson(document), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement('a');

  anchor.href = url;
  anchor.download = `${document.name.trim() || 'workflow'}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
};

export const getWorkflowJsonText = (document: ProjectGraphState): string =>
  JSON.stringify(serializeWorkflowJson(document), null, 2);

export const copyWorkflowJson = (document: ProjectGraphState): Promise<void> =>
  navigator.clipboard.writeText(getWorkflowJsonText(document));
