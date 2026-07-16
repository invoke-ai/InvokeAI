import type { Project } from '@workbench/types';

import { getProjectWidgetValues } from '@workbench/widgetState';

export const getSelectedModelBase = (project: Project): string | null => {
  const model = getProjectWidgetValues(project, 'generate').model;
  return model && typeof model === 'object' && 'base' in model && typeof model.base === 'string' ? model.base : null;
};
