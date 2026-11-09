import type { Project } from '@workbench/projectContracts';

import { getProjectWidgetValues } from '@workbench/widgetState';

export const getSelectedModelBase = (project: Project): string | null => {
  const model = getProjectWidgetValues(project, 'generate').model;
  return model && typeof model === 'object' && 'base' in model && typeof model.base === 'string' ? model.base : null;
};

export const getSelectedCfgScale = (project: Pick<Project, 'widgetInstances'>): number | null => {
  const cfgScale = getProjectWidgetValues(project, 'generate').cfgScale;
  return typeof cfgScale === 'number' && Number.isFinite(cfgScale) ? cfgScale : null;
};
