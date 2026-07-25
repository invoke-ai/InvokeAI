import { useModelsSelector } from '@features/models';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/** The main model's config, read from the generate widget values (drives adapter support). */
export const useSelectedMainModel = () => {
  const modelKey = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate');
    const model = values?.model;
    return model && typeof model === 'object' && 'key' in model ? String((model as { key: unknown }).key) : null;
  });
  const models = useModelsSelector((snapshot) => snapshot.models);
  return useMemo(() => models.find((model) => model.key === modelKey) ?? null, [models, modelKey]);
};
