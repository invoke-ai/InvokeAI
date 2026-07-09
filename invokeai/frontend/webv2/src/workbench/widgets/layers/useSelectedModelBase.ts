import { useModelsSelector } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/**
 * The main model's base (`sd-1` / `sdxl` / `flux` / …), read from the generate
 * widget values. Drives which control-adapter kinds and reference-image kinds a
 * region can consume. Shared by the control/regional settings and the add-layer
 * flow so a freshly created "regional reference image" mints the base-correct kind.
 */
export const useSelectedModelBase = (): string | null => {
  const modelKey = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate');
    const model = values?.model;
    return model && typeof model === 'object' && 'key' in model ? String((model as { key: unknown }).key) : null;
  });
  const models = useModelsSelector((snapshot) => snapshot.models);
  return useMemo(() => models.find((model) => model.key === modelKey)?.base ?? null, [models, modelKey]);
};
