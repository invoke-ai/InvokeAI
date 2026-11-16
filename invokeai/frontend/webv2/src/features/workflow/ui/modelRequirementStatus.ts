import type { ModelConfig, StarterModel } from '@features/models';
import type { ModelRequirementStatus } from '@features/workflow/core/modelRequirements';
import type { WorkflowPreviewModel } from '@features/workflow/core/workflowPreview';
import type { ElementType } from 'react';

import {
  ensureModelsLoaded,
  ensureStartersLoaded,
  useActiveInstallSources,
  useModelsSelector,
  useStartersSelector,
} from '@features/models';
import { resolveWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CheckIcon, DownloadIcon, TriangleAlertIcon } from 'lucide-react';
import { useMemo } from 'react';

/**
 * Where a workflow's model requirements stand against this server's catalog, for every view that lists them: the
 * library's rail and badges, and the preview's graph and Inputs view.
 */

const EMPTY_STARTERS: readonly StarterModel[] = [];
const NO_NODES: ReadonlySet<string> = new Set();

const selectInstalledModels = (snapshot: { models: ModelConfig[] }): readonly ModelConfig[] => snapshot.models;
const selectStarterModels = (snapshot: {
  response: { starter_models: StarterModel[] } | null;
}): readonly StarterModel[] => snapshot.response?.starter_models ?? EMPTY_STARTERS;

interface StatusPresentation {
  color: string;
  icon: ElementType | null;
  labelKey: string;
}

/** How each requirement state reads wherever a requirement is listed: the rail, and the preview's Inputs view. */
export const REQUIREMENT_STATUS_PRESENTATION: Record<ModelRequirementStatus, StatusPresentation> = {
  // Amber, per the mock, is the "you can fix this" signal and nothing else.
  installable: { color: 'fg.warning', icon: DownloadIcon, labelKey: 'workflowLibrary.requirementInstallable' },
  installed: { color: 'fg.muted', icon: CheckIcon, labelKey: 'workflowLibrary.requirementInstalled' },
  installing: { color: 'fg.muted', icon: null, labelKey: 'workflowLibrary.requirementInstalling' },
  unresolvable: { color: 'fg.muted', icon: TriangleAlertIcon, labelKey: 'workflowLibrary.requirementMissing' },
};

export interface ModelRequirementDeps {
  installedModels: readonly ModelConfig[];
  starterModels: readonly StarterModel[];
  activeInstallSources: ReadonlySet<string>;
}

/**
 * The catalog as far as it is loaded, without loading any of it: for the preview, which shows requirements only where
 * the surface around it (the library's rail) has loaded what they need.
 */
export const useLoadedModelRequirementDeps = (): ModelRequirementDeps => {
  const installedModels = useModelsSelector(selectInstalledModels);
  const starterModels = useStartersSelector(selectStarterModels);
  const activeInstallSources = useActiveInstallSources();

  return useMemo(
    () => ({ activeInstallSources, installedModels, starterModels }),
    [activeInstallSources, installedModels, starterModels]
  );
};

/** The catalog for the library's rail and badges, loading the installed and starter models as they mount. */
export const useModelRequirementDeps = (): ModelRequirementDeps => {
  useMountEffect(() => {
    void ensureModelsLoaded();
    ensureStartersLoaded();
  });

  return useLoadedModelRequirementDeps();
};

/**
 * The preview's nodes that need a model, as the rail counts one, from the catalog as loaded: a model a node names, or the kind an empty field asks
 * for, that is not installed and can be (not one installing now, nor one before the catalog has said either way).
 */
export const useNodesNeedingModel = (model: WorkflowPreviewModel | null): ReadonlySet<string> => {
  const deps = useLoadedModelRequirementDeps();

  return useMemo(() => {
    if (!model) {
      return NO_NODES;
    }

    const ids = new Set<string>();

    for (const node of model.nodes.values()) {
      if (
        node.modelRequirements.length > 0 &&
        resolveWorkflowModelRequirements(node.modelRequirements, deps).some(
          (requirement) => requirement.status === 'installable'
        )
      ) {
        ids.add(node.id);
      }
    }

    return ids.size > 0 ? ids : NO_NODES;
  }, [deps, model]);
};
