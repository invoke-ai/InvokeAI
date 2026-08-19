import type { ModelConfig, StarterModel } from '@features/models';
import type {
  ModelRequirementStatus,
  ResolvedModelRequirement,
  WorkflowModelRequirement,
} from '@features/workflow/core/modelRequirements';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { ElementType } from 'react';

import { DataList, Icon, Skeleton, Spinner, Stack, Text } from '@chakra-ui/react';
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
import { useTranslation } from 'react-i18next';

/**
 * "Requires": one row per model a workflow needs, resolved against what is
 * installed, what the starter catalog can fetch, and what is already
 * downloading. The rows stay neutral — only the *installable* state earns the
 * amber download treatment, because it is the only one the user can act on.
 */

const SKELETON_ROW_COUNT = 2;
const EMPTY_STARTERS: readonly StarterModel[] = [];
const EMPTY_COUNTS: ReadonlyMap<string, number> = new Map();

const selectInstalledModels = (snapshot: { models: ModelConfig[] }): readonly ModelConfig[] => snapshot.models;
const selectStarterModels = (snapshot: {
  response: { starter_models: StarterModel[] } | null;
}): readonly StarterModel[] => snapshot.response?.starter_models ?? EMPTY_STARTERS;

interface StatusPresentation {
  color: string;
  icon: ElementType | null;
  labelKey: string;
}

const STATUS_PRESENTATION: Record<ModelRequirementStatus, StatusPresentation> = {
  // Amber, per the mock, is the "you can fix this" signal and nothing else.
  installable: { color: 'fg.warning', icon: DownloadIcon, labelKey: 'workflowLibrary.requirementInstallable' },
  installed: { color: 'fg.muted', icon: CheckIcon, labelKey: 'workflowLibrary.requirementInstalled' },
  installing: { color: 'fg.muted', icon: null, labelKey: 'workflowLibrary.requirementInstalling' },
  unresolvable: { color: 'fg.muted', icon: TriangleAlertIcon, labelKey: 'workflowLibrary.requirementMissing' },
};

/** The extractor's own dedupe identity, reused so rows keep stable React keys. */
const getRequirementKey = (requirement: WorkflowModelRequirement): string =>
  requirement.kind === 'exact'
    ? `exact:${requirement.identifier.key}:${requirement.identifier.hash ?? ''}`
    : `slot:${requirement.base ?? ''}:${requirement.modelType ?? ''}`;

export interface ModelRequirementDeps {
  installedModels: readonly ModelConfig[];
  starterModels: readonly StarterModel[];
  activeInstallSources: ReadonlySet<string>;
}

/**
 * The three inputs every requirement resolution needs, plus the one-time load
 * of the two catalogs behind them. Both the detail panel and the grid's
 * missing-model badges read them through here, so they always resolve against
 * the same data.
 */
export const useModelRequirementDeps = (): ModelRequirementDeps => {
  const installedModels = useModelsSelector(selectInstalledModels);
  const starterModels = useStartersSelector(selectStarterModels);
  const activeInstallSources = useActiveInstallSources();

  useMountEffect(() => {
    void ensureModelsLoaded();
    ensureStartersLoaded();
  });

  return useMemo(
    () => ({ activeInstallSources, installedModels, starterModels }),
    [activeInstallSources, installedModels, starterModels]
  );
};

/**
 * How many models each card would have to *download* to run — the count the
 * grid badges and, on the selected card, exactly what the panel's Install
 * button offers to fetch. Requirements nothing in the catalog can satisfy are
 * deliberately excluded: an "Install 3 models" badge that installs 2 is worse
 * than a quiet card.
 */
export const useWorkflowLibraryMissingCounts = (
  entries: readonly WorkflowLibraryEntry[]
): ReadonlyMap<string, number> => {
  const deps = useModelRequirementDeps();

  return useMemo(() => {
    const counts = new Map<string, number>();

    for (const entry of entries) {
      if (entry.enrichment.status !== 'ready') {
        continue;
      }

      const installable = resolveWorkflowModelRequirements(entry.enrichment.requirements.requirements, deps).filter(
        (resolved) => resolved.status === 'installable'
      ).length;

      if (installable > 0) {
        counts.set(entry.item.workflow_id, installable);
      }
    }

    return counts.size > 0 ? counts : EMPTY_COUNTS;
  }, [deps, entries]);
};

const RequirementRow = ({ resolved }: { resolved: ResolvedModelRequirement }) => {
  const { t } = useTranslation();
  const presentation = STATUS_PRESENTATION[resolved.status];
  const statusLabel = t(presentation.labelKey);

  return (
    <DataList.Item alignItems="center" data-requirement-status={resolved.status} gap="1.5">
      <DataList.ItemLabel flex="0 0 auto" minW="0">
        {presentation.icon ? (
          <Icon aria-label={statusLabel} as={presentation.icon} boxSize="3" color={presentation.color} />
        ) : (
          <Spinner aria-label={statusLabel} borderWidth="1.5px" color={presentation.color} size="xs" />
        )}
      </DataList.ItemLabel>
      <DataList.ItemValue color={presentation.color} fontSize="2xs" minW="0">
        <Text truncate title={resolved.requirement.label}>
          {resolved.requirement.label}
        </Text>
      </DataList.ItemValue>
    </DataList.Item>
  );
};

export interface WorkflowRequirementsListProps {
  /** Quiet line shown instead of rows when the workflow itself could not be read. */
  errorMessage: string | null;
  /** `null` while the workflow is still being parsed in the background. */
  resolved: readonly ResolvedModelRequirement[] | null;
}

export const WorkflowRequirementsList = ({ errorMessage, resolved }: WorkflowRequirementsListProps) => {
  const { t } = useTranslation();

  // A readable workflow that needs no models says so by omission, not by an
  // empty section header.
  if (!errorMessage && resolved?.length === 0) {
    return null;
  }

  return (
    <Stack gap="1" minW="0">
      <Text color="fg.muted" fontSize="2xs" fontWeight="600">
        {t('workflowLibrary.requires')}
      </Text>
      {errorMessage ? (
        <Text color="fg.subtle" fontSize="2xs">
          {errorMessage}
        </Text>
      ) : null}
      {!errorMessage && resolved === null
        ? Array.from({ length: SKELETON_ROW_COUNT }, (_unused, index) => (
            <Skeleton key={index} data-requirement-placeholder h="3" rounded="sm" variant="pulse" w="24" />
          ))
        : null}
      {!errorMessage && resolved !== null && resolved.length > 0 ? (
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          {resolved.map((requirement) => (
            <RequirementRow key={getRequirementKey(requirement.requirement)} resolved={requirement} />
          ))}
        </DataList.Root>
      ) : null}
    </Stack>
  );
};
