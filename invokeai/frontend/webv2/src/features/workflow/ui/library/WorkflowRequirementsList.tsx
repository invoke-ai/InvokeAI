import type { ModelConfig, StarterModel } from '@features/models';
import type {
  ModelRequirementStatus,
  ResolvedModelRequirement,
  WorkflowModelRequirement,
} from '@features/workflow/core/modelRequirements';
import type { WorkflowLibraryEntry, WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';
import type { ElementType } from 'react';

import { DataList, Icon, Skeleton, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  ensureModelsLoaded,
  ensureStartersLoaded,
  useActiveInstallSources,
  useModelsSelector,
  useStartersSelector,
} from '@features/models';
import { getAddModelsSearchTerm, resolveWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CheckIcon, DownloadIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Highlight installable requirements only; other missing states have no actionable download. */

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

/** Share catalog loading and resolution inputs between details and grid badges. */
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

type ReadyEnrichment = Extract<WorkflowLibraryEntryEnrichment, { status: 'ready' }>;

/**
 * Cache by enrichment identity and catalog dependencies to avoid quadratic row resolution as individual entries
 * finish parsing.
 */
const resolutionCache = new WeakMap<
  ReadyEnrichment,
  { deps: ModelRequirementDeps; resolved: ResolvedModelRequirement[] }
>();

/** Resolves one entry's requirements, reusing the previous result when nothing it depends on changed. */
export const resolveEntryRequirements = (
  enrichment: ReadyEnrichment,
  deps: ModelRequirementDeps
): ResolvedModelRequirement[] => {
  const cached = resolutionCache.get(enrichment);

  if (cached && cached.deps === deps) {
    return cached.resolved;
  }

  const resolved = resolveWorkflowModelRequirements(enrichment.requirements.requirements, deps);

  resolutionCache.set(enrichment, { deps, resolved });

  return resolved;
};

/** Count only models the install action can download so card badges agree with the offered operation. */
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

      const installable = resolveEntryRequirements(entry.enrichment, deps).filter(
        (resolved) => resolved.status === 'installable'
      ).length;

      if (installable > 0) {
        counts.set(entry.item.workflow_id, installable);
      }
    }

    return counts.size > 0 ? counts : EMPTY_COUNTS;
  }, [deps, entries]);
};

/** Quiet by design: an underline on hover and nothing else — the rows are facts, not a toolbar. */
const REQUIREMENT_LINK_HOVER = { textDecoration: 'underline' } as const;

const RequirementRow = ({
  resolved,
  onFindModel,
}: {
  resolved: ResolvedModelRequirement;
  /** Absent when the panel has no way to reach Add Models. */
  onFindModel?: (query: string) => void;
}) => {
  const { t } = useTranslation();
  const presentation = STATUS_PRESENTATION[resolved.status];
  const statusLabel = t(presentation.labelKey);
  const { label } = resolved.requirement;
  const searchTerm = getAddModelsSearchTerm(resolved);
  const canFindModel = Boolean(onFindModel) && searchTerm !== null;
  const handleFindModel = useCallback(() => {
    if (searchTerm) {
      onFindModel?.(searchTerm);
    }
  }, [onFindModel, searchTerm]);

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
        {canFindModel ? (
          <Text asChild cursor="pointer" textAlign="start" truncate _hover={REQUIREMENT_LINK_HOVER}>
            <button
              data-requirement-link={searchTerm}
              title={t('workflowLibrary.findModel', { name: label })}
              type="button"
              onClick={handleFindModel}
            >
              {label}
            </button>
          </Text>
        ) : (
          <Text truncate title={label}>
            {label}
          </Text>
        )}
      </DataList.ItemValue>
    </DataList.Item>
  );
};

export interface WorkflowRequirementsListProps {
  /** Quiet line shown instead of rows when the workflow itself could not be read. */
  errorMessage: string | null;
  /** `null` while the workflow is still being parsed in the background. */
  resolved: readonly ResolvedModelRequirement[] | null;
  /** Turns rows for models the account is missing into links into Add Models. */
  onFindModel?: (query: string) => void;
}

export const WorkflowRequirementsList = ({ errorMessage, resolved, onFindModel }: WorkflowRequirementsListProps) => {
  const { t } = useTranslation();

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
            <Skeleton key={index} data-requirement-placeholder h="3" rounded="sm" w="24" />
          ))
        : null}
      {!errorMessage && resolved !== null && resolved.length > 0 ? (
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          {resolved.map((requirement) => (
            <RequirementRow
              key={getRequirementKey(requirement.requirement)}
              resolved={requirement}
              onFindModel={onFindModel}
            />
          ))}
        </DataList.Root>
      ) : null}
    </Stack>
  );
};
