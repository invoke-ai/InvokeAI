import type { ResolvedModelRequirement, WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type { InvocationTemplates } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry, WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';
import type { ModelRequirementDeps } from '@features/workflow/ui/modelRequirementStatus';

import { DataList, Icon, Skeleton, Spinner, Stack, Text } from '@chakra-ui/react';
import { getAddModelsSearchTerm, resolveWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { REQUIREMENT_STATUS_PRESENTATION, useModelRequirementDeps } from '@features/workflow/ui/modelRequirementStatus';
import { Tooltip } from '@platform/ui/Tooltip';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Highlight installable requirements only; other missing states have no actionable download. */

const SKELETON_ROW_COUNT = 2;
const EMPTY_COUNTS: ReadonlyMap<string, number> = new Map();

/** The extractor's own dedupe identity, reused so rows keep stable React keys. */
const getRequirementKey = (requirement: WorkflowModelRequirement): string =>
  requirement.kind === 'exact'
    ? `exact:${requirement.identifier.key}:${requirement.identifier.hash ?? ''}`
    : `slot:${requirement.base ?? ''}:${requirement.modelType ?? ''}`;

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

const nodePackCache = new WeakMap<ReadyEnrichment, { templates: InvocationTemplates; packs: readonly string[] }>();

/**
 * The node packs a workflow uses that this server does not have: the packs of its invocation nodes whose type is
 * absent from the loaded templates. Cached by enrichment and templates identity, so a card computes it once.
 */
export const getMissingNodePacks = (enrichment: ReadyEnrichment, templates: InvocationTemplates): readonly string[] => {
  const cached = nodePackCache.get(enrichment);

  if (cached && cached.templates === templates) {
    return cached.packs;
  }

  const packs = new Set<string>();

  for (const node of enrichment.document.nodes) {
    if (node.type === 'invocation' && templates[node.data.type] === undefined) {
      packs.add(node.data.nodePack || node.data.type);
    }
  }

  const result = [...packs].sort();

  nodePackCache.set(enrichment, { packs: result, templates });

  return result;
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
  const presentation = REQUIREMENT_STATUS_PRESENTATION[resolved.status];
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
          <Spinner aria-label={statusLabel} borderWidth="1.5px" color={presentation.color} />
        )}
      </DataList.ItemLabel>
      <DataList.ItemValue color={presentation.color} fontSize="xs" minW="0">
        {canFindModel ? (
          <Tooltip content={t('workflowLibrary.findModel', { name: label })}>
            <Text asChild cursor="pointer" textAlign="start" truncate _hover={REQUIREMENT_LINK_HOVER}>
              <button data-requirement-link={searchTerm} type="button" onClick={handleFindModel}>
                {label}
              </button>
            </Text>
          </Tooltip>
        ) : (
          <Tooltip content={label}>
            <Text truncate>{label}</Text>
          </Tooltip>
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
      <Text color="fg.muted" fontSize="xs" fontWeight="600">
        {t('workflowLibrary.requires')}
      </Text>
      {errorMessage ? (
        <Text color="fg.subtle" fontSize="xs">
          {errorMessage}
        </Text>
      ) : null}
      {!errorMessage && resolved === null
        ? Array.from({ length: SKELETON_ROW_COUNT }, (_unused, index) => (
            <Skeleton key={index} data-requirement-placeholder h="3" rounded="sm" w="24" />
          ))
        : null}
      {!errorMessage && resolved !== null && resolved.length > 0 ? (
        <DataList.Root gap="1.5" orientation="horizontal">
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
