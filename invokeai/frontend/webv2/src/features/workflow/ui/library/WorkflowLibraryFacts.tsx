import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { TFunction } from 'i18next';

import { Badge, Box, HStack, Skeleton, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models';
import { useTranslation } from 'react-i18next';

/**
 * The facts line a card and a list row share: base model, a readiness dot and one leading fact. The derivation is one
 * function so the two layouts can never disagree about what a workflow needs.
 */

/** What the facts line leads with once enrichment is ready. */
export type WorkflowLibraryReadiness = 'ready' | 'models' | 'nodePacks';

const READINESS_COLOR: Record<WorkflowLibraryReadiness, string> = {
  models: 'fg.warning',
  nodePacks: 'fg.error',
  ready: 'fg.success',
};

/**
 * Fact text in the readiness hue, pulled 30% toward the foreground: darker on light themes and lighter on dark ones,
 * so 10px text keeps 4.5:1 on the muted and emphasized fills of hovered and selected cards. The dot keeps the pure hue.
 */
const READINESS_TEXT_COLOR: Record<Exclude<WorkflowLibraryReadiness, 'ready'>, string> = {
  models: 'color-mix(in oklab, {colors.fg.warning} 70%, {colors.fg})',
  nodePacks: 'color-mix(in oklab, {colors.fg.error} 70%, {colors.fg})',
};

export interface WorkflowLibraryEntryFactsInput {
  entry: WorkflowLibraryEntry;
  /** The project already holds a copy of this library template. */
  isInProject: boolean;
  /** Models this workflow needs that are not installed; 0 when none. */
  missingCount: number;
  /** Node packs the workflow uses that this server lacks; null while node templates are unknown. */
  missingNodePackCount: number | null;
  /** When a project workflow last ran, already formatted; empty when it never ran or the caller does not say. */
  lastRun: string;
}

export interface WorkflowLibraryEntryFacts {
  isPending: boolean;
  primaryBase: string | null;
  /** Null until enrichment and node templates are known. */
  readiness: WorkflowLibraryReadiness | null;
  fact: string | null;
}

/** Missing node packs outrank missing models; a runnable workflow says it is in the project, ran, or its size. */
export const getWorkflowLibraryEntryFacts = (
  { entry, isInProject, lastRun, missingCount, missingNodePackCount }: WorkflowLibraryEntryFactsInput,
  t: TFunction
): WorkflowLibraryEntryFacts => {
  const ready = entry.enrichment.status === 'ready' ? entry.enrichment : null;
  const readiness: WorkflowLibraryReadiness | null =
    ready === null || missingNodePackCount === null
      ? null
      : missingNodePackCount > 0
        ? 'nodePacks'
        : missingCount > 0
          ? 'models'
          : 'ready';
  const fact =
    ready === null
      ? null
      : readiness === 'nodePacks'
        ? t('workflowLibrary.nodePacksMissing', { count: missingNodePackCount ?? 0 })
        : missingCount > 0
          ? t('workflowLibrary.modelsToInstall', { count: missingCount })
          : isInProject
            ? t('workflowLibrary.inProject')
            : lastRun
              ? t('workflowLibrary.ranWhen', { when: lastRun })
              : t('workflowLibrary.nodeCount', { count: ready.nodeCount });

  return {
    fact,
    isPending: entry.enrichment.status === 'pending',
    primaryBase: ready?.requirements.primaryBase ?? null,
    readiness,
  };
};

/**
 * On an accent fill (the selected list row) every readiness hue measures under 1.4:1, so the dot sits on a 2px ring
 * of the panel colour, against which each hue keeps at least 3:1 in every theme, and the fact text takes the row's.
 */
const ON_ACCENT_DOT_RING = '0 0 0 2px {colors.bg.panel}';

/** One 16px line, so async facts never change a card's or row's height. */
export const WorkflowLibraryFactsLine = ({
  facts,
  isOnAccent = false,
}: {
  facts: WorkflowLibraryEntryFacts;
  /** The line sits on the accent fill, which the readiness hues do not read on. */
  isOnAccent?: boolean;
}) => {
  const { t } = useTranslation();
  const { fact, isPending, primaryBase, readiness } = facts;

  return (
    <HStack gap="1.5" h="4" minW="0">
      {isPending ? (
        // Show placeholders only while enriching; unreadable workflows leave facts absent.
        <Skeleton data-enrichment-placeholder h="3" rounded="sm" w="14" />
      ) : null}
      {primaryBase ? (
        <Badge flexShrink={0} variant="subtle">
          {getModelBaseLabel(primaryBase)}
        </Badge>
      ) : null}
      {readiness ? (
        <Box
          bg={READINESS_COLOR[readiness]}
          boxShadow={isOnAccent ? ON_ACCENT_DOT_RING : undefined}
          boxSize="1.5"
          data-workflow-readiness={readiness}
          flexShrink={0}
          rounded="full"
        >
          {readiness === 'ready' ? <Text srOnly>{t('workflowLibrary.readyToRun')}</Text> : null}
        </Box>
      ) : null}
      {fact ? (
        <Text
          color={
            isOnAccent ? 'inherit' : readiness && readiness !== 'ready' ? READINESS_TEXT_COLOR[readiness] : 'fg.muted'
          }
          fontSize="xs"
          minW="0"
          truncate
        >
          {fact}
        </Text>
      ) : null}
    </HStack>
  );
};
