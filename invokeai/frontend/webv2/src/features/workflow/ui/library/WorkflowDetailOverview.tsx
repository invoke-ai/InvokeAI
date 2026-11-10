import type { ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowPreviewModel } from '@features/workflow/core/workflowPreview';

import { Badge, chakra, HStack, Kbd, Stack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { buildWorkflowPreviewModel } from '@features/workflow/core/workflowPreview';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { Button } from '@platform/ui/Button';
import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The detail rail's account of a workflow before it is opened: where it comes from, what its author says about it,
 * and what a person fills in. Read from the same preview model the stage draws, so the rail and the preview agree.
 */

const NOTES_LINES = 3;
/** First labels per heading the "You set" lines name before "+N". */
const LABELS_PER_GROUP = 3;
const SECTION_LABEL_PROPS = { color: 'fg.muted', fontSize: 'xs', fontWeight: '600' } as const;

/** The workflow's preview model once its document and the node templates are known. */
export const useWorkflowPreviewModel = (document: ProjectGraphState | null): WorkflowPreviewModel | null => {
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const templates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null;

  return useMemo(
    () => (document && templates ? buildWorkflowPreviewModel(document, templates) : null),
    [document, templates]
  );
};

/** "Template · by Invoke · v1.0.0": only the parts that are known. */
export const WorkflowProvenance = ({ document, kind }: { document: ProjectGraphState | null; kind: string }) => {
  const { t } = useTranslation();
  const author = document?.author.trim();
  const version = document?.workflowVersion.trim();
  const line = [
    kind,
    author ? t('workflowLibrary.preview.byAuthor', { author }) : null,
    version ? t('workflowLibrary.preview.version', { version }) : null,
  ]
    .filter((part) => part !== null)
    .join(' · ');

  return (
    <Text color="fg.muted" data-workflow-provenance="" fontSize="xs">
      {line}
    </Text>
  );
};

/** The "N inputs" fact beside the node count. */
export const WorkflowInputsBadge = ({ model }: { model: WorkflowPreviewModel | null }) => {
  const { t } = useTranslation();

  return model && model.inputs.length > 0 ? (
    <Badge data-workflow-inputs-count="" variant="subtle">
      {t('workflowLibrary.preview.inputCount', { count: model.inputs.length })}
    </Badge>
  ) : null;
};

/** The workflow's own notes, three lines until "Read all notes" opens them in place. */
export const WorkflowAuthorNotes = ({ notes }: { notes: string }) => {
  const { t } = useTranslation();
  const notesId = useId();
  const textRef = useRef<HTMLParagraphElement>(null);
  const [expandedNotes, setExpandedNotes] = useState<string | null>(null);
  const [clampedNotes, setClampedNotes] = useState<string | null>(null);
  const isExpanded = expandedNotes === notes;
  const toggle = useCallback(() => setExpandedNotes(isExpanded ? null : notes), [isExpanded, notes]);

  // Whether three lines cut the notes off decides the toggle; measured before paint, once per text.
  useLayoutEffect(() => {
    const element = textRef.current;

    if (element && !isExpanded) {
      setClampedNotes(element.scrollHeight > element.clientHeight + 1 ? notes : null);
    }
  }, [isExpanded, notes]);

  if (!notes.trim()) {
    return null;
  }

  return (
    <Stack data-workflow-author-notes="" gap="1" minW="0">
      <Text {...SECTION_LABEL_PROPS}>{t('workflowLibrary.preview.authorNotes')}</Text>
      <Text
        ref={textRef}
        color="fg.muted"
        fontSize="xs"
        id={notesId}
        lineClamp={isExpanded ? undefined : NOTES_LINES}
        overflowWrap="anywhere"
        whiteSpace="pre-wrap"
      >
        {notes}
      </Text>
      {clampedNotes === notes || isExpanded ? (
        <Button
          alignSelf="flex-start"
          aria-controls={notesId}
          aria-expanded={isExpanded}
          color="accent.fg"
          fontSize="xs"
          h="auto"
          px="0"
          size="xs"
          variant="plain"
          onClick={toggle}
        >
          {isExpanded ? t('workflowLibrary.preview.showLessNotes') : t('workflowLibrary.preview.readAllNotes')}
        </Button>
      ) : null}
    </Stack>
  );
};

/** "You set": a line per form heading naming its first inputs, and the way into the preview's Inputs view. */
export const WorkflowInputsSummary = ({
  model,
  onSeeAll,
}: {
  model: WorkflowPreviewModel | null;
  onSeeAll: () => void;
}) => {
  const { i18n, t } = useTranslation();
  const listFormat = useMemo(
    () => new Intl.ListFormat(i18n.resolvedLanguage, { style: 'short', type: 'unit' }),
    [i18n.resolvedLanguage]
  );

  if (!model || model.inputs.length === 0) {
    return null;
  }

  return (
    <Stack data-workflow-inputs-summary="" gap="1" minW="0">
      <Text {...SECTION_LABEL_PROPS}>{t('workflowLibrary.preview.youSet')}</Text>
      {model.inputGroups.map((group, index) => {
        const shown = group.labels.slice(0, LABELS_PER_GROUP);
        const rest = group.labels.length - shown.length;
        const labels = listFormat.format(
          rest > 0 ? [...shown, t('workflowLibrary.preview.more', { count: rest })] : shown
        );

        return (
          <Text key={index} fontSize="xs" overflowWrap="anywhere">
            {group.heading ? (
              <Text as="span" color="fg.muted">
                {group.heading} ·{' '}
              </Text>
            ) : null}
            {labels}
          </Text>
        );
      })}
      <Button
        alignSelf="flex-start"
        color="accent.fg"
        data-workflow-see-inputs=""
        fontSize="xs"
        h="auto"
        px="0"
        size="xs"
        variant="plain"
        onClick={onSeeAll}
      >
        {t('workflowLibrary.preview.seeAllInputs', { count: model.inputs.length })}
      </Button>
    </Stack>
  );
};

/**
 * For the rail's thumbnail slot: preview mode draws the graph on the stage, so a thumbnail that only drew the same
 * graph's silhouette steps aside (`data-thumbnail-kind`); a real cover image stays.
 */
export const WORKFLOW_SILHOUETTE_THUMBNAIL_CSS: SystemStyleObject = {
  '&:is([data-previewing] *):has([data-thumbnail-kind="silhouette"])': { display: 'none' },
};
const THUMBNAIL_BUTTON_CSS: SystemStyleObject = {
  cursor: 'pointer',
  inset: '0',
  position: 'absolute',
  rounded: 'md',
  _focusVisible: { outlineColor: 'accent.solid', outlineOffset: '-2px', outlineStyle: 'solid', outlineWidth: '2px' },
  '[data-previewing] &': { display: 'none' },
};

/**
 * The rail thumbnail as the way into the preview: a press anywhere on it previews the workflow, and its corner names
 * the shortcut a selected card answers. Laid over the thumbnail, under any controls drawn after it.
 */
export const WorkflowPreviewThumbnailButton = ({ name, onPreview }: { name: string; onPreview: () => void }) => {
  const { t } = useTranslation();

  return (
    <chakra.button
      aria-label={t('workflowLibrary.preview.previewNamed', { name })}
      css={THUMBNAIL_BUTTON_CSS}
      data-thumbnail-preview=""
      type="button"
      onClick={onPreview}
    >
      <HStack
        aria-hidden
        bg="bg.panel/90"
        borderColor="border.emphasized"
        borderWidth="1px"
        bottom="1.5"
        fontSize="sm"
        gap="1.5"
        h="5.5"
        insetEnd="1.5"
        position="absolute"
        px="2"
        rounded="md"
      >
        {t('workflowLibrary.preview.action')}
        <Kbd size="md">{t('workflowLibrary.preview.spaceKey')}</Kbd>
      </HStack>
    </chakra.button>
  );
};
