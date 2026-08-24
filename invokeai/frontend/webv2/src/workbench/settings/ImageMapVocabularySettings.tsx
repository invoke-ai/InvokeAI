/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { ImageIndexCounts } from '@workbench/image-map/indexProgress';
import type { ImageMapVocab } from '@workbench/image-map/vocabulary';

import { HStack, Icon, Input, Spinner, Stack, Tag, Text, Wrap } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field } from '@platform/ui';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { imageMapStore, refreshImageIndexStatus } from '@workbench/image-map/imageMapStore';
import { describeIndexProgress, isIndexing } from '@workbench/image-map/indexProgress';
import { imageMapVocabKeys, imageMapVocabQueryOptions, updateImageMapVocab } from '@workbench/image-map/vocabulary';
import { PlusIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * How often the index counts are re-read while a rebuild is queued behind one.
 * Coarse on purpose: it only feeds a "waiting for indexing" line, and the
 * backend pushes `image_index_status` per batch anyway — this exists so the
 * line is right when the panel is opened mid-run with no event due.
 */
const INDEX_STATUS_POLL_MS = 5_000;

/**
 * Normalize one draft entry the way the server will, so the chips a save adds
 * are exactly the chips that come back. Splitting on commas and newlines makes
 * a pasted list import in one keystroke.
 */
const parseDraft = (draft: string): string[] => {
  const seen = new Set<string>();
  const terms: string[] = [];

  for (const entry of draft.split(/[\n,]/)) {
    const term = entry.split(/\s+/).filter(Boolean).join(' ').toLowerCase();

    if (term.length === 0 || seen.has(term)) {
      continue;
    }

    seen.add(term);
    terms.push(term);
  }

  return terms;
};

/**
 * Editor for the supplementary cluster-label vocabulary. Each add/remove
 * persists immediately (the whole list is replaced server-side); after a save
 * the server rebuilds the label embeddings in the background, so the query
 * polls while that runs to keep the status line honest. Re-fetching an open
 * map's labels when the rebuild lands is NOT this component's job — the data
 * module's watcher does it, because the dialog may well be closed by then.
 */
export const ImageMapVocabularySettings = () => {
  const { t } = useTranslation();
  const { canManageImageMapVocabulary } = useCapabilities();
  const queryClient = useQueryClient();
  const query = useQuery({
    ...imageMapVocabQueryOptions(),
    refetchInterval: (current) => (current.state.data?.state === 'building' ? 2_000 : false),
  });
  const [draft, setDraft] = useState('');
  const [inputError, setInputError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const { isBusy: isSaving, run } = useScopedAction();
  // The rebuild runs on the index worker, which services it only once it has
  // no images left to embed — so during a backfill the spinner can stand for
  // as long as the backfill does. The counts turn that into a stated wait.
  const indexCounts = imageMapStore.useSelector((snapshot) => snapshot.indexCounts);
  const isBuilding = query.data?.state === 'building';

  useEffect(() => {
    if (!isBuilding) {
      return;
    }

    refreshImageIndexStatus();
    const timer = setInterval(refreshImageIndexStatus, INDEX_STATUS_POLL_MS);

    return () => clearInterval(timer);
  }, [isBuilding]);

  const persist = (nextTerms: string[]): Promise<boolean> => {
    setSaveError(null);

    return run(
      async (owner) => {
        const updated = await updateImageMapVocab(nextTerms);

        assertAccountScopeCurrent(owner);
        // Cancel first: a poll that was issued before the PUT can resolve
        // after it, and TanStack Query would write that older list over the
        // response — resurrecting a removed chip and, worse, overwriting
        // 'building' with a stale 'ready' so the status line goes quiet while
        // the server is still rebuilding.
        await queryClient.cancelQueries({ queryKey: imageMapVocabKeys.all });
        assertAccountScopeCurrent(owner);
        queryClient.setQueryData(imageMapVocabKeys.all, updated);
      },
      (_message, error) => {
        setSaveError(getApiErrorMessage(error, t('settings.imageMapVocabulary.saveFailed')));
      }
    );
  };

  const addTerms = async (vocab: ImageMapVocab) => {
    const candidates = parseDraft(draft);

    if (candidates.length === 0) {
      return;
    }

    const overlong = candidates.find((term) => term.length > vocab.maxTermLength);

    if (overlong) {
      setInputError(t('settings.imageMapVocabulary.termTooLong', { max: vocab.maxTermLength }));
      return;
    }

    const existing = new Set(vocab.terms);
    const fresh = candidates.filter((term) => !existing.has(term));

    if (fresh.length === 0) {
      setInputError(t('settings.imageMapVocabulary.alreadyInList'));
      return;
    }

    if (vocab.terms.length + fresh.length > vocab.maxTerms) {
      setInputError(t('settings.imageMapVocabulary.tooManyTerms', { max: vocab.maxTerms }));
      return;
    }

    setInputError(null);

    // Clear the draft only once it is saved, so a failure never eats the text.
    if (await persist([...vocab.terms, ...fresh])) {
      setDraft('');
    }
  };

  if (query.isPending) {
    return <Spinner size="sm" />;
  }

  if (query.isError) {
    return (
      <Stack align="flex-start" gap="2">
        <Text color="fg.error" fontSize="xs">
          {t('settings.imageMapVocabulary.loadFailed')}
        </Text>
        <Button size="xs" variant="outline" onClick={() => void query.refetch()}>
          {t('common.retry')}
        </Button>
      </Stack>
    );
  }

  const vocab = query.data;

  return (
    <Stack gap="3">
      {canManageImageMapVocabulary ? (
        <Field
          error={inputError ?? saveError}
          helpText={t('settings.imageMapVocabulary.addHelp')}
          label={t('settings.imageMapVocabulary.addLabel')}
        >
          <HStack gap="1.5" w="full">
            <Input
              aria-invalid={inputError || saveError ? true : undefined}
              placeholder={t('settings.imageMapVocabulary.addPlaceholder')}
              size="xs"
              value={draft}
              onChange={(event) => {
                setDraft(event.currentTarget.value);
                setInputError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void addTerms(vocab);
                }
              }}
              onPaste={(event) => {
                // A single-line input silently flattens pasted newlines, which
                // would fuse a copied term-per-line list into one bogus term.
                // Rewrite the paste as the comma-separated form the parser
                // (and the placeholder) already speak.
                const text = event.clipboardData.getData('text');

                if (!text.includes('\n')) {
                  return;
                }

                event.preventDefault();
                const merged = text
                  .split(/[\n,]/)
                  .map((entry) => entry.trim())
                  .filter(Boolean)
                  .join(', ');
                const element = event.currentTarget;
                const start = element.selectionStart ?? element.value.length;
                const end = element.selectionEnd ?? element.value.length;

                setDraft(element.value.slice(0, start) + merged + element.value.slice(end));
                setInputError(null);
              }}
            />
            <Button
              disabled={draft.trim().length === 0}
              loading={isSaving}
              size="xs"
              variant="outline"
              onClick={() => {
                void addTerms(vocab);
              }}
            >
              <Icon as={PlusIcon} boxSize="3.5" />
              {t('common.add')}
            </Button>
          </HStack>
        </Field>
      ) : (
        <Text color="fg.subtle" fontSize="xs">
          {t('settings.imageMapVocabulary.adminOnly')}
        </Text>
      )}
      {vocab.terms.length > 0 ? (
        <Wrap gap="1">
          {vocab.terms.map((term) => (
            <Tag.Root key={term} size="sm" variant="surface">
              <Tag.Label>{term}</Tag.Label>
              {canManageImageMapVocabulary ? (
                <Tag.EndElement>
                  <Tag.CloseTrigger
                    aria-label={t('settings.imageMapVocabulary.removeTerm', { term })}
                    // Disabled while a save is in flight: the busy guard would
                    // silently drop a second removal, resurrecting the chip
                    // with no feedback.
                    disabled={isSaving}
                    onClick={() => {
                      void persist(vocab.terms.filter((existing) => existing !== term));
                    }}
                  />
                </Tag.EndElement>
              ) : null}
            </Tag.Root>
          ))}
        </Wrap>
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          {t('settings.imageMapVocabulary.noTermsYet')}
        </Text>
      )}
      <VocabularyStatusLine
        indexCounts={indexCounts}
        vocab={vocab}
        onRetry={
          canManageImageMapVocabulary
            ? () => {
                // Re-saving the same list re-triggers the invalidation, which
                // is the server's retry path for a failed embedding build.
                void persist(vocab.terms);
              }
            : undefined
        }
      />
    </Stack>
  );
};

const VocabularyStatusLine = ({
  indexCounts,
  onRetry,
  vocab,
}: {
  indexCounts: ImageIndexCounts | null;
  onRetry?: () => void;
  vocab: ImageMapVocab;
}) => {
  const { t } = useTranslation();
  // Embedding a handful of terms takes seconds; a wait longer than that means
  // the rebuild is queued behind image indexing, not running slowly. Saying
  // which is the difference between "working" and "hung".
  //
  // No age is passed, so no "no progress reported for N" note: that needs a
  // clock ticking in render, which the compiler rightly refuses, and the whole
  // stall treatment already exists on the map widget where the counts live. A
  // stalled index shows up here as a number that stops moving.
  const progress = isIndexing(indexCounts) ? describeIndexProgress(indexCounts) : null;

  return (
    <Stack gap="1">
      <Text color="fg.subtle" fontSize="2xs">
        {t('settings.imageMapVocabulary.count', { count: vocab.terms.length, max: vocab.maxTerms })}
      </Text>
      {vocab.state === 'building' ? (
        <HStack gap="1.5">
          <Spinner size="xs" />
          <Text color="fg.subtle" fontSize="2xs">
            {progress
              ? t('settings.imageMapVocabulary.rebuildingQueued', { progress: progress.counts })
              : t('settings.imageMapVocabulary.rebuilding')}
          </Text>
        </HStack>
      ) : null}
      {vocab.state === 'error' && vocab.error ? (
        <HStack gap="2">
          <Text color="fg.error" fontSize="2xs" role="alert">
            {t('settings.imageMapVocabulary.buildFailed', { message: vocab.error })}
          </Text>
          {onRetry ? (
            <Button size="2xs" variant="outline" onClick={onRetry}>
              {t('common.retry')}
            </Button>
          ) : null}
        </HStack>
      ) : null}
      {vocab.state === 'unavailable' ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('settings.imageMapVocabulary.indexOff')}
        </Text>
      ) : null}
    </Stack>
  );
};
