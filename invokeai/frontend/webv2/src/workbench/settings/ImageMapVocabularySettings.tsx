/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { ImageMapVocab } from '@workbench/image-map/vocabulary';

import { HStack, Icon, Input, Spinner, Stack, Tag, Text, Wrap } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field } from '@platform/ui';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { refetchClusterLabels } from '@workbench/image-map/imageMapStore';
import { imageMapVocabKeys, imageMapVocabQueryOptions, updateImageMapVocab } from '@workbench/image-map/vocabulary';
import { PlusIcon } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

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
 * polls while that runs and re-fetches any open map's labels when it lands.
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

  // A vocabulary edit changes what the labels say without moving a single
  // point, so nothing in the map's own flow would re-fetch them: do it here,
  // on the rebuild's building -> ready edge.
  const buildState = query.data?.state;
  const previousBuildState = useRef(buildState);

  useEffect(() => {
    if (previousBuildState.current === 'building' && buildState === 'ready') {
      refetchClusterLabels();
    }

    previousBuildState.current = buildState;
  }, [buildState]);

  const persist = (nextTerms: string[]): Promise<boolean> => {
    setSaveError(null);

    return run(
      async (owner) => {
        const updated = await updateImageMapVocab(nextTerms);

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
      <VocabularyStatusLine vocab={vocab} />
    </Stack>
  );
};

const VocabularyStatusLine = ({ vocab }: { vocab: ImageMapVocab }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="1">
      <Text color="fg.subtle" fontSize="2xs">
        {t('settings.imageMapVocabulary.count', { count: vocab.terms.length, max: vocab.maxTerms })}
      </Text>
      {vocab.state === 'building' ? (
        <HStack gap="1.5">
          <Spinner size="xs" />
          <Text color="fg.subtle" fontSize="2xs">
            {t('settings.imageMapVocabulary.rebuilding')}
          </Text>
        </HStack>
      ) : null}
      {vocab.state === 'error' && vocab.error ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {t('settings.imageMapVocabulary.buildFailed', { message: vocab.error })}
        </Text>
      ) : null}
      {vocab.state === 'unavailable' ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('settings.imageMapVocabulary.indexOff')}
        </Text>
      ) : null}
    </Stack>
  );
};
