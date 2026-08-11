/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { HStack, Icon, Input, Tag, Text, Wrap } from '@chakra-ui/react';
import { triggerPhraseSchema } from '@features/models/core/schemas';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, Field } from '@platform/ui';
import { PlusIcon } from 'lucide-react';
import { memo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface TriggerPhrasesEditorState {
  draft: string;
  error: string | null;
  isSaving: boolean;
  modelKey: string;
}

/**
 * Tag-style editor for a model's trigger phrases. Each add/remove persists
 * immediately — there is no separate save step for phrases.
 */
export const TriggerPhrasesEditor = ({
  modelKey,
  onError,
  phrases,
}: {
  modelKey: string;
  onError: (message: string) => void;
  phrases: readonly string[];
}) => {
  const { t } = useTranslation();
  const [editor, setEditor] = useState<TriggerPhrasesEditorState>(() => ({
    draft: '',
    error: null,
    isSaving: false,
    modelKey,
  }));
  const isEditorCurrent = editor.modelKey === modelKey;
  const draft = isEditorCurrent ? editor.draft : '';
  const error = isEditorCurrent ? editor.error : null;
  const isSaving = isEditorCurrent ? editor.isSaving : false;
  // The saving flag lives in the modelKey-scoped editor state (so it resets on
  // model switch); the hook's own isBusy is unused, only its scope plumbing.
  const { run } = useScopedAction();

  const persist = async (nextPhrases: string[]): Promise<boolean> => {
    setEditor((current) => ({
      draft: current.modelKey === modelKey ? current.draft : '',
      error: current.modelKey === modelKey ? current.error : null,
      isSaving: true,
      modelKey,
    }));

    const stopSaving = () =>
      setEditor((current) => (current.modelKey === modelKey ? { ...current, isSaving: false } : current));
    let saved = false;

    await run(
      async (owner) => {
        const updated = await updateModel(modelKey, { trigger_phrases: nextPhrases }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        saved = true;
        stopSaving();
      },
      (_message, persistError) => {
        onError(persistError instanceof Error ? persistError.message : t('models.failedToUpdateTriggerPhrases'));
        stopSaving();
      }
    );

    return saved;
  };

  const addPhrase = async () => {
    const parsed = triggerPhraseSchema.safeParse(draft);

    if (!parsed.success) {
      setEditor({
        draft,
        error: parsed.error.issues[0]?.message ?? t('models.invalidTriggerPhrase'),
        isSaving,
        modelKey,
      });
      return;
    }

    if (phrases.some((phrase) => phrase.toLowerCase() === parsed.data.toLowerCase())) {
      setEditor({ draft, error: t('models.triggerPhraseDuplicate'), isSaving, modelKey });
      return;
    }

    setEditor({ draft, error: null, isSaving, modelKey });

    // Clear the draft only once it is saved, so a failure never eats the text.
    if (await persist([...phrases, parsed.data])) {
      setEditor((current) => (current.modelKey === modelKey ? { ...current, draft: '' } : current));
    }
  };

  return (
    <Field error={error} helpText={t('models.triggerPhrasesHelp')} label={t('models.triggerPhrases')}>
      <HStack gap="1.5">
        <Input
          aria-invalid={error ? true : undefined}
          placeholder={t('models.addTriggerPhrase')}
          size="xs"
          value={draft}
          onChange={(event) => {
            setEditor({ draft: event.currentTarget.value, error: null, isSaving, modelKey });
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void addPhrase();
            }
          }}
        />
        <Button
          disabled={draft.trim().length === 0}
          loading={isSaving}
          size="xs"
          variant="outline"
          onClick={() => {
            void addPhrase();
          }}
        >
          <Icon as={PlusIcon} boxSize="3.5" />
          {t('common.add')}
        </Button>
      </HStack>
      {phrases.length > 0 ? (
        <Wrap gap="1">
          {phrases.map((phrase) => (
            <Tag.Root key={phrase} size="sm" variant="surface">
              <Tag.Label>{phrase}</Tag.Label>
              <Tag.EndElement>
                <Tag.CloseTrigger
                  aria-label={t('models.removeTriggerPhrase', { phrase })}
                  onClick={() => {
                    void persist(phrases.filter((existing) => existing !== phrase));
                  }}
                />
              </Tag.EndElement>
            </Tag.Root>
          ))}
        </Wrap>
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.noTriggerPhrasesYet')}
        </Text>
      )}
    </Field>
  );
};

export const MemoizedTriggerPhrasesEditor = memo(TriggerPhrasesEditor);
