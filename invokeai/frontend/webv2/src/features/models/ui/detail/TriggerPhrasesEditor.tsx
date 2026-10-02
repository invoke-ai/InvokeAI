/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import { TagsInput, Text } from '@chakra-ui/react';
import { triggerPhraseSchema } from '@features/models/core/schemas';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Field } from '@platform/ui/Field';
import { memo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface TriggerPhrasesEditorState {
  error: string | null;
  modelKey: string;
  /** The phrases prop this local value was derived from; a new prop replaces the value. */
  phrases: readonly string[];
  value: string[];
}

/** Persist trigger-phrase edits immediately and restore the prior list on save failure. */
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
    error: null,
    modelKey,
    phrases,
    value: [...phrases],
  }));
  // Replace local phrases during render when the prop list changes, including model switches and completed saves.
  if (editor.modelKey !== modelKey || editor.phrases !== phrases) {
    setEditor({ error: null, modelKey, phrases, value: [...phrases] });
  }
  const { error, value } = editor;
  const { isBusy: isSaving, run } = useScopedAction();

  const persist = (nextPhrases: string[]) => {
    setEditor({ error: null, modelKey, phrases, value: nextPhrases });
    void run(
      async (owner) => {
        const updated = await updateModel(modelKey, { trigger_phrases: nextPhrases }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
      },
      (_message, persistError) => {
        onError(persistError instanceof Error ? persistError.message : t('models.failedToUpdateTriggerPhrases'));
        setEditor({ error: null, modelKey, phrases, value: [...phrases] });
      }
    );
  };

  return (
    <Field error={error} helpText={t('models.triggerPhrasesHelp')} label={t('models.triggerPhrases')}>
      <TagsInput.Root
        blurBehavior="add"
        disabled={isSaving}
        editable
        size="sm"
        validate={({ inputValue, value: current }) => {
          const parsed = triggerPhraseSchema.safeParse(inputValue);

          if (!parsed.success) {
            setEditor({
              error: parsed.error.issues[0]?.message ?? t('models.invalidTriggerPhrase'),
              modelKey,
              phrases,
              value,
            });
            return false;
          }

          if (current.some((phrase) => phrase.toLowerCase() === parsed.data.toLowerCase())) {
            setEditor({ error: t('models.triggerPhraseDuplicate'), modelKey, phrases, value });
            return false;
          }

          return true;
        }}
        value={value}
        w="full"
        onValueChange={({ value: next }) => persist(next)}
      >
        <TagsInput.Control>
          <TagsInput.Context>
            {(api) =>
              api.value.map((phrase, index) => (
                <TagsInput.Item key={`${phrase}:${index}`} index={index} value={phrase}>
                  <TagsInput.ItemPreview>
                    <TagsInput.ItemText>{phrase}</TagsInput.ItemText>
                    <TagsInput.ItemDeleteTrigger aria-label={t('models.removeTriggerPhrase', { phrase })} />
                  </TagsInput.ItemPreview>
                  <TagsInput.ItemInput />
                </TagsInput.Item>
              ))
            }
          </TagsInput.Context>
          <TagsInput.Input aria-label={t('models.triggerPhrases')} placeholder={t('models.addTriggerPhrase')} />
        </TagsInput.Control>
        <TagsInput.HiddenInput />
      </TagsInput.Root>
      {value.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.noTriggerPhrasesYet')}
        </Text>
      ) : null}
    </Field>
  );
};

export const MemoizedTriggerPhrasesEditor = memo(TriggerPhrasesEditor);
