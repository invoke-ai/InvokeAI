/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { HStack, Icon, Input, Tag, Text, Wrap } from '@chakra-ui/react';
import { Button, Field } from '@workbench/components/ui';
import { updateModel } from '@workbench/models/api';
import { replaceModelInStore } from '@workbench/models/modelsStore';
import { triggerPhraseSchema } from '@workbench/models/schemas';
import { PlusIcon } from 'lucide-react';
import { memo, useState } from 'react';

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

  const persist = async (nextPhrases: string[]): Promise<boolean> => {
    setEditor((current) => ({
      draft: current.modelKey === modelKey ? current.draft : '',
      error: current.modelKey === modelKey ? current.error : null,
      isSaving: true,
      modelKey,
    }));

    try {
      replaceModelInStore(await updateModel(modelKey, { trigger_phrases: nextPhrases }));

      return true;
    } catch (persistError) {
      onError(persistError instanceof Error ? persistError.message : 'Failed to update trigger phrases.');

      return false;
    } finally {
      setEditor((current) => (current.modelKey === modelKey ? { ...current, isSaving: false } : current));
    }
  };

  const addPhrase = async () => {
    const parsed = triggerPhraseSchema.safeParse(draft);

    if (!parsed.success) {
      setEditor({ draft, error: parsed.error.issues[0]?.message ?? 'Invalid trigger phrase.', isSaving, modelKey });
      return;
    }

    if (phrases.some((phrase) => phrase.toLowerCase() === parsed.data.toLowerCase())) {
      setEditor({ draft, error: 'That trigger phrase is already on this model.', isSaving, modelKey });
      return;
    }

    setEditor({ draft, error: null, isSaving, modelKey });

    // Clear the draft only once it is saved, so a failure never eats the text.
    if (await persist([...phrases, parsed.data])) {
      setEditor((current) => (current.modelKey === modelKey ? { ...current, draft: '' } : current));
    }
  };

  return (
    <Field
      error={error}
      helpText="Phrases that activate this model, surfaced in the prompt editor."
      label="Trigger Phrases"
    >
      <HStack gap="1.5">
        <Input
          aria-invalid={error ? true : undefined}
          placeholder="Add a trigger phrase…"
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
          Add
        </Button>
      </HStack>
      {phrases.length > 0 ? (
        <Wrap gap="1">
          {phrases.map((phrase) => (
            <Tag.Root key={phrase} size="sm" variant="surface">
              <Tag.Label>{phrase}</Tag.Label>
              <Tag.EndElement>
                <Tag.CloseTrigger
                  aria-label={`Remove trigger phrase ${phrase}`}
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
          No trigger phrases yet.
        </Text>
      )}
    </Field>
  );
};

export const MemoizedTriggerPhrasesEditor = memo(TriggerPhrasesEditor);
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
