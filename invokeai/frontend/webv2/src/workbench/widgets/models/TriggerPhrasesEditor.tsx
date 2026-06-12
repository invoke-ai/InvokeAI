import { HStack, Icon, Input, Tag, Text, Wrap } from '@chakra-ui/react';
import { PlusIcon } from 'lucide-react';
import { useState } from 'react';

import { Button } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { updateModel } from '../../models/api';
import { replaceModelInStore } from '../../models/modelsStore';
import { triggerPhraseSchema } from '../../models/schemas';
import type { ModelConfig } from '../../models/types';

/**
 * Tag-style editor for a model's trigger phrases. Each add/remove persists
 * immediately — there is no separate save step for phrases.
 */
export const TriggerPhrasesEditor = ({
  model,
  onError,
}: {
  model: ModelConfig;
  onError: (message: string) => void;
}) => {
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const phrases = model.trigger_phrases ?? [];

  const persist = async (nextPhrases: string[]): Promise<boolean> => {
    setIsSaving(true);

    try {
      replaceModelInStore(await updateModel(model.key, { trigger_phrases: nextPhrases }));

      return true;
    } catch (persistError) {
      onError(persistError instanceof Error ? persistError.message : 'Failed to update trigger phrases.');

      return false;
    } finally {
      setIsSaving(false);
    }
  };

  const addPhrase = async () => {
    const parsed = triggerPhraseSchema.safeParse(draft);

    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? 'Invalid trigger phrase.');
      return;
    }

    if (phrases.some((phrase) => phrase.toLowerCase() === parsed.data.toLowerCase())) {
      setError('That trigger phrase is already on this model.');
      return;
    }

    setError(null);

    // Clear the draft only once it is saved, so a failure never eats the text.
    if (await persist([...phrases, parsed.data])) {
      setDraft('');
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
          size="sm"
          value={draft}
          onChange={(event) => {
            setDraft(event.currentTarget.value);
            setError(null);
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
          size="sm"
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
