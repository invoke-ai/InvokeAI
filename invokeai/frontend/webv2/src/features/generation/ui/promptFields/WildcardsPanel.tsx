/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { WildcardRecord } from '@features/generation/data/wildcards';
import type { WildcardCatalog } from '@features/generation/ui/useWildcards';
import type { ChangeEvent } from 'react';

import { HStack, Input, Stack, Text } from '@chakra-ui/react';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { Button, IconButton } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { CheckIcon, PencilIcon, PlusIcon, TrashIcon, XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** One value per line: the same shape the user is editing a variant in. */
const toValuesText = (values: string[]): string => values.join('\n');
const fromValuesText = (text: string): string[] => text.split('\n');

interface WildcardDraft {
  id: string | null;
  name: string;
  valuesText: string;
}

export const WildcardsPanel = ({
  catalog,
  onInsert,
  showSyntaxHighlighting,
}: {
  catalog: WildcardCatalog;
  showSyntaxHighlighting: boolean;
  /** Splices `__name__` into the prompt at the caret. */
  onInsert: (reference: string) => void;
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<WildcardDraft | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startCreate = useCallback(() => {
    setError(null);
    setDraft({ id: null, name: '', valuesText: '' });
  }, []);

  const startEdit = useCallback((wildcard: WildcardRecord) => {
    setError(null);
    setDraft({ id: wildcard.id, name: wildcard.name, valuesText: toValuesText(wildcard.values) });
  }, []);

  const cancel = useCallback(() => {
    setError(null);
    setDraft(null);
  }, []);

  const save = useCallback(async () => {
    if (!draft) {
      return;
    }

    const values = fromValuesText(draft.valuesText);

    try {
      if (draft.id === null) {
        await catalog.create({ name: draft.name, values });
      } else {
        await catalog.update(draft.id, { name: draft.name, values });
      }
      setDraft(null);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t('widgets.generate.dynamicPrompts.couldNotSaveWildcard'));
    }
  }, [catalog, draft, t]);

  if (draft) {
    return (
      <Stack gap="2">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
          {draft.id === null
            ? t('widgets.generate.dynamicPrompts.newWildcard')
            : t('widgets.generate.dynamicPrompts.editWildcard')}
        </Text>
        <Input
          aria-label={t('widgets.generate.dynamicPrompts.wildcardName')}
          placeholder={t('widgets.generate.dynamicPrompts.wildcardNamePlaceholder')}
          size="xs"
          value={draft.name}
          onChange={(event: ChangeEvent<HTMLInputElement>) => setDraft({ ...draft, name: event.currentTarget.value })}
        />
        {/* Values are expanded by dynamicprompts too, so a nested `{a|b}` or
            `__other__` is live syntax here and is coloured as such. The gutter
            numbers the values, which is what a line means in this editor. */}
        <PromptTextarea
          aria-label={t('widgets.generate.dynamicPrompts.wildcardValues')}
          defaultHeightPx={144}
          fontSize="0.72rem"
          highlightDynamicPrompts
          knownWildcards={catalog.knownNames}
          maxHeightPx={320}
          minHeightPx={96}
          placeholder={t('widgets.generate.dynamicPrompts.wildcardValuesPlaceholder')}
          resizeHandleAriaLabel={t('widgets.generate.dynamicPrompts.resizeWildcardValues')}
          showLineNumbers
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="xs"
          value={draft.valuesText}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            setDraft({ ...draft, valuesText: event.currentTarget.value })
          }
        />
        {error ? (
          <Text color="fg.error" fontSize="2xs" wordBreak="break-word">
            {error}
          </Text>
        ) : null}
        <HStack justify="end">
          <Button size="xs" variant="ghost" onClick={cancel}>
            <XIcon />
            {t('common.cancel')}
          </Button>
          <Button disabled={!draft.name.trim()} size="xs" onClick={() => void save()}>
            <CheckIcon />
            {t('common.save')}
          </Button>
        </HStack>
      </Stack>
    );
  }

  return (
    <Stack gap="2">
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
          {t('widgets.generate.dynamicPrompts.wildcards')}
        </Text>
        <Button size="2xs" variant="ghost" onClick={startCreate}>
          <PlusIcon />
          {t('widgets.generate.dynamicPrompts.newWildcard')}
        </Button>
      </HStack>

      <Scrollable h="14rem" label={t('widgets.generate.dynamicPrompts.wildcards')}>
        {catalog.wildcards.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs" px="2" py="1.5">
            {t('widgets.generate.dynamicPrompts.noWildcardsYet')}
          </Text>
        ) : (
          <Stack gap="0">
            {catalog.wildcards.map((wildcard) => (
              <WildcardRow
                key={wildcard.id}
                catalog={catalog}
                wildcard={wildcard}
                onEdit={startEdit}
                onInsert={onInsert}
              />
            ))}
          </Stack>
        )}
      </Scrollable>
    </Stack>
  );
};

const WildcardRow = ({
  catalog,
  onEdit,
  onInsert,
  wildcard,
}: {
  catalog: WildcardCatalog;
  wildcard: WildcardRecord;
  onEdit: (wildcard: WildcardRecord) => void;
  onInsert: (reference: string) => void;
}) => {
  const { t } = useTranslation();

  return (
    <HStack align="start" gap="1" pr="1">
      <Button
        alignItems="start"
        flex="1"
        h="auto"
        justifyContent="start"
        minW="0"
        px="2"
        py="1.5"
        size="xs"
        title={t('widgets.generate.dynamicPrompts.insertWildcard')}
        variant="ghost"
        onClick={() => onInsert(`__${wildcard.name}__`)}
      >
        <Stack align="start" gap="0" minW="0">
          <Text as="span" color="fg" fontFamily="mono" fontSize="0.72rem">
            __{wildcard.name}__
          </Text>
          <Text as="span" color="fg.subtle" fontSize="2xs" truncate>
            {wildcard.values.length > 0
              ? wildcard.values.join(', ')
              : t('widgets.generate.dynamicPrompts.wildcardHasNoValues')}
          </Text>
        </Stack>
      </Button>
      <Tooltip content={t('common.edit')}>
        <IconButton aria-label={t('common.edit')} size="2xs" variant="ghost" onClick={() => onEdit(wildcard)}>
          <PencilIcon />
        </IconButton>
      </Tooltip>
      <Tooltip content={t('common.delete')}>
        <IconButton
          aria-label={t('common.delete')}
          colorPalette="red"
          size="2xs"
          variant="ghost"
          onClick={() => void catalog.remove(wildcard.id)}
        >
          <TrashIcon />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};
