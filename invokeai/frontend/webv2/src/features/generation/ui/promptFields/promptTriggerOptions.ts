/**
 * What `<` and `__` can insert: wildcards, the selected model's and each enabled
 * LoRA's trigger phrases, and embeddings compatible with the base.
 *
 * Shared by the two ways of reaching that list — the `+` button's browsable
 * popover and the caret-anchored autocomplete — so the two can never drift into
 * offering different things.
 */

import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';

import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export interface PromptTriggerOption {
  group: string;
  label: string;
  value: string;
}

export interface PromptTriggerOptionGroup {
  group: string;
  options: PromptTriggerOption[];
}

const getTriggerPhrases = (model: unknown): string[] => {
  if (!model || typeof model !== 'object') {
    return [];
  }

  const triggerPhrases = (model as { trigger_phrases?: unknown }).trigger_phrases;

  return Array.isArray(triggerPhrases)
    ? triggerPhrases.filter((phrase): phrase is string => typeof phrase === 'string' && phrase.trim().length > 0)
    : [];
};

export const getPromptTriggerOptions = ({
  compatibleEmbeddingsLabel,
  loras,
  mainModelLabel,
  models,
  selectedModel,
  wildcards,
  wildcardsLabel,
}: {
  compatibleEmbeddingsLabel: string;
  loras: GenerateLora[];
  mainModelLabel: string;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  wildcards: readonly { name: string; values: string[] }[];
  wildcardsLabel: string;
}): PromptTriggerOption[] => {
  const options: PromptTriggerOption[] = [];

  // A wildcard inserts as `__name__`, which is just another trigger the picker
  // can splice in, so it rides in the same list as embeddings and LoRA phrases.
  for (const wildcard of wildcards) {
    options.push({ group: wildcardsLabel, label: wildcard.name, value: `__${wildcard.name}__` });
  }

  for (const phrase of getTriggerPhrases(selectedModel)) {
    options.push({ group: selectedModel?.name ?? mainModelLabel, label: phrase, value: phrase });
  }

  for (const lora of loras) {
    if (!lora.isEnabled) {
      continue;
    }

    for (const phrase of getTriggerPhrases(lora.model)) {
      options.push({ group: lora.model.name, label: phrase, value: phrase });
    }
  }

  if (selectedModel) {
    for (const model of models) {
      if (model.type === 'embedding' && model.base === selectedModel.base) {
        options.push({ group: compatibleEmbeddingsLabel, label: model.name, value: `<${model.name}>` });
      }
    }
  }

  const seen = new Set<string>();

  return options.filter((option) => {
    const key = `${option.group}:${option.value}`;

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
};

export const filterPromptTriggerOptions = (
  options: readonly PromptTriggerOption[],
  search: string
): PromptTriggerOption[] => {
  const query = search.trim().toLowerCase();

  return options.filter(
    (option) => !query || option.label.toLowerCase().includes(query) || option.group.toLowerCase().includes(query)
  );
};

export const groupPromptTriggerOptions = (options: readonly PromptTriggerOption[]): PromptTriggerOptionGroup[] =>
  options.reduce<PromptTriggerOptionGroup[]>((groups, option) => {
    const existing = groups.find((group) => group.group === option.group);

    if (existing) {
      existing.options.push(option);
    } else {
      groups.push({ group: option.group, options: [option] });
    }

    return groups;
  }, []);

/** The live option list for a field, already localized. */
export const usePromptTriggerOptions = (
  loras: GenerateLora[],
  selectedModel: GenerateModelConfig | undefined
): PromptTriggerOption[] => {
  const { t } = useTranslation();
  const { catalog: models } = useGenerationUi().models;
  const { wildcards } = useWildcards();

  return useMemo(
    () =>
      getPromptTriggerOptions({
        compatibleEmbeddingsLabel: t('widgets.generate.compatibleEmbeddings'),
        loras,
        mainModelLabel: t('widgets.generate.mainModel'),
        models,
        selectedModel,
        wildcards,
        wildcardsLabel: t('widgets.generate.dynamicPrompts.wildcards'),
      }),
    [loras, models, selectedModel, t, wildcards]
  );
};
