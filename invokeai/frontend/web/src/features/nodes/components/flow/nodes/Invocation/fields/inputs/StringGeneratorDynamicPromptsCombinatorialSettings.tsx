import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { GeneratorTextareaWithFileUpload } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/GeneratorTextareaWithFileUpload';
import type { StringGeneratorDynamicPromptsCombinatorial } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDynamicPromptsQuery } from 'services/api/endpoints/utilities';
import { useDebounce } from 'use-debounce';

type StringGeneratorDynamicPromptsCombinatorialSettingsProps = {
  state: StringGeneratorDynamicPromptsCombinatorial;
  onChange: (state: StringGeneratorDynamicPromptsCombinatorial) => void;
};
export const StringGeneratorDynamicPromptsCombinatorialSettings = memo(
  ({ state, onChange }: StringGeneratorDynamicPromptsCombinatorialSettingsProps) => {
    const { t } = useTranslation();
    const loadingValues = useMemo(() => [`<${t('nodes.generatorLoading')}>`], [t]);

    const onChangeInput = useCallback(
      (input: string) => {
        onChange({ ...state, input, values: loadingValues });
      },
      [onChange, state, loadingValues]
    );
    const onChangeMaxPrompts = useCallback(
      (v: number) => {
        onChange({ ...state, maxPrompts: v, values: loadingValues });
      },
      [onChange, state, loadingValues]
    );

    const arg = useMemo(() => {
      return { prompt: state.input, max_prompts: state.maxPrompts, combinatorial: true };
    }, [state]);
    const [debouncedArg] = useDebounce(arg, 300);

    const { data, isLoading } = useDynamicPromptsQuery(debouncedArg);

    useEffect(() => {
      if (isLoading) {
        return;
      }

      if (!data) {
        onChange({ ...state, values: [] });
        return;
      }

      onChange({ ...state, values: data.prompts });
    }, [data, isLoading, onChange, state]);

    return (
      <Flex gap={2} flexDir="column">
        <FormControl orientation="vertical">
          <FormLabel>{t('dynamicPrompts.maxPrompts')}</FormLabel>
          <CompositeNumberInput value={state.maxPrompts} onChange={onChangeMaxPrompts} min={1} max={1000} w="full" />
        </FormControl>
        <GeneratorTextareaWithFileUpload value={state.input} onChange={onChangeInput} />
      </Flex>
    );
  }
);
StringGeneratorDynamicPromptsCombinatorialSettings.displayName = 'StringGeneratorDynamicPromptsCombinatorialSettings';
