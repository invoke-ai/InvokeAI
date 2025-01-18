import { Checkbox, CompositeNumberInput, Flex, FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { LoadTextFromFileIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/LoadTextFromFileIconButton';
import type { StringGeneratorDynamicPromptsRandom } from 'features/nodes/types/field';
import { isNil, random } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDynamicPromptsQuery } from 'services/api/endpoints/utilities';
import { useDebounce } from 'use-debounce';

type StringGeneratorDynamicPromptsRandomSettingsProps = {
  state: StringGeneratorDynamicPromptsRandom;
  onChange: (state: StringGeneratorDynamicPromptsRandom) => void;
};
export const StringGeneratorDynamicPromptsRandomSettings = memo(
  ({ state, onChange }: StringGeneratorDynamicPromptsRandomSettingsProps) => {
    const { t } = useTranslation();
    const loadingValues = useMemo(() => [`<${t('nodes.generatorLoading')}>`], [t]);

    const onChangeInput = useCallback(
      (e: ChangeEvent<HTMLTextAreaElement>) => {
        onChange({ ...state, input: e.target.value, values: loadingValues });
      },
      [onChange, state, loadingValues]
    );
    const onChangeCount = useCallback(
      (v: number) => {
        onChange({ ...state, count: v, values: loadingValues });
      },
      [onChange, state, loadingValues]
    );
    const onToggleSeed = useCallback(() => {
      onChange({ ...state, seed: isNil(state.seed) ? 0 : null, values: loadingValues });
    }, [onChange, state, loadingValues]);
    const onChangeSeed = useCallback(
      (seed?: number | null) => {
        onChange({ ...state, seed, values: loadingValues });
      },
      [onChange, state, loadingValues]
    );

    const onLoadFile = useCallback(
      (value: string) => {
        onChange({ ...state, input: value });
      },
      [onChange, state]
    );

    const arg = useMemo(() => {
      const { input, count, seed } = state;
      return { prompt: input, max_prompts: count, combinatorial: false, seed: seed ?? random() };
    }, [state]);
    const [debouncedArg] = useDebounce(arg, 300);

    const { data, isLoading } = useDynamicPromptsQuery(debouncedArg);

    useEffect(() => {
      if (isLoading) {
        onChange({ ...state, values: loadingValues });
      } else if (data) {
        onChange({ ...state, values: data.prompts });
      } else {
        onChange({ ...state, values: [] });
      }
    }, [data, isLoading, loadingValues, onChange, state]);

    return (
      <Flex gap={2} flexDir="column">
        <Flex gap={2}>
          <FormControl orientation="vertical">
            <FormLabel alignItems="center" justifyContent="space-between" display="flex" w="full" pe={0.5}>
              {t('common.seed')}
              <Checkbox onChange={onToggleSeed} isChecked={!isNil(state.seed)} />
            </FormLabel>
            <CompositeNumberInput
              isDisabled={isNil(state.seed)}
              // This cast is save only because we disable the element when seed is not a number - the `...` is
              // rendered in the input field in this case
              value={state.seed ?? ('...' as unknown as number)}
              onChange={onChangeSeed}
              min={-Infinity}
              max={Infinity}
            />
          </FormControl>
          <FormControl orientation="vertical">
            <FormLabel>{t('common.count')}</FormLabel>
            <CompositeNumberInput value={state.count} onChange={onChangeCount} min={1} max={1000} />
          </FormControl>
        </Flex>
        <FormControl orientation="vertical" position="relative">
          <FormLabel>{t('common.input')}</FormLabel>
          <Textarea
            className="nowheel nodrag nopan"
            value={state.input}
            onChange={onChangeInput}
            p={2}
            resize="none"
            rows={5}
            fontSize="sm"
          />
          <LoadTextFromFileIconButton position="absolute" top={10} right={2} onLoadFile={onLoadFile} />
        </FormControl>
      </Flex>
    );
  }
);
StringGeneratorDynamicPromptsRandomSettings.displayName = 'StringGeneratorDynamicPromptsRandomSettings';
