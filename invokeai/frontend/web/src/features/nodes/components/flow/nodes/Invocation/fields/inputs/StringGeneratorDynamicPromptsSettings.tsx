import {
  Checkbox,
  CompositeNumberInput,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Textarea,
} from '@invoke-ai/ui-library';
import { getStore } from 'app/store/nanostores/store';
import type { StringGeneratorDynamicPrompts } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShuffleSimpleBold } from 'react-icons/pi';
import { utilitiesApi } from 'services/api/endpoints/utilities';
import { useDebounce } from 'use-debounce';

const processDynamicPrompts = async (state: StringGeneratorDynamicPrompts) => {
  const { input, maxPrompts, combinatorial } = state;
  const { dispatch } = getStore();
  const req = dispatch(
    utilitiesApi.endpoints.dynamicPrompts.initiate(
      { prompt: input, max_prompts: maxPrompts, combinatorial },
      { subscribe: false }
    )
  );
  try {
    const { prompts } = await req.unwrap();
    return prompts;
  } catch {
    return [];
  }
};

type StringGeneratorDynamicPromptsSettingsProps = {
  state: StringGeneratorDynamicPrompts;
  onChange: (state: StringGeneratorDynamicPrompts) => void;
};
export const StringGeneratorDynamicPromptsSettings = memo(
  ({ state, onChange }: StringGeneratorDynamicPromptsSettingsProps) => {
    const { t } = useTranslation();

    const onChangeInput = useCallback(
      (e: ChangeEvent<HTMLTextAreaElement>) => {
        onChange({ ...state, input: e.target.value });
      },
      [onChange, state]
    );

    const onChangeMaxPrompts = useCallback(
      (v: number) => {
        onChange({ ...state, maxPrompts: v });
      },
      [onChange, state]
    );

    const onChangeCombinatorial = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        onChange({ ...state, combinatorial: e.target.checked });
      },
      [onChange, state]
    );

    const [debouncedState] = useDebounce(state, 1000);

    useEffect(() => {
      processDynamicPrompts(debouncedState).then((prompts) => {
        onChange({ ...debouncedState, values: prompts });
      });
    }, [onChange, debouncedState]);

    const reroll = useCallback(() => {
      processDynamicPrompts(debouncedState).then((prompts) => {
        onChange({ ...debouncedState, values: prompts });
      });
    }, [debouncedState, onChange]);

    return (
      <Flex gap={2} flexDir="column">
        <Flex gap={2}>
          <FormControl orientation="vertical">
            <FormLabel>Max Prompts</FormLabel>
            {/* <FormLabel>{t('nodes.splitOn')}</FormLabel> */}
            <CompositeNumberInput value={state.maxPrompts} onChange={onChangeMaxPrompts} min={1} max={1000} />
          </FormControl>
          <FormControl orientation="vertical">
            <FormLabel>Combinatorial</FormLabel>
            {/* <FormLabel>{t('nodes.splitOn')}</FormLabel> */}
            <Checkbox isChecked={state.combinatorial} onChange={onChangeCombinatorial} />
          </FormControl>
          <IconButton
            aria-label="Reroll"
            isDisabled={state.combinatorial}
            onClick={reroll}
            icon={<PiShuffleSimpleBold />}
            variant="ghost"
          />
        </Flex>
        <FormControl orientation="vertical">
          <FormLabel>{t('common.input')}</FormLabel>
          <Textarea
            className="nowheel nodrag nopan"
            value={state.input}
            onChange={onChangeInput}
            p={2}
            resize="none"
            rows={5}
          />
        </FormControl>
      </Flex>
    );
  }
);
StringGeneratorDynamicPromptsSettings.displayName = 'StringGeneratorDynamicPromptsSettings';
