import { FormControl, Textarea } from '@chakra-ui/react';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  handlePromptCheckers,
  setNegativePrompt,
} from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { ChangeEvent, useState } from 'react';

const NegativePromptInput = () => {
  const negativePrompt = useAppSelector(
    (state: RootState) => state.generation.negativePrompt
  );

  const [promptTimer, setPromptTimer] = useState<number | undefined>(undefined);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleNegativeChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) => {
    dispatch(setNegativePrompt(e.target.value));

    // Debounce Prompt UI Checking
    clearTimeout(promptTimer);
    const newPromptTimer = window.setTimeout(() => {
      dispatch(
        handlePromptCheckers({ prompt: e.target.value, toNegative: true })
      );
    }, 500);
    setPromptTimer(newPromptTimer);
  };

  return (
    <FormControl>
      <Textarea
        id="negativePrompt"
        name="negativePrompt"
        value={negativePrompt}
        onChange={handleNegativeChangePrompt}
        background="var(--prompt-bg-color)"
        placeholder={t('parameters.negativePrompts')}
        _placeholder={{ fontSize: '0.8rem' }}
        borderColor="var(--border-color)"
        _hover={{
          borderColor: 'var(--border-color-light)',
        }}
        _focusVisible={{
          borderColor: 'var(--border-color-invalid)',
          boxShadow: '0 0 10px var(--box-shadow-color-invalid)',
        }}
        fontSize="0.9rem"
        color="var(--text-color-secondary)"
      />
    </FormControl>
  );
};

export default NegativePromptInput;
