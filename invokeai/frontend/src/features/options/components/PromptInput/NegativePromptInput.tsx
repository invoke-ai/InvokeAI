import { FormControl, Textarea } from '@chakra-ui/react';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setNegativePrompt } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export function NegativePromptInput() {
  const negativePrompt = useAppSelector(
    (state: RootState) => state.options.negativePrompt
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <FormControl>
      <Textarea
        id="negativePrompt"
        name="negativePrompt"
        value={negativePrompt}
        onChange={(e) => dispatch(setNegativePrompt(e.target.value))}
        background="var(--prompt-bg-color)"
        placeholder={t('options:negativePrompts')}
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
}
