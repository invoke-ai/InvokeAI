import { Box, FormControl } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, KeyboardEvent, useCallback, useRef } from 'react';

import { createSelector } from '@reduxjs/toolkit';
import {
  GenerationState,
  clampSymmetrySteps,
  setPositivePrompt,
} from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

import { userInvoked } from 'app/store/actions';
import IAIIconButton from 'common/components/IAIIconButton';
import IAITextarea from 'common/components/IAITextarea';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import { toggleEmbeddingPicker } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { BiCode } from 'react-icons/bi';

const promptInputSelector = createSelector(
  [(state: RootState) => state.generation, activeTabNameSelector],
  (parameters: GenerationState, activeTabName) => {
    return {
      prompt: parameters.positivePrompt,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Prompt input text area.
 */
const ParamPositiveConditioning = () => {
  const dispatch = useAppDispatch();
  const { prompt, activeTabName } = useAppSelector(promptInputSelector);
  const isReady = useIsReadyToInvoke();

  const promptRef = useRef<HTMLTextAreaElement>(null);

  const { t } = useTranslation();

  const handleChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) => {
    dispatch(setPositivePrompt(e.target.value));
  };

  useHotkeys(
    'alt+a',
    () => {
      promptRef.current?.focus();
    },
    []
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey === false && isReady) {
        e.preventDefault();
        dispatch(clampSymmetrySteps());
        dispatch(userInvoked(activeTabName));
      }
    },
    [dispatch, activeTabName, isReady]
  );

  const shouldShowEmbeddingPicker = useAppSelector(
    (state: RootState) => state.ui.shouldShowEmbeddingPicker
  );

  return (
    <Box>
      <IAIIconButton
        size="xs"
        aria-label="Toggle Embedding Picker"
        tooltip="Toggle Embedding Picker"
        icon={<BiCode />}
        sx={{ position: 'absolute', top: 8, right: 2, zIndex: 2 }}
        isChecked={shouldShowEmbeddingPicker}
        onClick={() => dispatch(toggleEmbeddingPicker())}
      ></IAIIconButton>
      <FormControl>
        <IAITextarea
          id="prompt"
          name="prompt"
          placeholder={t('parameters.positivePromptPlaceholder')}
          value={prompt}
          onChange={handleChangePrompt}
          onKeyDown={handleKeyDown}
          resize="vertical"
          ref={promptRef}
          minH={32}
          paddingRight={8}
        />
      </FormControl>
    </Box>
  );
};

export default ParamPositiveConditioning;
