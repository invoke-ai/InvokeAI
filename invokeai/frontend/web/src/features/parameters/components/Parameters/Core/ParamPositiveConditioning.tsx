import { Box, FormControl } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, KeyboardEvent, useCallback, useRef } from 'react';

import { createSelector } from '@reduxjs/toolkit';
import { readinessSelector } from 'app/selectors/readinessSelector';
import {
  GenerationState,
  clampSymmetrySteps,
  setPositivePrompt,
} from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

import { isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { userInvoked } from 'app/store/actions';
import IAITextarea from 'common/components/IAITextarea';

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
  const { isReady } = useAppSelector(readinessSelector);

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

  return (
    <Box>
      <FormControl
        isInvalid={prompt.length === 0 || Boolean(prompt.match(/^[\s\r\n]+$/))}
      >
        <IAITextarea
          id="prompt"
          name="prompt"
          placeholder={t('parameters.promptPlaceholder')}
          value={prompt}
          onChange={handleChangePrompt}
          onKeyDown={handleKeyDown}
          resize="vertical"
          ref={promptRef}
          minH={{ base: 20, lg: 40 }}
        />
      </FormControl>
    </Box>
  );
};

export default ParamPositiveConditioning;
