import { Box, FormControl, useDisclosure } from '@chakra-ui/react';
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
import IAITextarea from 'common/components/IAITextarea';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import AddEmbeddingButton from 'features/embedding/components/AddEmbeddingButton';
import ParamEmbeddingPopover from 'features/embedding/components/ParamEmbeddingPopover';
import { isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

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
  const { isOpen, onClose, onOpen } = useDisclosure();
  const { t } = useTranslation();

  const handleChangePrompt = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(setPositivePrompt(e.target.value));
    },
    [dispatch]
  );

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
      if (e.key === '<') {
        onOpen();
      }
    },
    [isReady, dispatch, activeTabName, onOpen]
  );

  const handleSelect = useCallback(
    (v: string) => {
      const caret = promptRef.current?.selectionStart;

      if (caret === undefined) {
        return;
      }

      let newPrompt = prompt.slice(0, caret);

      if (newPrompt[newPrompt.length - 1] !== '<') {
        newPrompt += '<';
      }

      newPrompt += `${v}>`;
      newPrompt += prompt.slice(caret);

      dispatch(setPositivePrompt(newPrompt));
    },
    [dispatch, prompt]
  );

  return (
    <Box>
      <FormControl>
        <ParamEmbeddingPopover
          isOpen={isOpen}
          onClose={onClose}
          onSelect={handleSelect}
        >
          <IAITextarea
            id="prompt"
            name="prompt"
            ref={promptRef}
            value={prompt}
            placeholder={t('parameters.positivePromptPlaceholder')}
            onChange={handleChangePrompt}
            onKeyDown={handleKeyDown}
            resize="vertical"
            minH={32}
          />
        </ParamEmbeddingPopover>
      </FormControl>
      {!isOpen && (
        <Box
          sx={{
            position: 'absolute',
            top: 6,
            insetInlineEnd: 0,
          }}
        >
          <AddEmbeddingButton onClick={onOpen} />
        </Box>
      )}
    </Box>
  );
};

export default ParamPositiveConditioning;
