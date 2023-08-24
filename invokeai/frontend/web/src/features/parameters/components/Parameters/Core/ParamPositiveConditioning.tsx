import { Box, FormControl, useDisclosure } from '@chakra-ui/react';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, KeyboardEvent, memo, useCallback, useRef } from 'react';

import { createSelector } from '@reduxjs/toolkit';
import {
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
import { flushSync } from 'react-dom';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { useFeatureStatus } from '../../../../system/hooks/useFeatureStatus';

const promptInputSelector = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ generation }, activeTabName) => {
    return {
      prompt: generation.positivePrompt,
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

  const handleSelectEmbedding = useCallback(
    (v: string) => {
      if (!promptRef.current) {
        return;
      }

      // this is where we insert the TI trigger
      const caret = promptRef.current.selectionStart;

      if (caret === undefined) {
        return;
      }

      let newPrompt = prompt.slice(0, caret);

      if (newPrompt[newPrompt.length - 1] !== '<') {
        newPrompt += '<';
      }

      newPrompt += `${v}>`;

      // we insert the cursor after the `>`
      const finalCaretPos = newPrompt.length;

      newPrompt += prompt.slice(caret);

      // must flush dom updates else selection gets reset
      flushSync(() => {
        dispatch(setPositivePrompt(newPrompt));
      });

      // set the caret position to just after the TI trigger
      promptRef.current.selectionStart = finalCaretPos;
      promptRef.current.selectionEnd = finalCaretPos;
      onClose();
    },
    [dispatch, onClose, prompt]
  );

  const isEmbeddingEnabled = useFeatureStatus('embedding').isFeatureEnabled;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey === false && isReady) {
        e.preventDefault();
        dispatch(clampSymmetrySteps());
        dispatch(userInvoked(activeTabName));
      }
      if (isEmbeddingEnabled && e.key === '<') {
        onOpen();
      }
    },
    [isReady, dispatch, activeTabName, onOpen, isEmbeddingEnabled]
  );

  // const handleSelect = (e: MouseEvent<HTMLTextAreaElement>) => {
  //   const target = e.target as HTMLTextAreaElement;
  // setCaret({ start: target.selectionStart, end: target.selectionEnd });
  // };

  return (
    <Box position="relative">
      <FormControl>
        <ParamEmbeddingPopover
          isOpen={isOpen}
          onClose={onClose}
          onSelect={handleSelectEmbedding}
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
      {!isOpen && isEmbeddingEnabled && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineEnd: 0,
          }}
        >
          <AddEmbeddingButton onClick={onOpen} />
        </Box>
      )}
    </Box>
  );
};

export default memo(ParamPositiveConditioning);
