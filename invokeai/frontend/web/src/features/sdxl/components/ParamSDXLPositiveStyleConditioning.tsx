import { Box, FormControl, useDisclosure } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import AddEmbeddingButton from 'features/embedding/components/AddEmbeddingButton';
import ParamEmbeddingPopover from 'features/embedding/components/ParamEmbeddingPopover';
import { setPositiveStylePromptSDXL } from 'features/sdxl/store/sdxlSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { AnimatePresence } from 'framer-motion';
import { ChangeEvent, KeyboardEvent, memo, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import SDXLConcatLink from './SDXLConcatLink';

const promptInputSelector = createMemoizedSelector(
  [stateSelector],
  ({ sdxl }) => {
    const { positiveStylePrompt, shouldConcatSDXLStylePrompt } = sdxl;

    return {
      prompt: positiveStylePrompt,
      shouldConcatSDXLStylePrompt,
    };
  }
);

/**
 * Prompt input text area.
 */
const ParamSDXLPositiveStyleConditioning = () => {
  const dispatch = useAppDispatch();
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const { isOpen, onClose, onOpen } = useDisclosure();
  const { t } = useTranslation();

  const { prompt, shouldConcatSDXLStylePrompt } =
    useAppSelector(promptInputSelector);

  const handleChangePrompt = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(setPositiveStylePromptSDXL(e.target.value));
    },
    [dispatch]
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
        dispatch(setPositiveStylePromptSDXL(newPrompt));
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
      if (isEmbeddingEnabled && e.key === '<') {
        onOpen();
      }
    },
    [onOpen, isEmbeddingEnabled]
  );

  return (
    <Box position="relative">
      <AnimatePresence>
        {shouldConcatSDXLStylePrompt && (
          <Box
            sx={{
              position: 'absolute',
              left: '3',
              w: '94%',
              top: '-17px',
            }}
          >
            <SDXLConcatLink />
          </Box>
        )}
      </AnimatePresence>
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
            placeholder={t('sdxl.posStylePrompt')}
            onChange={handleChangePrompt}
            onKeyDown={handleKeyDown}
            resize="vertical"
            minH={16}
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

export default memo(ParamSDXLPositiveStyleConditioning);
