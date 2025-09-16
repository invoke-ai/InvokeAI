import {
  Button,
  Divider,
  Flex,
  IconButton,
  Input,
  Kbd,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Text,
  useShiftModifier,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import {
  positivePromptChanged,
  promptHistoryCleared,
  promptRemovedFromHistory,
  selectPositivePromptHistory,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { PiArrowArcLeftBold, PiClockCounterClockwise, PiTrashBold, PiTrashSimpleBold } from 'react-icons/pi';

export const PositivePromptHistoryIconButton = memo(() => {
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          size="sm"
          variant="promptOverlay"
          aria-label="Positive Prompt History"
          icon={<PiClockCounterClockwise />}
          tooltip="Prompt History"
        />
      </PopoverTrigger>
      <Portal>
        <PopoverContent>
          <PopoverBody maxH={300} maxW={400} h={300} w={400}>
            <PromptHistoryContent />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

PositivePromptHistoryIconButton.displayName = 'PositivePromptHistoryIconButton';

const PromptHistoryContent = memo(() => {
  const dispatch = useAppDispatch();
  const positivePromptHistory = useAppSelector(selectPositivePromptHistory);
  const [searchTerm, setSearchTerm] = useState('');

  const onClickClearHistory = useCallback(() => {
    dispatch(promptHistoryCleared());
  }, [dispatch]);

  const filteredPrompts = useMemo(() => {
    const trimmedSearchTerm = searchTerm.trim();
    if (!trimmedSearchTerm) {
      return positivePromptHistory;
    }
    return positivePromptHistory.filter((prompt) => prompt.toLowerCase().includes(trimmedSearchTerm.toLowerCase()));
  }, [positivePromptHistory, searchTerm]);

  const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex alignItems="center" gap={2} justifyContent="space-between">
        <Text fontWeight="semibold" color="base.300">
          Prompt History
        </Text>
        <Input
          size="sm"
          variant="outline"
          placeholder="Search..."
          value={searchTerm}
          onChange={onChangeSearchTerm}
          width="max-content"
          isDisabled={positivePromptHistory.length === 0}
        />
        <Button
          size="sm"
          variant="link"
          leftIcon={<PiTrashSimpleBold />}
          onClick={onClickClearHistory}
          isDisabled={positivePromptHistory.length === 0}
        >
          Clear History
        </Button>
      </Flex>
      <Divider />
      <Flex flexDir="column" flexGrow={1} minH={0}>
        {positivePromptHistory.length === 0 && (
          <Flex w="full" h="full" alignItems="center" justifyContent="center">
            <Text color="base.300">No prompt history recorded.</Text>
          </Flex>
        )}
        {positivePromptHistory.length !== 0 && filteredPrompts.length === 0 && (
          <Flex w="full" h="full" alignItems="center" justifyContent="center">
            <Text color="base.300">No matching prompts in history.</Text>{' '}
          </Flex>
        )}
        {filteredPrompts.length > 0 && (
          <ScrollableContent>
            <Flex flexDir="column">
              {filteredPrompts.map((prompt, index) => (
                <PromptItem key={`${prompt}-${index}`} prompt={prompt} />
              ))}
            </Flex>
          </ScrollableContent>
        )}
      </Flex>
      <Flex alignItems="center" justifyContent="center" pt={1}>
        <Text color="base.300" textAlign="center">
          <Kbd textTransform="lowercase">alt+up/down</Kbd> to switch between prompts.
        </Text>
      </Flex>
    </Flex>
  );
});
PromptHistoryContent.displayName = 'PromptHistoryContent';

const PromptItem = memo(({ prompt }: { prompt: string }) => {
  const dispatch = useAppDispatch();
  const shiftKey = useShiftModifier();

  const onClickUse = useCallback(() => {
    dispatch(positivePromptChanged(prompt));
  }, [dispatch, prompt]);

  const onClickDelete = useCallback(() => {
    dispatch(promptRemovedFromHistory(prompt));
  }, [dispatch, prompt]);

  return (
    <Flex gap={2}>
      {!shiftKey && (
        <IconButton
          size="sm"
          variant="ghost"
          aria-label="Use prompt"
          icon={<PiArrowArcLeftBold />}
          onClick={onClickUse}
        />
      )}
      {shiftKey && (
        <IconButton
          size="sm"
          variant="ghost"
          aria-label="Delete"
          icon={<PiTrashBold />}
          onClick={onClickDelete}
          colorScheme="error"
        />
      )}
      <Text color="base.300">{prompt}</Text>
    </Flex>
  );
});
PromptItem.displayName = 'PromptItem';
