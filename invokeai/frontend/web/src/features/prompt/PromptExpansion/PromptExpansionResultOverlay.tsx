import { ButtonGroup, Flex, Icon, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { positivePromptChanged, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { useCallback } from 'react';
import { PiCheckBold, PiMagicWandBold, PiPlusBold, PiXBold } from 'react-icons/pi';

import { promptExpansionApi } from './state';

interface PromptExpansionResultOverlayProps {
  expandedText: string;
}

export const PromptExpansionResultOverlay = ({ expandedText }: PromptExpansionResultOverlayProps) => {
  const dispatch = useAppDispatch();
  const positivePrompt = useAppSelector(selectPositivePrompt);

  const handleReplace = useCallback(() => {
    dispatch(positivePromptChanged(expandedText));
    promptExpansionApi.reset();
  }, [dispatch, expandedText]);

  const handleInsert = useCallback(() => {
    const currentText = positivePrompt;
    const newText = currentText ? `${currentText}\n${expandedText}` : expandedText;
    dispatch(positivePromptChanged(newText));
    promptExpansionApi.reset();
  }, [dispatch, expandedText, positivePrompt]);

  const handleDiscard = useCallback(() => {
    promptExpansionApi.reset();
  }, []);

  return (
    <Flex pos="absolute" inset={0} bg="base.800" backdropFilter="blur(8px)" zIndex={10} direction="column">
      <Flex flex={1} p={2} borderRadius="md" overflowY="auto" minH={0}>
        <Text fontSize="sm" w="full" pr={7}>
          <Icon as={PiMagicWandBold} boxSize={5} display="inline" mr={2} color="invokeYellow.500" />
          {expandedText}
        </Text>
      </Flex>

      <Flex gap={2} p={1} justify="flex-end" pos="absolute" bottom={0} right={0} flexDirection="column">
        <ButtonGroup orientation="vertical">
          <Tooltip label="Replace" placement="right">
            <IconButton
              onClick={handleReplace}
              icon={<PiCheckBold />}
              colorScheme="invokeGreen"
              size="xs"
              aria-label="Replace"
            />
          </Tooltip>

          <Tooltip label="Insert" placement="right">
            <IconButton
              onClick={handleInsert}
              icon={<PiPlusBold />}
              colorScheme="invokeBlue"
              size="xs"
              aria-label="Insert"
            />
          </Tooltip>
        </ButtonGroup>
        <Tooltip label="Discard" placement="right">
          <IconButton
            onClick={handleDiscard}
            icon={<PiXBold />}
            colorScheme="invokeRed"
            size="xs"
            aria-label="Discard"
          />
        </Tooltip>
      </Flex>
    </Flex>
  );
};
