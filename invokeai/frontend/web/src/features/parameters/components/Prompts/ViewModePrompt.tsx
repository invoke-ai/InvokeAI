import { Flex, Icon, Text, Tooltip, Box } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { useCallback, useMemo } from 'react';
import { PiEyeBold, PiQuestionBold } from 'react-icons/pi';
import { viewModeChanged } from '../../../stylePresets/store/stylePresetSlice';
import { getViewModeChunks } from '../../../stylePresets/util/getViewModeChunks';

export const ViewModePrompt = ({
  presetPrompt,
  prompt,
  height,
}: {
  presetPrompt: string;
  prompt: string;
  height: number;
}) => {
  const dispatch = useAppDispatch();

  const presetChunks = useMemo(() => {
    return getViewModeChunks(prompt, presetPrompt);
  }, [presetPrompt, prompt]);

  const handleExitViewMode = useCallback(() => {
    dispatch(viewModeChanged(false));
  }, [dispatch]);

  return (
    <Flex
      flexDir="column"
      layerStyle="second"
      padding="8px 10px"
      borderRadius="base"
      height={height}
      onClick={handleExitViewMode}
      justifyContent="space-between"
      position="relative"
    >
      <Flex overflow="scroll">
        <Text fontSize="sm" lineHeight="1rem">
          {presetChunks.map((chunk, index) => {
            return (
              chunk && (
                <Text as="span" color={index === 1 ? 'white' : 'base.300'}>
                  {chunk.trim()}{' '}
                </Text>
              )
            );
          })}
        </Text>
      </Flex>

      <Box position="absolute" top={0} right={0} backgroundColor="rgba(0,0,0,0.75)" padding="2px 5px">
        <Flex alignItems="center" gap="1">
          <Tooltip
            label={
              'This is how your prompt will look with your currently selected preset. To edit your prompt, click anywhere in the text box.'
            }
          >
            <Flex>
              <Icon as={PiEyeBold} color="base.500" boxSize="12px" />
            </Flex>
          </Tooltip>
        </Flex>
      </Box>
    </Flex>
  );
};
