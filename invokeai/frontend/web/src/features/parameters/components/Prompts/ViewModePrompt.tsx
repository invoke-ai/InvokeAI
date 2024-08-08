import { Box, Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { viewModeChanged } from 'features/stylePresets/store/stylePresetSlice';
import { getViewModeChunks } from 'features/stylePresets/util/getViewModeChunks';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold } from 'react-icons/pi';

export const ViewModePrompt = ({
  presetPrompt,
  prompt,
  height,
  onExit,
}: {
  presetPrompt: string;
  prompt: string;
  height: number;
  onExit: () => void;
}) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const presetChunks = useMemo(() => {
    return getViewModeChunks(prompt, presetPrompt);
  }, [presetPrompt, prompt]);

  const handleExitViewMode = useCallback(() => {
    dispatch(viewModeChanged(false));
    onExit();
  }, [dispatch, onExit]);

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
                <Text as="span" color={index === 1 ? 'white' : 'base.300'} key={index}>
                  {chunk.trim()}{' '}
                </Text>
              )
            );
          })}
        </Text>
      </Flex>

      <Box position="absolute" top={0} right={0} backgroundColor="rgba(0,0,0,0.75)" padding="2px 5px">
        <Flex alignItems="center" gap="1">
          <Tooltip label={t('stylePresets.viewModeTooltip')}>
            <Flex>
              <Icon as={PiEyeBold} color="base.500" boxSize="12px" />
            </Flex>
          </Tooltip>
        </Flex>
      </Box>
    </Flex>
  );
};
