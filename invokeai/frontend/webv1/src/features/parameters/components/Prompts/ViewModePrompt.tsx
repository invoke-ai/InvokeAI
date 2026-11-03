import { Box, Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { viewModeChanged } from 'features/stylePresets/store/stylePresetSlice';
import { getViewModeChunks } from 'features/stylePresets/util/getViewModeChunks';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold } from 'react-icons/pi';

import { PromptLabel } from './PromptLabel';

export const ViewModePrompt = ({
  presetPrompt,
  prompt,
  label,
}: {
  presetPrompt: string;
  prompt: string;
  label: string;
}) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const presetChunks = useMemo(() => {
    return getViewModeChunks(prompt, presetPrompt);
  }, [presetPrompt, prompt]);

  const handleExitViewMode = useCallback(() => {
    dispatch(viewModeChanged(false));
  }, [dispatch]);

  return (
    <Box position="absolute" top={0} bottom={0} left={0} right={0} layerStyle="second" borderRadius="base">
      <Flex
        flexDir="column"
        onClick={handleExitViewMode}
        justifyContent="space-between"
        h="full"
        borderWidth={1}
        borderTopWidth={24} // This prevents the prompt from being hidden behind the header
        borderColor="transparent"
        paddingInlineEnd={10}
        paddingInlineStart={3}
        paddingTop={0}
        paddingBottom={3}
      >
        <PromptLabel label={label} />
        <Flex overflow="scroll">
          <Text w="full" lineHeight="short">
            {presetChunks.map((chunk, index) => (
              <Text
                as="span"
                color={index === 1 ? 'white' : 'base.200'}
                fontWeight={index === 1 ? 'semibold' : 'normal'}
                key={index}
              >
                {chunk}
              </Text>
            ))}
          </Text>
        </Flex>

        <Tooltip label={t('stylePresets.viewModeTooltip')}>
          <Flex
            position="absolute"
            insetInlineEnd={0}
            insetBlockStart={0}
            alignItems="center"
            justifyContent="center"
            p={2}
            bg="base.900"
            opacity={0.8}
            backgroundClip="border-box"
            borderBottomStartRadius="base"
          >
            <Icon as={PiEyeBold} color="base.500" boxSize={4} />
          </Flex>
        </Tooltip>
      </Flex>
    </Box>
  );
};
