import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import type { DynamicPromptsExpansion } from '@features/generation/ui/useDynamicPrompts';

import { Box, Flex, HStack, Icon, Text } from '@chakra-ui/react';
import { describeDynamicPromptsFailure, findDynamicPromptComment } from '@features/generation/core/dynamicPrompts';
import { describeGenerationIssue } from '@features/generation/core/generationIssues';
import { Button } from '@platform/ui/Button';
import { CircleAlertIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Under the prompt, while dynamic prompts are on: why its expansion cannot be used (Invoke waits on it), else that a
 * comment marker will drop text. Each offers the way out, turning dynamic prompts off, and a failed request a retry.
 * The live region stays mounted, empty when there is nothing to say, so screen readers announce what appears in it.
 */
export const DynamicPromptsNotice = ({
  config,
  expansion,
  prompt,
}: {
  config: DynamicPromptsFieldConfig;
  expansion: DynamicPromptsExpansion;
  /** The prompt as it expands: the authored text inside the applied template. */
  prompt: string;
}) => {
  const { t } = useTranslation();
  const { onChange } = config;
  const turnOff = useCallback(() => onChange({ enabled: false }), [onChange]);
  const failure = config.enabled ? describeDynamicPromptsFailure(expansion) : null;
  // A comment always makes the prompt dynamic, so a prompt that does not expand has none to find.
  const comment = config.enabled && !failure && expansion.isDynamic ? findDynamicPromptComment(prompt) : null;
  const color = failure ? 'fg.error' : 'fg.warning';

  return (
    <Box aria-live="polite" role="status">
      {failure || comment ? (
        <Flex alignItems="flex-start" columnGap="1.5" flexWrap="wrap" rowGap="1">
          <HStack alignItems="flex-start" flex="1 1 14rem" gap="1.5" minW="0">
            <Icon as={failure ? CircleAlertIcon : TriangleAlertIcon} boxSize="3" color={color} flexShrink={0} mt="px" />
            <Text color={color} flex="1" fontSize="xs" minW="0" textWrap="pretty">
              {failure
                ? describeGenerationIssue(t, failure)
                : t('widgets.generate.dynamicPrompts.commentDropsText', { marker: comment })}
            </Text>
          </HStack>
          <HStack gap="1.5" ms="auto">
            {expansion.isError ? (
              <Button loading={expansion.isRetrying} size="xs" variant="outline" onClick={expansion.retry}>
                {t('common.retry')}
              </Button>
            ) : null}
            <Button size="xs" variant="outline" onClick={turnOff}>
              {t('widgets.generate.dynamicPrompts.turnOff')}
            </Button>
          </HStack>
        </Flex>
      ) : null}
    </Box>
  );
};
