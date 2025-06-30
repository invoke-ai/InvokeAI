import { Badge, Flex, HStack, IconButton, Text, Tooltip, VStack } from '@invoke-ai/ui-library';
import { useClipboard } from 'common/hooks/useClipboard';
import { RecallButton } from 'features/metadata/components/RecallButton';
import { memo, useCallback } from 'react';
import { PiCopyBold } from 'react-icons/pi';

type MetadataItemViewProps = {
  onRecall?: () => void;
  label: string;
  renderedValue: React.ReactNode;
  isDisabled: boolean;
  direction?: 'row' | 'column';
  /** Display mode for the metadata item */
  displayMode?: 'default' | 'badge' | 'simple' | 'card';
  /** Color scheme for badge display mode */
  colorScheme?: string;
  /** Whether to show copy functionality */
  showCopy?: boolean;
  /** Raw value for copy functionality */
  valueOrNull?: unknown;
};

export const MetadataItemView = memo(
  ({
    label,
    onRecall,
    isDisabled,
    renderedValue,
    direction = 'row',
    displayMode = 'default',
    colorScheme = 'invokeBlue',
    showCopy = false,
    valueOrNull,
  }: MetadataItemViewProps) => {
    const clipboard = useClipboard();

    const handleCopy = useCallback(() => {
      if (valueOrNull !== null) {
        clipboard.writeText(String(valueOrNull));
      }
    }, [clipboard, valueOrNull]);

    // Default display mode (original behavior)
    if (displayMode === 'default') {
      return (
        <Flex gap={2}>
          {onRecall && <RecallButton label={label} onClick={onRecall} isDisabled={isDisabled} />}
          <Flex direction={direction} fontSize="sm">
            <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
              {label}:
            </Text>
            {renderedValue}
          </Flex>
        </Flex>
      );
    }

    // Card display mode (for prompts)
    if (displayMode === 'card') {
      return (
        <VStack align="start" spacing={1} w="full">
          <Text fontSize="xs" fontWeight="medium" color="base.300" textTransform="uppercase" letterSpacing="wide">
            {label}
          </Text>
          <VStack
            position="relative"
            w="full"
            _hover={{
              '& .hover-actions': {
                opacity: 1,
              },
            }}
          >
            <Text
              fontSize="sm"
              bg="base.800"
              p={3}
              borderRadius="lg"
              w="full"
              wordBreak="break-word"
              border="1px solid"
              borderColor="base.700"
              color="base.100"
              lineHeight="tall"
            >
              {renderedValue}
            </Text>
            <HStack
              className="hover-actions"
              position="absolute"
              top={2}
              right={2}
              opacity={0}
              transition="opacity 0.2s"
              spacing={1}
            >
              {showCopy && (
                <Tooltip label="Copy to clipboard">
                  <IconButton
                    size="xs"
                    icon={<PiCopyBold />}
                    onClick={handleCopy}
                    colorScheme="base"
                    variant="ghost"
                    aria-label={`Copy ${label} to clipboard`}
                  />
                </Tooltip>
              )}
              {onRecall && <RecallButton label={label} onClick={onRecall} isDisabled={isDisabled} />}
            </HStack>
          </VStack>
        </VStack>
      );
    }

    // Simple display mode (for seed, steps, etc.)
    if (displayMode === 'simple') {
      return (
        <VStack align="start" spacing={1} w="full">
          <Text fontSize="xs" fontWeight="medium" color="base.300" textTransform="uppercase" letterSpacing="wide">
            {label}
          </Text>
          <VStack
            position="relative"
            w="full"
            _hover={{
              '& .hover-actions': {
                opacity: 1,
              },
            }}
          >
            <Text
              fontSize="sm"
              color="base.100"
              fontFamily="mono"
              bg="base.800"
              px={3}
              py={2}
              borderRadius="md"
              w="full"
              textAlign="center"
            >
              {renderedValue}
            </Text>
            <HStack
              className="hover-actions"
              position="absolute"
              top={1}
              right={1}
              opacity={0}
              transition="opacity 0.2s"
              spacing={1}
            >
              {showCopy && (
                <Tooltip label="Copy to clipboard">
                  <IconButton
                    size="xs"
                    icon={<PiCopyBold />}
                    onClick={handleCopy}
                    colorScheme="base"
                    variant="ghost"
                    aria-label={`Copy ${label} to clipboard`}
                  />
                </Tooltip>
              )}
              {onRecall && <RecallButton label={label} onClick={onRecall} isDisabled={isDisabled} />}
            </HStack>
          </VStack>
        </VStack>
      );
    }

    // Badge display mode (for models, etc.)
    if (displayMode === 'badge') {
      return (
        <VStack align="start" spacing={1} w="full">
          <Text fontSize="xs" fontWeight="medium" color="base.300" textTransform="uppercase" letterSpacing="wide">
            {label}
          </Text>
          <VStack
            position="relative"
            w="fit-content"
            _hover={{
              '& .hover-actions': {
                opacity: 1,
              },
            }}
          >
            <Badge colorScheme={colorScheme} variant="subtle" fontSize="sm" px={3} py={2} borderRadius="md">
              {renderedValue}
            </Badge>
            <HStack
              className="hover-actions"
              position="absolute"
              top={-2}
              right={-2}
              opacity={0}
              transition="opacity 0.2s"
              spacing={1}
              bg="base.900"
              borderRadius="md"
              p={1}
              border="1px solid"
              borderColor="base.700"
              shadow="lg"
            >
              {showCopy && (
                <Tooltip label="Copy to clipboard">
                  <IconButton
                    size="xs"
                    icon={<PiCopyBold />}
                    onClick={handleCopy}
                    colorScheme="base"
                    variant="ghost"
                    aria-label={`Copy ${label} to clipboard`}
                  />
                </Tooltip>
              )}
              {onRecall && <RecallButton label={label} onClick={onRecall} isDisabled={isDisabled} />}
            </HStack>
          </VStack>
        </VStack>
      );
    }

    return null;
  }
);

MetadataItemView.displayName = 'MetadataItemView';
