import { Box, Code, Icon, ScrollArea } from '@chakra-ui/react';
import { CheckIcon, CopyIcon } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { useNotify } from '../../useNotify';
import { IconButton } from './Button';

/**
 * The workbench's standard JSON preview: a monospace block with a copy button
 * that owns its scrolling in both axes — long strings scroll horizontally
 * instead of stretching the surrounding layout. Pass `value` to serialize, or
 * `text` when the JSON string already exists (an export payload that must be
 * copied byte-for-byte). Defaults to a bounded height; pass `maxH` (or wrap in
 * a flex parent and pass `maxH="100%"`) to control it.
 */
export const JsonPreview = ({
  label = 'JSON preview',
  maxH = '24rem',
  text,
  value,
}: {
  /** Accessible name for the scroll viewport. */
  label?: string;
  maxH?: string;
  text?: string;
  value?: unknown;
}) => {
  const notify = useNotify();
  const [hasCopied, setHasCopied] = useState(false);
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const json = useMemo(() => text ?? JSON.stringify(value, null, 2) ?? 'null', [text, value]);

  useEffect(
    () => () => {
      if (copyResetTimerRef.current !== null) {
        clearTimeout(copyResetTimerRef.current);
      }
    },
    []
  );

  const copy = () => {
    navigator.clipboard
      .writeText(json)
      .then(() => {
        setHasCopied(true);

        if (copyResetTimerRef.current !== null) {
          clearTimeout(copyResetTimerRef.current);
        }

        copyResetTimerRef.current = setTimeout(() => setHasCopied(false), 1500);
      })
      .catch(() => notify.error('Failed to copy JSON'));
  };

  return (
    <Box
      bg="bg.inset"
      borderColor="border.subtle"
      borderWidth="1px"
      display="flex"
      flexDirection="column"
      maxH={maxH}
      maxW="full"
      minH="0"
      minW="0"
      overflow="hidden"
      position="relative"
      rounded="md"
      w="full"
    >
      <IconButton
        aria-label="Copy JSON"
        bg="bg.muted"
        position="absolute"
        right="1.5"
        size="2xs"
        title="Copy JSON"
        top="1.5"
        variant="ghost"
        zIndex="1"
        onClick={copy}
      >
        <Icon as={hasCopied ? CheckIcon : CopyIcon} boxSize="3" color={hasCopied ? 'green.solid' : undefined} />
      </IconButton>
      <ScrollArea.Root flex="1" maxW="full" minH="0" minW="0" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport aria-label={label} h="full" maxH={maxH} minW="0" w="full">
          <ScrollArea.Content w="full">
            <Code
              bg="transparent"
              display="block"
              fontSize="2xs"
              minW="max-content"
              p="2"
              whiteSpace="pre"
              wordBreak="normal"
            >
              {json}
            </Code>
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar>
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
        <ScrollArea.Scrollbar orientation="horizontal">
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
        <ScrollArea.Corner />
      </ScrollArea.Root>
    </Box>
  );
};
