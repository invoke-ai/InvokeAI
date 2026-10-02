import { Box, Code, Icon, ScrollArea } from '@chakra-ui/react';
import { CheckIcon, CopyIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { IconButton } from './Button';
import { toaster } from './toaster';
import { useScrollAreaPhantomHeal } from './useScrollAreaPhantomHeal';

/** Pass text for byte-preserving copy, or value to serialize. Owns both scroll axes; maxH controls the bound. */
export const JsonPreview = ({
  copyFailedLabel = 'Failed to copy JSON',
  copyLabel = 'Copy JSON',
  h,
  label = 'JSON preview',
  maxH = '24rem',
  text,
  value,
}: {
  /** Localized failure toast title. */
  copyFailedLabel?: string;
  /** Localized accessible name for the copy control. */
  copyLabel?: string;
  h?: string;
  /** Accessible name for the scroll viewport. */
  label?: string;
  maxH?: string;
  text?: string;
  value?: unknown;
}) => {
  const [hasCopied, setHasCopied] = useState(false);
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  useScrollAreaPhantomHeal(viewportRef);
  const json = useMemo(() => text ?? JSON.stringify(value, null, 2) ?? 'null', [text, value]);

  useEffect(
    () => () => {
      if (copyResetTimerRef.current !== null) {
        clearTimeout(copyResetTimerRef.current);
      }
    },
    []
  );

  const copy = useCallback(() => {
    navigator.clipboard
      .writeText(json)
      .then(() => {
        setHasCopied(true);

        if (copyResetTimerRef.current !== null) {
          clearTimeout(copyResetTimerRef.current);
        }

        copyResetTimerRef.current = setTimeout(() => setHasCopied(false), 1500);
      })
      .catch(() => toaster.create({ title: copyFailedLabel, type: 'error' }));
  }, [copyFailedLabel, json]);

  return (
    <Box
      bg="bg.inset"
      display="flex"
      flexDirection="column"
      h={h}
      maxH={maxH}
      maxW="full"
      minH="0"
      minW="0"
      overflow="hidden"
      position="relative"
      w="full"
    >
      <IconButton
        aria-label={copyLabel}
        bg="bg.muted"
        position="absolute"
        right="1.5"
        size="2xs"
        title={copyLabel}
        top="1.5"
        variant="ghost"
        zIndex="1"
        onClick={copy}
      >
        <Icon as={hasCopied ? CheckIcon : CopyIcon} boxSize="3" color={hasCopied ? 'green.solid' : undefined} />
      </IconButton>
      <ScrollArea.Root flex="1" maxW="full" minH="0" minW="0" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={viewportRef} aria-label={label} h="full" maxH={maxH} minW="0" role="region" w="full">
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
