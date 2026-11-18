import type { ClipboardEvent, KeyboardEvent } from 'react';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { Button, IconButton } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { Scrollable } from '@platform/ui/Scrollable';
import { refetchFrontendConfig } from '@workbench/frontendConfig';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { ArrowDownToLineIcon, CloudOffIcon, SquareTerminalIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import type { ServerOutputController } from './controller';

import { readServerOutput } from './api';
import { createLogViewport, type LogViewport } from './logViewport';
import { createServerOutputReader, SERVER_OUTPUT_READER_TIMING, type ServerOutputReader } from './reader';
import { LogBlock, SERVER_OUTPUT_LOG_CSS } from './ServerOutputLog';
import { ServerOutputLoading } from './ServerOutputStates';
import { getTranscriptText, type Transcript } from './transcript';

const EMPTY_ICON = <SquareTerminalIcon />;
const WAITING_ICON = <CloudOffIcon />;
// The viewport re-pins the row being read itself; browser anchoring would be a second owner.
const LOG_VIEWPORT_CSS = { overflowAnchor: 'none' } as const;
// The scroll area clips everything outside its viewport, so the ring is drawn inside the viewport's edge.
const LOG_FOCUS_STYLE = { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' } as const;

interface PanelRuntime {
  reader: ServerOutputReader;
  viewport: LogViewport;
}

const createPanelRuntime = (controller: ServerOutputController): PanelRuntime => ({
  reader: createServerOutputReader({
    lifetime: captureAccountScope().signal,
    onRefused: (reason) => {
      controller.refuse(reason);
      void refetchFrontendConfig();
    },
    read: readServerOutput,
    timing: SERVER_OUTPUT_READER_TIMING,
  }),
  viewport: createLogViewport(),
});

/** Where a selection boundary falls within its row, in UTF-16 units of the row's text. */
const rowOffset = (row: HTMLElement, container: Node, offset: number): number => {
  const range = document.createRange();
  range.setStart(row, 0);
  range.setEnd(container, offset);
  return range.toString().length;
};

/**
 * The selected text as the transcript holds it. Rows are matched by ID, so text between the first and last selected
 * rows comes from the transcript even where the DOM skipped rendering it. Null when nothing in the log is selected.
 */
const getSelectedText = (content: HTMLElement, transcript: Transcript): string | null => {
  const selection = document.getSelection();

  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return null;
  }

  const range = selection.getRangeAt(0);
  let first: HTMLElement | null = null;
  let last: HTMLElement | null = null;

  for (const block of content.children) {
    if (!range.intersectsNode(block)) {
      continue;
    }
    for (const row of block.children as HTMLCollectionOf<HTMLElement>) {
      if (range.intersectsNode(row)) {
        first ??= row;
        last = row;
      }
    }
  }

  if (!first || !last) {
    return null;
  }

  const lastLength = last.textContent?.length ?? 0;

  return getTranscriptText(transcript, {
    fromId: Number(first.dataset.row),
    fromOffset: first.contains(range.startContainer) ? rowOffset(first, range.startContainer, range.startOffset) : 0,
    toId: Number(last.dataset.row),
    toOffset: last.contains(range.endContainer) ? rowOffset(last, range.endContainer, range.endOffset) : lastLength,
  });
};

const Notice = ({ children, onDismiss }: { children: string; onDismiss: () => void }) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" bg="bg.muted" color="fg.muted" flexShrink={0} fontSize="xs" gap="2" ps="2" pe="1" py="0.5">
      <Text flex="1" minW="0">
        {children}
      </Text>
      <IconButton
        aria-label={t('widgets.serverOutput.notices.dismiss')}
        size="xs"
        tooltip={t('widgets.serverOutput.notices.dismiss')}
        variant="ghost"
        onClick={onDismiss}
      >
        <XIcon />
      </IconButton>
    </Flex>
  );
};

/**
 * Owns the reader while expanded: it starts on mount and stops when the panel collapses, unmounts or hides. The header
 * label and Copy action reach it through the controller.
 */
export const ServerOutputPanel = ({ controller }: { controller: ServerOutputController }) => {
  const { t } = useTranslation();
  const [runtime] = useState(() => createPanelRuntime(controller));
  const { reader, viewport } = runtime;
  const { phase, transcript } = useSyncExternalStore(reader.subscribe, reader.getSnapshot, reader.getSnapshot);
  const { following } = useSyncExternalStore(viewport.subscribe, viewport.getSnapshot, viewport.getSnapshot);
  const wrapLines = useWorkbenchPreferenceSelector((preferences) => preferences.serverOutputWrapLines);
  const [dismissedReset, setDismissedReset] = useState(0);
  const contentRef = useRef<HTMLElement | null>(null);
  const scrollerRef = useRef<HTMLElement | null>(null);
  const attachContent = useCallback(
    (element: HTMLElement | null) => {
      contentRef.current = element;
      viewport.attachContent(element);
    },
    [viewport]
  );
  const attachScroller = useCallback(
    (element: HTMLDivElement | null) => {
      scrollerRef.current = element;
      viewport.attachScroller(element);
    },
    [viewport]
  );

  const getCopyText = useCallback((): string => {
    const current = reader.getSnapshot().transcript;
    const content = contentRef.current;
    return (content ? getSelectedText(content, current) : null) ?? getTranscriptText(current);
  }, [reader]);

  const jumpToLatest = useCallback(() => {
    viewport.jumpToLatest();
    // The button goes away under focus; the log keeps it.
    scrollerRef.current?.focus();
  }, [viewport]);

  const dismissReset = useCallback(() => {
    setDismissedReset(reader.getSnapshot().transcript.reset?.serial ?? 0);
    scrollerRef.current?.focus();
  }, [reader]);

  const logProps = useMemo(
    () => ({
      'aria-label': t('widgets.serverOutput.logLabel'),
      'aria-live': 'off' as const,
      css: LOG_VIEWPORT_CSS,
      _focusVisible: LOG_FOCUS_STYLE,
      // Native copy needs no Clipboard API (plain-HTTP origins too); with nothing selected it copies everything.
      onCopy: (event: ClipboardEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.clipboardData.setData('text/plain', getCopyText());
      },
      // Select-all stays inside the log rather than taking the whole page.
      onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
        if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'a') {
          event.preventDefault();
          if (contentRef.current) {
            document.getSelection()?.selectAllChildren(contentRef.current);
          }
        }
      },
      role: 'log',
      tabIndex: 0,
    }),
    [getCopyText, t]
  );

  useMountEffect(() => {
    const detach = controller.attach({ getCopyText, reader });
    const stop = reader.start();

    return () => {
      stop();
      detach();
    };
  });

  const isEmpty = transcript.rowCount === 0;
  const reset = transcript.reset && transcript.reset.serial !== dismissedReset ? transcript.reset : null;

  return (
    <Stack flex="1" gap="0" h="full" minH="0">
      {reset ? <Notice onDismiss={dismissReset}>{t(`widgets.serverOutput.notices.${reset.reason}`)}</Notice> : null}
      <Box flex="1" minH="0" position="relative">
        {isEmpty ? (
          phase === 'connecting' ? (
            <ServerOutputLoading />
          ) : (
            <Flex align="center" h="full" justify="center" p="4">
              {phase === 'reconnecting' ? (
                <EmptyState icon={WAITING_ICON} title={t('widgets.serverOutput.waiting')} />
              ) : (
                <EmptyState
                  description={t('widgets.serverOutput.empty.description')}
                  icon={EMPTY_ICON}
                  title={t('widgets.serverOutput.empty.title')}
                />
              )}
            </Flex>
          )
        ) : null}
        <Scrollable
          display={isEmpty ? 'none' : undefined}
          h="full"
          orientation={wrapLines ? 'vertical' : 'both'}
          viewportProps={logProps}
          viewportRef={attachScroller}
        >
          {transcript.truncated ? (
            <Text color="fg.subtle" fontSize="xs" fontStyle="italic" px="2" pt="1">
              {t('widgets.serverOutput.notices.truncated')}
            </Text>
          ) : null}
          <Box
            ref={attachContent}
            css={SERVER_OUTPUT_LOG_CSS}
            data-wrap={wrapLines ? '' : undefined}
            fontFamily="mono"
            fontSize="sm"
            fontVariantLigatures="none"
            lineHeight="1.5"
            overflowWrap={wrapLines ? 'anywhere' : 'normal'}
            px="2"
            py="1"
            whiteSpace={wrapLines ? 'pre-wrap' : 'pre'}
          >
            {transcript.blocks.map((block) => (
              <LogBlock key={block.key} block={block} />
            ))}
          </Box>
        </Scrollable>
        {!following && !isEmpty ? (
          <Button bottom="3" position="absolute" right="4" shadow="md" size="xs" onClick={jumpToLatest}>
            <ArrowDownToLineIcon />
            {t('widgets.serverOutput.jumpToLatest')}
          </Button>
        ) : null}
      </Box>
    </Stack>
  );
};
