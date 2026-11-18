import type { WidgetLabelProps, WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack, Text } from '@chakra-ui/react';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { IconButton } from '@platform/ui/Button';
import { toaster } from '@platform/ui/toaster';
import { CopyIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { serverOutputControllers } from './controller';
import { getServerOutputStatus, type ServerOutputStatus } from './serverOutputStatus';
import { useServerOutputAccess, useServerOutputController, useServerOutputReader } from './useServerOutputAccess';

const STATUS_DOT: Record<ServerOutputStatus, string> = {
  checking: 'fg.subtle',
  connecting: 'fg.subtle',
  live: 'green.solid',
  reconnecting: 'orange.solid',
  unavailable: 'fg.subtle',
};

/**
 * The title, then the reader's state as a dot and one word. The word is the panel's only live region, and it changes
 * only with the state, so assistive technology hears transitions, not output.
 */
export const ServerOutputHeaderLabel = ({ instanceId }: WidgetLabelProps) => {
  const { t } = useTranslation();
  const snapshot = useServerOutputController(serverOutputControllers.useValue(instanceId));
  const access = useServerOutputAccess(snapshot);
  const reader = useServerOutputReader(snapshot);
  const status = getServerOutputStatus(access, reader?.phase ?? null);

  // Baseline alignment puts the smaller status word on the title's baseline, whatever either line height is; the dot
  // is inline in the word and centered on its x-height, so the status is one text box with one baseline.
  return (
    <HStack align="baseline" gap="2.5" minW="0">
      <Text data-widget-identity-label="" flexShrink={0} fontWeight="700">
        {t('widgets.labels.serverOutput')}
      </Text>
      <Text
        color={status === 'reconnecting' ? 'fg.warning' : 'fg.muted'}
        data-server-output-status={status}
        fontSize="xs"
        minW="0"
        role="status"
        truncate
      >
        <Box
          aria-hidden
          as="span"
          bg={STATUS_DOT[status]}
          borderRadius="full"
          boxSize="1.5"
          display="inline-block"
          me="1.5"
          verticalAlign="middle"
        />
        {t(`widgets.serverOutput.status.${status}`)}
      </Text>
    </HStack>
  );
};

/** Copies the selection inside the log, or all held output; Ctrl/⌘+C in the focused log does the same. Hidden while
 * the view refuses. */
export const ServerOutputHeaderActions = ({ instance }: WidgetViewProps) => {
  const { t } = useTranslation();
  const snapshot = useServerOutputController(serverOutputControllers.useValue(instance.id));
  const { attachment } = snapshot;
  const access = useServerOutputAccess(snapshot);
  const reader = useServerOutputReader(snapshot);
  const isUnavailable = getServerOutputStatus(access, reader?.phase ?? null) === 'unavailable';
  const hasOutput = (reader?.transcript.rowCount ?? 0) > 0;
  const canCopy = canCopyText();
  const copy = useCallback(() => {
    if (!attachment) {
      return;
    }

    copyText(attachment.getCopyText()).then(
      () => toaster.create({ duration: 2500, title: t('widgets.serverOutput.copied'), type: 'success' }),
      () => toaster.create({ title: t('widgets.serverOutput.copyFailed'), type: 'error' })
    );
  }, [attachment, t]);

  // Nothing can be read while refused; an empty transcript only disables it.
  if (isUnavailable) {
    return null;
  }

  return (
    <IconButton
      aria-label={t('widgets.serverOutput.copy')}
      disabled={!hasOutput}
      size="sm"
      tooltip={t('widgets.serverOutput.copyHint')}
      unavailableReason={canCopy ? undefined : t('common.copyUnavailable')}
      variant="ghost"
      onClick={copy}
    >
      <CopyIcon />
    </IconButton>
  );
};
