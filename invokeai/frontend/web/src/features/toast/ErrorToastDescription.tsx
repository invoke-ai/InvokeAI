import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useClipboard } from 'common/hooks/useClipboard';
import { ExternalLink } from 'features/gallery/components/ImageViewer/NoContentForViewer';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

type DescriptionProps = { errorType: string; errorMessage?: string | null; sessionId: string; isLocal: boolean };

export const getTitle = (errorType: string) => {
  return errorType === 'OutOfMemoryError' ? t('toast.outOfMemoryError') : t('toast.serverError');
};

export default function ErrorToastDescription({ errorType, isLocal, sessionId, errorMessage }: DescriptionProps) {
  const { t } = useTranslation();
  const clipboard = useClipboard();

  const description = useMemo(() => {
    if (errorType === 'OutOfMemoryError') {
      if (isLocal) {
        return (
          <Trans
            i18nKey="toast.outOfMemoryErrorDescLocal"
            components={{
              LinkComponent: <ExternalLink href="https://invoke-ai.github.io/InvokeAI/features/low-vram/" />,
            }}
          />
        );
      } else {
        return t('toast.outOfMemoryErrorDesc');
      }
    } else if (errorMessage) {
      return `${errorType}: ${errorMessage}`;
    }
  }, [errorMessage, errorType, isLocal, t]);

  const copySessionId = useCallback(() => clipboard.writeText(sessionId), [sessionId, clipboard]);

  return (
    <Flex flexDir="column">
      {description && (
        <Text noOfLines={4} fontSize="md">
          {description}
        </Text>
      )}
      {!isLocal && (
        <Flex gap="2" alignItems="center">
          <Text fontSize="sm" fontStyle="italic">
            {t('toast.sessionRef', { sessionId })}
          </Text>
          <IconButton
            size="sm"
            aria-label="Copy"
            icon={<PiCopyBold />}
            onClick={copySessionId}
            variant="ghost"
            sx={{ svg: { fill: 'base.50' } }}
          />
        </Flex>
      )}
    </Flex>
  );
}
