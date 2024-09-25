import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

function onCopy(sessionId: string) {
  navigator.clipboard.writeText(sessionId);
}

const ERROR_TYPE_TO_TITLE: Record<string, string> = {
  OutOfMemoryError: 'toast.outOfMemoryError',
};

const COMMERCIAL_ERROR_TYPE_TO_DESC: Record<string, string> = {
  OutOfMemoryError: 'toast.outOfMemoryErrorDesc',
};

export const getTitleFromErrorType = (errorType: string) => {
  return t(ERROR_TYPE_TO_TITLE[errorType] ?? 'toast.serverError');
};

type Props = { errorType: string; errorMessage?: string | null; sessionId: string; isLocal: boolean };

export default function ErrorToastDescription({ errorType, errorMessage, sessionId, isLocal }: Props) {
  const { t } = useTranslation();
  const description = useMemo(() => {
    // Special handling for commercial error types
    const descriptionTKey = isLocal ? null : COMMERCIAL_ERROR_TYPE_TO_DESC[errorType];
    if (descriptionTKey) {
      return t(descriptionTKey);
    }
    if (errorMessage) {
      return `${errorType}: ${errorMessage}`;
    }
  }, [errorMessage, errorType, isLocal, t]);
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
            onClick={onCopy.bind(null, sessionId)}
            variant="ghost"
            sx={sx}
          />
        </Flex>
      )}
    </Flex>
  );
}

const sx = { svg: { fill: 'base.50' } };
