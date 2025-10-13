import { Flex, Text } from '@invoke-ai/ui-library';
import { ExternalLink } from 'features/gallery/components/ImageViewer/NoContentForViewer';
import { t } from 'i18next';
import { useMemo } from 'react';
import { Trans } from 'react-i18next';

type DescriptionProps = { errorType: string; errorMessage?: string | null };

export const getTitle = (errorType: string) => {
  return errorType === 'OutOfMemoryError' ? t('toast.outOfMemoryError') : t('toast.serverError');
};

export default function ErrorToastDescription({ errorType, errorMessage }: DescriptionProps) {
  const description = useMemo(() => {
    if (errorType === 'OutOfMemoryError') {
      return (
        <Trans
          i18nKey="toast.outOfMemoryErrorDescLocal"
          components={{
            LinkComponent: <ExternalLink href="https://invoke-ai.github.io/InvokeAI/features/low-vram/" />,
          }}
        />
      );
    } else if (errorMessage) {
      return `${errorType}: ${errorMessage}`;
    }
  }, [errorMessage, errorType]);

  return (
    <Flex flexDir="column">
      <Text noOfLines={4} fontSize="md">
        {description}
      </Text>
    </Flex>
  );
}
