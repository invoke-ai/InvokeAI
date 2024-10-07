import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

type Props = {
  result: string;
};
export const HuggingFaceResultItem = memo(({ result }: Props) => {
  const { t } = useTranslation();

  const [installModel] = useInstallModel();

  const onClick = useCallback(() => {
    installModel({ source: result });
  }, [installModel, result]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.split('/').slice(-1)[0]}</Text>
        <Text variant="subtext" noOfLines={1} wordBreak="break-all">
          {result}
        </Text>
      </Flex>
      <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onClick={onClick} size="sm" />
    </Flex>
  );
});

HuggingFaceResultItem.displayName = 'HuggingFaceResultItem';
