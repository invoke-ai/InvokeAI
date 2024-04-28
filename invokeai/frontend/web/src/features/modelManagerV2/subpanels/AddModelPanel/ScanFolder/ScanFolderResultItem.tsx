import { Badge, Box, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import type { ScanFolderResponse } from 'services/api/endpoints/models';

type Props = {
  result: ScanFolderResponse[number];
  installModel: (source: string) => void;
};
export const ScanModelResultItem = ({ result, installModel }: Props) => {
  const { t } = useTranslation();

  const handleInstall = useCallback(() => {
    installModel(result.path);
  }, [installModel, result]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.path.split('\\').slice(-1)[0]}</Text>
        <Text variant="subtext">{result.path}</Text>
      </Flex>
      <Box>
        {result.is_installed ? (
          <Badge>{t('common.installed')}</Badge>
        ) : (
          <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onClick={handleInstall} size="sm" />
        )}
      </Box>
    </Flex>
  );
};
