import { Button, Flex, Heading, Image, Link, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import newGithubIssueUrl from 'new-github-issue-url';
import InvokeLogoYellow from 'public/assets/images/invoke-symbol-ylw-lrg.svg';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiArrowSquareOutBold, PiCopyBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

type Props = {
  error: Error;
  resetErrorBoundary: () => void;
};

const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);

const AppErrorBoundaryFallback = ({ error, resetErrorBoundary }: Props) => {
  const { t } = useTranslation();
  const isLocal = useAppSelector(selectIsLocal);

  const handleCopy = useCallback(() => {
    const text = JSON.stringify(serializeError(error), null, 2);
    navigator.clipboard.writeText(`\`\`\`\n${text}\n\`\`\``);
    toast({
      id: 'ERROR_COPIED',
      title: t('toast.errorCopied'),
    });
  }, [error, t]);

  const url = useMemo(() => {
    if (isLocal) {
      return newGithubIssueUrl({
        user: 'invoke-ai',
        repo: 'InvokeAI',
        template: 'BUG_REPORT.yml',
        title: `[bug]: ${error.name}: ${error.message}`,
      });
    } else {
      return 'https://support.invoke.ai/support/tickets/new';
    }
  }, [error.message, error.name, isLocal]);

  return (
    <Flex layerStyle="body" w="100dvw" h="100dvh" alignItems="center" justifyContent="center" p={4}>
      <Flex layerStyle="first" flexDir="column" borderRadius="base" justifyContent="center" gap={8} p={16}>
        <Flex alignItems="center" gap="2">
          <Image src={InvokeLogoYellow} alt="invoke-logo" w="24px" h="24px" minW="24px" minH="24px" userSelect="none" />
          <Heading fontSize="2xl">{t('common.somethingWentWrong')}</Heading>
        </Flex>

        <Flex
          layerStyle="second"
          px={8}
          py={4}
          gap={4}
          borderRadius="base"
          justifyContent="space-between"
          alignItems="center"
        >
          <Text fontWeight="semibold" color="error.400">
            {error.name}: {error.message}
          </Text>
        </Flex>
        <Flex gap={4}>
          <Button leftIcon={<PiArrowCounterClockwiseBold />} onClick={resetErrorBoundary}>
            {t('accessibility.resetUI')}
          </Button>
          <Button leftIcon={<PiCopyBold />} onClick={handleCopy}>
            {t('common.copyError')}
          </Button>
          <Link href={url} isExternal>
            <Button leftIcon={<PiArrowSquareOutBold />}>
              {isLocal ? t('accessibility.createIssue') : t('accessibility.submitSupportTicket')}
            </Button>
          </Link>
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(AppErrorBoundaryFallback);
