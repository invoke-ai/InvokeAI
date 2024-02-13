import { Button, Flex, Heading, Link, Text, useToast } from '@invoke-ai/ui-library';
import newGithubIssueUrl from 'new-github-issue-url';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiArrowSquareOutBold, PiCopyBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

type Props = {
  error: Error;
  resetErrorBoundary: () => void;
};

const AppErrorBoundaryFallback = ({ error, resetErrorBoundary }: Props) => {
  const toast = useToast();
  const { t } = useTranslation();

  const handleCopy = useCallback(() => {
    const text = JSON.stringify(serializeError(error), null, 2);
    navigator.clipboard.writeText(`\`\`\`\n${text}\n\`\`\``);
    toast({
      title: 'Error Copied',
    });
  }, [error, toast]);

  const url = useMemo(
    () =>
      newGithubIssueUrl({
        user: 'invoke-ai',
        repo: 'InvokeAI',
        template: 'BUG_REPORT.yml',
        title: `[bug]: ${error.name}: ${error.message}`,
      }),
    [error.message, error.name]
  );
  return (
    <Flex layerStyle="body" w="100vw" h="100vh" alignItems="center" justifyContent="center" p={4}>
      <Flex layerStyle="first" flexDir="column" borderRadius="base" justifyContent="center" gap={8} p={16}>
        <Heading>{t('common.somethingWentWrong')}</Heading>
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
            <Button leftIcon={<PiArrowSquareOutBold />}>{t('accessibility.createIssue')}</Button>
          </Link>
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(AppErrorBoundaryFallback);
