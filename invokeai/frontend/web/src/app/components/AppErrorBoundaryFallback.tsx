import { Flex, Heading, Link, Text, useToast } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import newGithubIssueUrl from 'new-github-issue-url';
import { memo, useCallback, useMemo } from 'react';
import { FaCopy, FaExternalLinkAlt } from 'react-icons/fa';
import { FaArrowRotateLeft } from 'react-icons/fa6';
import { serializeError } from 'serialize-error';

type Props = {
  error: Error;
  resetErrorBoundary: () => void;
};

const AppErrorBoundaryFallback = ({ error, resetErrorBoundary }: Props) => {
  const toast = useToast();

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
    <Flex
      layerStyle="body"
      sx={{
        w: '100vw',
        h: '100vh',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
      }}
    >
      <Flex
        layerStyle="first"
        sx={{
          flexDir: 'column',
          borderRadius: 'base',
          justifyContent: 'center',
          gap: 8,
          p: 16,
        }}
      >
        <Heading>Something went wrong</Heading>
        <Flex
          layerStyle="second"
          sx={{
            px: 8,
            py: 4,
            borderRadius: 'base',
            gap: 4,
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Text
            sx={{
              fontWeight: 600,
              color: 'error.500',
              _dark: { color: 'error.400' },
            }}
          >
            {error.name}: {error.message}
          </Text>
        </Flex>
        <Flex sx={{ gap: 4 }}>
          <IAIButton
            leftIcon={<FaArrowRotateLeft />}
            onClick={resetErrorBoundary}
          >
            Reset UI
          </IAIButton>
          <IAIButton leftIcon={<FaCopy />} onClick={handleCopy}>
            Copy Error
          </IAIButton>
          <Link href={url} isExternal>
            <IAIButton leftIcon={<FaExternalLinkAlt />}>Create Issue</IAIButton>
          </Link>
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(AppErrorBoundaryFallback);
