import { Flex, Link, Spacer, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectName, $projectUrl } from 'app/store/nanostores/projectId';
import { memo } from 'react';

export const GalleryHeader = memo(() => {
  const projectName = useStore($projectName);
  const projectUrl = useStore($projectUrl);

  if (projectName && projectUrl) {
    return (
      <Flex gap={2} alignItems="center" justifyContent="space-evenly" pe={2} w="50%">
        <Text fontSize="md" fontWeight="semibold" noOfLines={1} wordBreak="break-all" w="full" textAlign="center">
          <Link href={projectUrl}>{projectName}</Link>
        </Text>
      </Flex>
    );
  }

  return <Spacer />;
});

GalleryHeader.displayName = 'GalleryHeader';
