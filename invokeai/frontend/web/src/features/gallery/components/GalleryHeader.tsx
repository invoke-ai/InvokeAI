import { Flex, Link, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectName, $projectUrl } from 'app/store/nanostores/projectId';
import { memo } from 'react';

export const GalleryHeader = memo(() => {
  const projectName = useStore($projectName);
  const projectUrl = useStore($projectUrl);

  if (projectName && projectUrl) {
    return (
      <Flex gap={2} w="full" alignItems="center" justifyContent="space-evenly" pe={2}>
        <Text fontSize="md" fontWeight="semibold" noOfLines={1} wordBreak="break-all" w="full" textAlign="center">
          <Link href={projectUrl}>{projectName}</Link>
        </Text>
      </Flex>
    );
  }

  return null;
});

GalleryHeader.displayName = 'GalleryHeader';
