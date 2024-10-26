import { Link } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectName, $projectUrl } from 'app/store/nanostores/projectId';
import { memo } from 'react';

export const GalleryHeader = memo(() => {
  const projectName = useStore($projectName);
  const projectUrl = useStore($projectUrl);

  if (projectName && projectUrl) {
    return (
      <Link fontSize="md" fontWeight="semibold" noOfLines={1} wordBreak="break-all" href={projectUrl}>
        {projectName}
      </Link>
    );
  }

  return null;
});

GalleryHeader.displayName = 'GalleryHeader';
