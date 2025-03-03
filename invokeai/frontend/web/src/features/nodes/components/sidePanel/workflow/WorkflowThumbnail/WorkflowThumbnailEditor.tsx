import { Button, Flex } from '@invoke-ai/ui-library';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import { toast } from 'features/toast/toast';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDeleteWorkflowThumbnailMutation, useSetWorkflowThumbnailMutation } from 'services/api/endpoints/workflows';

import { WorkflowThumbnailField } from './WorkflowThumbnailField';

export const WorkflowThumbnailEditor = ({
  workflowId,
  thumbnailUrl,
}: {
  workflowId: string;
  thumbnailUrl?: string | null;
}) => {
  const { t } = useTranslation();

  const [localThumbnailUrl, setLocalThumbnailUrl] = useState<string | null>(null);
  const [canSaveChanges, setCanSaveChanges] = useState(false);

  const [setThumbnail, { isLoading }] = useSetWorkflowThumbnailMutation();
  const [deleteThumbnail, { isLoading: isDeleting }] = useDeleteWorkflowThumbnailMutation();

  const handleLocalThumbnailUrlChange = useCallback((url: string | null) => {
    setLocalThumbnailUrl(url);
    setCanSaveChanges(true);
  }, []);

  const handleSaveChanges = useCallback(async () => {
    try {
      if (localThumbnailUrl) {
        const blob = await convertImageUrlToBlob(localThumbnailUrl);
        if (!blob) {
          return;
        }
        const file = new File([blob], 'workflow_thumbnail.png', { type: 'image/png' });
        await setThumbnail({ workflow_id: workflowId, image: file }).unwrap();
      } else {
        await deleteThumbnail(workflowId).unwrap();
      }

      setCanSaveChanges(false);
      toast({ status: 'success', title: 'Workflow thumbnail updated' });
    } catch (error) {
      toast({ status: 'error', title: 'Failed to update thumbnail' });
    }
  }, [deleteThumbnail, setThumbnail, workflowId, localThumbnailUrl]);

  return (
    <Flex alignItems="center" gap={4}>
      <WorkflowThumbnailField imageUrl={thumbnailUrl} onChange={handleLocalThumbnailUrlChange} />

      <Button
        size="sm"
        isLoading={isLoading || isDeleting}
        onClick={handleSaveChanges}
        isDisabled={!canSaveChanges || !workflowId}
      >
        {t('common.saveChanges')}
      </Button>
    </Flex>
  );
};
