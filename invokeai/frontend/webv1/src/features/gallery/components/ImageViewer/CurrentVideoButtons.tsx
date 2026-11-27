import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import type { VideoDTO } from 'services/api/types';

/**
 * Viewer toolbar actions for a selected video. Counterpart to CurrentImageButtons, trimmed
 * to what videos support: loading the saved workflow/graph into the editor. (Metadata
 * recall and canvas sends are image-only concepts.)
 */
export const CurrentVideoButtons = memo(({ videoDTO }: { videoDTO: VideoDTO }) => {
  const { t } = useTranslation();
  const hasTemplates = useStore($hasTemplates);
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const isEnabled = useMemo(() => videoDTO.has_workflow && hasTemplates, [hasTemplates, videoDTO]);

  const load = useCallback(() => {
    if (!isEnabled) {
      return;
    }
    loadWorkflowWithDialog({ type: 'video', data: videoDTO.video_name });
  }, [isEnabled, loadWorkflowWithDialog, videoDTO.video_name]);

  return (
    <IconButton
      icon={<PiFlowArrowBold />}
      tooltip={t('nodes.loadWorkflow')}
      aria-label={t('nodes.loadWorkflow')}
      isDisabled={!isEnabled}
      variant="link"
      alignSelf="stretch"
      onClick={load}
    />
  );
});

CurrentVideoButtons.displayName = 'CurrentVideoButtons';
