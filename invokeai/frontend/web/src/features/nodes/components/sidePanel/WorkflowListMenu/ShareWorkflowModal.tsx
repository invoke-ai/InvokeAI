import {
  Button,
  Flex,
  Heading,
  IconButton,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

const $workflowToShare = atom<WorkflowRecordListItemDTO | null>(null);
const clearWorkflowToShare = () => $workflowToShare.set(null);

export const useShareWorkflow = () => {
  const copyWorkflowLink = useCallback((workflow: WorkflowRecordListItemDTO) => {
    $workflowToShare.set(workflow);
  }, []);

  return copyWorkflowLink;
};

export const ShareWorkflowModal = () => {
  useAssertSingleton('ShareWorkflowModal');
  const workflowToShare = useStore($workflowToShare);
  const projectUrl = useStore($projectUrl);
  const { t } = useTranslation();

  const workflowLink = useMemo(() => {
    if (!workflowToShare || !projectUrl) {
      return null;
    }
    return `${window.location.origin}${projectUrl}/studio?selectedWorkflowId=${workflowToShare.workflow_id}`;
  }, [projectUrl, workflowToShare]);

  const handleCopy = useCallback(() => {
    if (!workflowLink) {
      return;
    }
    navigator.clipboard.writeText(workflowLink);
    toast({
      status: 'success',
      title: t('toast.linkCopied'),
    });
    $workflowToShare.set(null);
  }, [workflowLink, t]);

  return (
    <Modal isOpen={workflowToShare !== null} onClose={clearWorkflowToShare} isCentered size="lg" useInert={false}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          <Flex flexDir="column" gap={2}>
            <Heading fontSize="xl">{t('workflows.copyShareLinkForWorkflow')}</Heading>
            <Text fontSize="md">{workflowToShare?.name}</Text>
          </Flex>
        </ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Flex layerStyle="third" p={4} borderRadius="base" alignItems="center">
            <Text fontWeight="semibold">{workflowLink}</Text>
            <IconButton
              variant="ghost"
              aria-label={t('common.copy')}
              tooltip={t('common.copy')}
              icon={<PiCopyBold />}
              onClick={handleCopy}
            />
          </Flex>
        </ModalBody>

        <ModalFooter>
          <Button onClick={clearWorkflowToShare}>{t('common.close')}</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};
