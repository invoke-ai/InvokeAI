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
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

type Arg = {
  isOpen: boolean;
  onClose: () => void;
  workflowId: string;
  workflowName: string;
};

export const CopyWorkflowLinkModal = ({ isOpen, onClose, workflowId, workflowName }: Arg) => {
  const projectUrl = useStore($projectUrl);
  const { t } = useTranslation();

  const workflowLink = useMemo(() => {
    return `${projectUrl}/studio?selectedWorkflowId=${workflowId}`;
  }, [projectUrl, workflowId]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(workflowLink);
    toast({
      status: 'success',
      title: t('toast.linkCopied'),
    });
    onClose();
  }, [workflowLink, t, onClose]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="lg" useInert={false}>
      <ModalContent>
        <ModalHeader>
          <Flex flexDir="column" gap={2}>
            <Heading fontSize="xl">{t('workflows.copyShareLinkForWorkflow')}</Heading>
            <Text fontSize="md">{workflowName}</Text>
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
          <Button onClick={onClose}>{t('common.close')}</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};
