import {
  Button,
  Checkbox,
  Collapse,
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
  Spinner,
  Text,
  useToast,
} from '@invoke-ai/ui-library';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretRightBold } from 'react-icons/pi';
import { useDeleteOrphanedModelsMutation, useGetOrphanedModelsQuery } from 'services/api/endpoints/models';

type OrphanedModel = {
  path: string;
  absolute_path: string;
  files: string[];
  size_bytes: number;
};

type SyncModelsDialogProps = {
  isOpen: boolean;
  onClose: () => void;
};

export const SyncModelsDialog = memo(({ isOpen, onClose }: SyncModelsDialogProps) => {
  const { t } = useTranslation();
  const toast = useToast();
  const { data: orphanedModels, isLoading, error } = useGetOrphanedModelsQuery(undefined, { skip: !isOpen });
  const [deleteOrphanedModels, { isLoading: isDeleting }] = useDeleteOrphanedModelsMutation();

  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectAll, setSelectAll] = useState(true);
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  // Initialize selected models when data loads
  useEffect(() => {
    if (orphanedModels && orphanedModels.length > 0) {
      // Default all models to selected
      setSelectedModels(new Set(orphanedModels.map((m: OrphanedModel) => m.path)));
      setSelectAll(true);
    }
  }, [orphanedModels]);

  // Show toast if no orphaned models found
  useEffect(() => {
    if (!isLoading && !error && orphanedModels && orphanedModels.length === 0) {
      toast({
        id: 'no-orphaned-models',
        title: t('modelManager.noOrphanedModels'),
        status: 'success',
        duration: 3000,
      });
      onClose();
    }
  }, [isLoading, error, orphanedModels, t, toast, onClose]);

  const handleToggleModel = useCallback((path: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const handleToggleSelectAll = useCallback(() => {
    if (selectAll && orphanedModels) {
      // Deselect all
      setSelectedModels(new Set());
      setSelectAll(false);
    } else if (orphanedModels) {
      // Select all
      setSelectedModels(new Set(orphanedModels.map((m: OrphanedModel) => m.path)));
      setSelectAll(true);
    }
  }, [selectAll, orphanedModels]);

  const handleToggleExpanded = useCallback((path: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const createToggleExpandedHandler = useCallback(
    (path: string) => () => handleToggleExpanded(path),
    [handleToggleExpanded]
  );

  const createToggleModelHandler = useCallback((path: string) => () => handleToggleModel(path), [handleToggleModel]);

  const handleDelete = useCallback(async () => {
    try {
      const result = await deleteOrphanedModels({ paths: Array.from(selectedModels) }).unwrap();

      if (result.deleted.length > 0) {
        toast({
          title: t('modelManager.orphanedModelsDeleted', { count: result.deleted.length }),
          status: 'success',
          duration: 3000,
        });
      }

      if (Object.keys(result.errors).length > 0) {
        toast({
          title: t('modelManager.orphanedModelsDeleteErrors'),
          description: Object.values(result.errors).join(', '),
          status: 'error',
          duration: 5000,
        });
      }

      onClose();
    } catch {
      toast({
        title: t('modelManager.orphanedModelsDeleteFailed'),
        status: 'error',
        duration: 5000,
      });
    }
  }, [selectedModels, deleteOrphanedModels, toast, t, onClose]);

  const formatSize = useCallback((bytes: number) => {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(2)} KB`;
    }
    if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }, []);

  // Early return if error
  if (error) {
    return (
      <Modal isOpen={isOpen} onClose={onClose} size="xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{t('modelManager.syncModels')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Text color="error.300">{t('modelManager.errorLoadingOrphanedModels')}</Text>
          </ModalBody>
          <ModalFooter>
            <Button onClick={onClose}>{t('common.close')}</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Modal isOpen={isOpen} onClose={onClose} size="xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{t('modelManager.syncModels')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex justifyContent="center" alignItems="center" minH="200px">
              <Spinner size="xl" />
            </Flex>
          </ModalBody>
        </ModalContent>
      </Modal>
    );
  }

  // No orphaned models found
  if (!orphanedModels || orphanedModels.length === 0) {
    return null;
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent maxH="80vh">
        <ModalHeader>{t('modelManager.orphanedModelsFound')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody overflowY="auto">
          <Flex flexDir="column" gap={4}>
            <Text>{t('modelManager.orphanedModelsDescription')}</Text>

            <Flex justifyContent="space-between" alignItems="center">
              <Heading size="sm">{t('modelManager.foundOrphanedModels', { count: orphanedModels.length })}</Heading>
              <Checkbox isChecked={selectAll} onChange={handleToggleSelectAll}>
                {selectAll ? t('modelManager.deselectAll') : t('modelManager.selectAll')}
              </Checkbox>
            </Flex>

            <Flex flexDir="column" gap={2}>
              {orphanedModels.map((model: OrphanedModel) => (
                <Flex key={model.path} p={3} borderWidth={1} borderRadius="md" flexDir="column" gap={2} bg="base.750">
                  <Flex justifyContent="space-between" alignItems="center">
                    <Flex alignItems="center" gap={2} flex={1}>
                      <IconButton
                        aria-label={expandedModels.has(model.path) ? 'Collapse' : 'Expand'}
                        icon={expandedModels.has(model.path) ? <PiCaretDownBold /> : <PiCaretRightBold />}
                        size="xs"
                        variant="ghost"
                        onClick={createToggleExpandedHandler(model.path)}
                      />
                      <Checkbox
                        isChecked={selectedModels.has(model.path)}
                        onChange={createToggleModelHandler(model.path)}
                      >
                        <Text fontWeight="semibold">{model.path}</Text>
                      </Checkbox>
                    </Flex>
                    <Text fontSize="sm" color="base.400">
                      {formatSize(model.size_bytes)}
                    </Text>
                  </Flex>
                  <Flex justifyContent="space-between" alignItems="center">
                    <Text fontSize="sm" color="base.400">
                      {t('modelManager.filesCount', { count: model.files.length })}
                    </Text>
                  </Flex>
                  <Collapse in={expandedModels.has(model.path)}>
                    <Flex
                      flexDir="column"
                      gap={1}
                      mt={2}
                      p={2}
                      bg="base.800"
                      borderRadius="md"
                      maxH="200px"
                      overflowY="auto"
                    >
                      {model.files.map((file) => (
                        <Text key={file} fontSize="xs" color="base.300" fontFamily="mono">
                          {file}
                        </Text>
                      ))}
                    </Flex>
                  </Collapse>
                </Flex>
              ))}
            </Flex>
          </Flex>
        </ModalBody>
        <ModalFooter gap={2}>
          <Button onClick={onClose} variant="ghost">
            {t('common.cancel')}
          </Button>
          <Button
            colorScheme="error"
            onClick={handleDelete}
            isLoading={isDeleting}
            isDisabled={selectedModels.size === 0}
          >
            {t('modelManager.deleteSelected', { count: selectedModels.size })}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});

SyncModelsDialog.displayName = 'SyncModelsDialog';
