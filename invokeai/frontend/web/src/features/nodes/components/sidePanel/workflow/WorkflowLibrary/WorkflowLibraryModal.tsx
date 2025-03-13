import {
  Divider,
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import {
  $workflowLibraryCategoriesOptions,
  selectWorkflowLibraryView,
  workflowLibraryViewChanged,
} from 'features/nodes/store/workflowLibrarySlice';
import { memo, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetCountsByCategoryQuery } from 'services/api/endpoints/workflows';

import { WorkflowLibrarySideNav } from './WorkflowLibrarySideNav';
import { WorkflowLibraryTopNav } from './WorkflowLibraryTopNav';
import { WorkflowList } from './WorkflowList';

export const WorkflowLibraryModal = memo(() => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const didSync = useSyncInitialWorkflowLibraryCategories();
  return (
    <Modal isOpen={workflowLibraryModal.isOpen} onClose={workflowLibraryModal.close} isCentered>
      <ModalOverlay />
      <ModalContent
        w="calc(100% - var(--invoke-sizes-40))"
        maxW="calc(100% - var(--invoke-sizes-40))"
        h="calc(100% - var(--invoke-sizes-40))"
        maxH="calc(100% - var(--invoke-sizes-40))"
      >
        <ModalHeader>{t('workflows.workflowLibrary')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody pb={6}>
          {didSync && (
            <Flex gap={4} h="100%">
              <WorkflowLibrarySideNav />
              <Divider orientation="vertical" />
              <Flex flexDir="column" flex={1} gap={4}>
                <WorkflowLibraryTopNav />
                <WorkflowList />
              </Flex>
            </Flex>
          )}
          {!didSync && <IAINoContentFallback label={t('workflows.loading')} icon={null} />}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

WorkflowLibraryModal.displayName = 'WorkflowLibraryModal';

/**
 * On first app load, if the user's selected view has no workflows, switches to the next available view.
 */
const useSyncInitialWorkflowLibraryCategories = () => {
  const dispatch = useAppDispatch();
  const view = useAppSelector(selectWorkflowLibraryView);
  const categoryOptions = useStore($workflowLibraryCategoriesOptions);
  const [didSync, setDidSync] = useState(false);
  const recentWorkflowsCountQueryArg = useMemo(
    () =>
      ({
        categories: ['user', 'project', 'default'],
        has_been_opened: true,
      }) satisfies Parameters<typeof useGetCountsByCategoryQuery>[0],
    []
  );
  const yourWorkflowsCountQueryArg = useMemo(
    () =>
      ({
        categories: ['user', 'project'],
      }) satisfies Parameters<typeof useGetCountsByCategoryQuery>[0],
    []
  );
  const queryOptions = useMemo(
    () =>
      ({
        selectFromResult: ({ data, isLoading }) => {
          if (!data) {
            return { count: 0, isLoading: true };
          }
          return {
            count: Object.values(data).reduce((acc, count) => acc + count, 0),
            isLoading,
          };
        },
      }) satisfies Parameters<typeof useGetCountsByCategoryQuery>[1],
    []
  );

  const { count: recentWorkflowsCount, isLoading: isLoadingRecentWorkflowsCount } = useGetCountsByCategoryQuery(
    recentWorkflowsCountQueryArg,
    queryOptions
  );
  const { count: yourWorkflowsCount, isLoading: isLoadingYourWorkflowsCount } = useGetCountsByCategoryQuery(
    yourWorkflowsCountQueryArg,
    queryOptions
  );

  useEffect(() => {
    if (didSync || isLoadingRecentWorkflowsCount || isLoadingYourWorkflowsCount) {
      return;
    }
    // If the user's selected view has no workflows, switch to the next available view
    if (recentWorkflowsCount === 0 && view === 'recent') {
      if (yourWorkflowsCount > 0) {
        dispatch(workflowLibraryViewChanged('yours'));
      } else {
        dispatch(workflowLibraryViewChanged('defaults'));
      }
    } else if (yourWorkflowsCount === 0 && (view === 'yours' || view === 'shared' || view === 'private')) {
      if (recentWorkflowsCount > 0) {
        dispatch(workflowLibraryViewChanged('recent'));
      } else {
        dispatch(workflowLibraryViewChanged('defaults'));
      }
    }
    setDidSync(true);
  }, [
    categoryOptions,
    didSync,
    dispatch,
    isLoadingRecentWorkflowsCount,
    isLoadingYourWorkflowsCount,
    recentWorkflowsCount,
    view,
    yourWorkflowsCount,
  ]);

  return didSync;
};
