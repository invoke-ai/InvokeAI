/* eslint-disable i18next/no-literal-string */
import { Button, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { useAppSelector } from 'app/store/storeHooks';
import type { WorkflowLibraryCategory } from 'features/nodes/store/types';
import { selectWorkflowBrowsingCategory, workflowBrowsingCategoryChanged } from 'features/nodes/store/workflowSlice';
import { useCallback } from 'react';
import { PiUsersBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';

export const WorkflowLibrarySideNav = () => {
  const dispatch = useDispatch();
  const browsingCategory = useAppSelector(selectWorkflowBrowsingCategory);
  const workflowCategories = useStore($workflowCategories);

  const handleCategoryChange = useCallback(
    (category: WorkflowLibraryCategory) => {
      dispatch(workflowBrowsingCategoryChanged(category));
    },
    [dispatch]
  );

  return (
    <Flex flexDir="column" gap={2} borderRight="1px solid" borderColor="base.400" h="full" pr={4}>
      <Button
        variant="ghost"
        fontWeight="bold"
        justifyContent="flex-start"
        size="md"
        isActive={browsingCategory === 'account'}
        onClick={handleCategoryChange.bind(null, 'account')}
        _active={{
          bg: 'base.700',
          color: 'base.100',
        }}
      >
        Your Workflows
      </Button>
      {workflowCategories.includes('project') && (
        <Flex flexDir="column" gap={2} pl={4}>
          <Button
            variant="ghost"
            fontWeight="bold"
            justifyContent="flex-start"
            size="sm"
            isActive={browsingCategory === 'private'}
            onClick={handleCategoryChange.bind(null, 'private')}
            _active={{
              bg: 'base.700',
              color: 'base.100',
            }}
          >
            Private
          </Button>
          <Button
            variant="ghost"
            fontWeight="bold"
            justifyContent="flex-start"
            size="sm"
            rightIcon={<PiUsersBold />}
            isActive={browsingCategory === 'shared'}
            onClick={handleCategoryChange.bind(null, 'shared')}
            _active={{
              bg: 'base.700',
              color: 'base.100',
            }}
          >
            Shared
          </Button>
        </Flex>
      )}
      <Button
        variant="ghost"
        fontWeight="bold"
        justifyContent="flex-start"
        size="md"
        isActive={browsingCategory === 'default'}
        onClick={handleCategoryChange.bind(null, 'default')}
        _active={{
          bg: 'base.700',
          color: 'base.100',
        }}
      >
        Browse Workflows
      </Button>
    </Flex>
  );
};
