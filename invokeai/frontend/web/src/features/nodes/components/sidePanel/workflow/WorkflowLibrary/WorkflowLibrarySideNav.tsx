/* eslint-disable i18next/no-literal-string */
import { Button, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowCategories, workflowCategoriesChanged } from 'features/nodes/store/workflowSlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { useCallback, useMemo } from 'react';
import { PiUsersBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';

export const WorkflowLibrarySideNav = () => {
  const dispatch = useDispatch();
  const categories = useAppSelector(selectWorkflowCategories);
  const categoryOptions = useStore($workflowCategories);

  const handleCategoryChange = useCallback(
    (categories: WorkflowCategory[]) => {
      dispatch(workflowCategoriesChanged(categories));
    },
    [dispatch]
  );

  const handleSelectYourWorkflows = useCallback(() => {
    if (categoryOptions.includes('project')) {
      handleCategoryChange(['user', 'project']);
    } else {
      handleCategoryChange(['user']);
    }
  }, [categoryOptions, handleCategoryChange]);

  const isYourWorkflowsActive = useMemo(() => {
    if (categoryOptions.includes('project')) {
      return categories.includes('user') && categories.includes('project');
    } else {
      return categories.includes('user');
    }
  }, [categoryOptions, categories]);

  return (
    <Flex flexDir="column" gap={2} h="full">
      <Button
        variant="ghost"
        fontWeight="bold"
        justifyContent="flex-start"
        size="md"
        isActive={isYourWorkflowsActive}
        onClick={handleSelectYourWorkflows}
        _active={{
          bg: 'base.700',
          color: 'base.100',
        }}
      >
        Your Workflows
      </Button>
      {categoryOptions.includes('project') && (
        <Flex flexDir="column" gap={2} pl={4}>
          <Button
            variant="ghost"
            fontWeight="bold"
            justifyContent="flex-start"
            size="sm"
            isActive={categories.length === 1 && categories.includes('user')}
            onClick={handleCategoryChange.bind(null, ['user'])}
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
            isActive={categories.length === 1 && categories.includes('project')}
            onClick={handleCategoryChange.bind(null, ['project'])}
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
        isActive={categories.includes('default')}
        onClick={handleCategoryChange.bind(null, ['default'])}
        _active={{
          bg: 'base.700',
          color: 'base.100',
        }}
      >
        Browse Workflows
      </Button>

      {/* these are obviously placeholders - we need to figure out the best way to do this. leaning towards "tags" so that we can filter and/or have multiple selected eventually */}
      <Flex flexDir="column" gap={2} pl={4}>
        <Button
          variant="ghost"
          fontWeight="bold"
          justifyContent="flex-start"
          size="sm"
          _active={{
            bg: 'base.700',
            color: 'base.100',
          }}
        >
          Architecture
        </Button>
        <Button
          variant="ghost"
          fontWeight="bold"
          justifyContent="flex-start"
          size="sm"
          _active={{
            bg: 'base.700',
            color: 'base.100',
          }}
        >
          Fashion
        </Button>
      </Flex>
    </Flex>
  );
};
