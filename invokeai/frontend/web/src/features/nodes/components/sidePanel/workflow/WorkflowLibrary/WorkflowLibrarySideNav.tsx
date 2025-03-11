import type { ButtonProps, CheckboxProps } from '@invoke-ai/ui-library';
import { Button, Checkbox, Collapse, Flex, Icon, Spacer, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { WorkflowTagCategory } from 'features/nodes/store/workflowLibrarySlice';
import {
  $workflowLibraryCategoriesOptions,
  $workflowLibraryTagCategoriesOptions,
  $workflowLibraryTagOptions,
  selectWorkflowLibraryCategories,
  selectWorkflowLibraryTags,
  workflowLibraryCategoriesChanged,
  workflowLibraryTagsReset,
  workflowLibraryTagToggled,
} from 'features/nodes/store/workflowLibrarySlice';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { NewWorkflowButton } from 'features/workflowLibrary/components/NewWorkflowButton';
import { UploadWorkflowButton } from 'features/workflowLibrary/components/UploadWorkflowButton';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUsersBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';
import { useGetTagCountsWithFilterQuery, useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import type { S } from 'services/api/types';

export const WorkflowLibrarySideNav = () => {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const categories = useAppSelector(selectWorkflowLibraryCategories);
  const categoryOptions = useStore($workflowLibraryCategoriesOptions);
  const tags = useAppSelector(selectWorkflowLibraryTags);
  const tagCategoryOptions = useStore($workflowLibraryTagCategoriesOptions);

  const selectYourWorkflows = useCallback(() => {
    dispatch(workflowLibraryCategoriesChanged(categoryOptions.includes('project') ? ['user', 'project'] : ['user']));
  }, [categoryOptions, dispatch]);

  const selectPrivateWorkflows = useCallback(() => {
    dispatch(workflowLibraryCategoriesChanged(['user']));
  }, [dispatch]);

  const selectSharedWorkflows = useCallback(() => {
    dispatch(workflowLibraryCategoriesChanged(['project']));
  }, [dispatch]);

  const selectDefaultWorkflows = useCallback(() => {
    dispatch(workflowLibraryCategoriesChanged(['default']));
  }, [dispatch]);

  const resetTags = useCallback(() => {
    dispatch(workflowLibraryTagsReset());
  }, [dispatch]);

  const isYourWorkflowsSelected = useMemo(() => {
    if (categoryOptions.includes('project')) {
      return categories.includes('user') && categories.includes('project');
    } else {
      return categories.includes('user');
    }
  }, [categoryOptions, categories]);

  const isPrivateWorkflowsExclusivelySelected = useMemo(() => {
    return categories.length === 1 && categories.includes('user');
  }, [categories]);

  const isSharedWorkflowsExclusivelySelected = useMemo(() => {
    return categories.length === 1 && categories.includes('project');
  }, [categories]);

  const isDefaultWorkflowsExclusivelySelected = useMemo(() => {
    return categories.length === 1 && categories.includes('default');
  }, [categories]);

  return (
    <Flex h="full" minH={0} overflow="hidden" flexDir="column" w={64} gap={1}>
      <Flex flexDir="column" w="full" pb={2}>
        <Text px={3} py={2} fontSize="md" fontWeight="semibold">
          {t('workflows.recentlyOpened')}
        </Text>
        <Flex flexDir="column" gap={2} pl={4}>
          <RecentWorkflows />
        </Flex>
      </Flex>
      <Flex flexDir="column" w="full" pb={2}>
        <CategoryButton isSelected={isYourWorkflowsSelected} onClick={selectYourWorkflows}>
          {t('workflows.yourWorkflows')}
        </CategoryButton>
        {categoryOptions.includes('project') && (
          <Collapse
            in={
              isYourWorkflowsSelected || isPrivateWorkflowsExclusivelySelected || isSharedWorkflowsExclusivelySelected
            }
          >
            <Flex flexDir="column" gap={2} pl={4} pt={2}>
              <CategoryButton
                size="sm"
                onClick={selectPrivateWorkflows}
                isSelected={isPrivateWorkflowsExclusivelySelected}
              >
                {t('workflows.private')}
              </CategoryButton>
              <CategoryButton
                size="sm"
                rightIcon={<PiUsersBold />}
                onClick={selectSharedWorkflows}
                isSelected={isSharedWorkflowsExclusivelySelected}
              >
                {t('workflows.shared')}
                <Spacer />
              </CategoryButton>
            </Flex>
          </Collapse>
        )}
      </Flex>
      <Flex h="full" minH={0} overflow="hidden" flexDir="column">
        <CategoryButton isSelected={isDefaultWorkflowsExclusivelySelected} onClick={selectDefaultWorkflows}>
          {t('workflows.browseWorkflows')}
        </CategoryButton>
        <Collapse in={isDefaultWorkflowsExclusivelySelected}>
          <Flex flexDir="column" gap={2} pl={4} py={2} overflow="hidden" h="100%" minH={0}>
            <Button
              isDisabled={!isDefaultWorkflowsExclusivelySelected || tags.length === 0}
              onClick={resetTags}
              size="sm"
              variant="link"
              fontWeight="bold"
              justifyContent="flex-start"
              flexGrow={0}
              flexShrink={0}
              leftIcon={<PiArrowCounterClockwiseBold />}
              h={8}
            >
              {t('workflows.resetFilters')}
            </Button>
            <Flex flexDir="column" gap={2} overflow="auto">
              {tagCategoryOptions.map((tagCategory) => (
                <TagCategory
                  key={tagCategory.categoryTKey}
                  tagCategory={tagCategory}
                  isDisabled={!isDefaultWorkflowsExclusivelySelected}
                />
              ))}
            </Flex>
          </Flex>
        </Collapse>
      </Flex>
      <Spacer />
      <NewWorkflowButton />
      <UploadWorkflowButton />
    </Flex>
  );
};

const recentWorkflowsQueryArg = {
  page: 0,
  per_page: 5,
  order_by: 'opened_at',
  direction: 'DESC',
} satisfies Parameters<typeof useListWorkflowsQuery>[0];

const RecentWorkflows = memo(() => {
  const { t } = useTranslation();
  const { data, isLoading } = useListWorkflowsQuery(recentWorkflowsQueryArg);

  if (isLoading) {
    return <Text variant="subtext">{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text variant="subtext">{t('workflows.noRecentWorkflows')}</Text>;
  }

  return (
    <>
      {data.items.map((workflow) => {
        return <RecentWorkflowButton key={workflow.workflow_id} workflow={workflow} />;
      })}
    </>
  );
});
RecentWorkflows.displayName = 'RecentWorkflows';

const useCountForIndividualTag = (tag: string) => {
  const allTags = useStore($workflowLibraryTagOptions);
  const tags = useAppSelector(selectWorkflowLibraryTags);
  const queryArg = useMemo(
    () =>
      ({
        tags_to_count: allTags,
        selected_tags: tags,
        categories: ['default'], // We only allow filtering by tag for default workflows
      }) satisfies Parameters<typeof useGetTagCountsWithFilterQuery>[0],
    [allTags, tags]
  );
  const queryOptions = useMemo(
    () =>
      ({
        selectFromResult: ({ data }) => ({
          count: data?.[tag] ?? 0,
        }),
      }) satisfies Parameters<typeof useGetTagCountsWithFilterQuery>[1],
    [tag]
  );

  const { count } = useGetTagCountsWithFilterQuery(queryArg, queryOptions);

  return count;
};

const useCountForTagCategory = (tagCategory: WorkflowTagCategory) => {
  const allTags = useStore($workflowLibraryTagOptions);
  const tags = useAppSelector(selectWorkflowLibraryTags);
  const queryArg = useMemo(
    () =>
      ({
        tags_to_count: allTags,
        selected_tags: tags,
        categories: ['default'], // We only allow filtering by tag for default workflows
      }) satisfies Parameters<typeof useGetTagCountsWithFilterQuery>[0],
    [allTags, tags]
  );
  const queryOptions = useMemo(
    () =>
      ({
        selectFromResult: ({ data }) => {
          if (!data) {
            return { count: 0 };
          }
          return {
            count: tagCategory.tags.reduce((acc, tag) => acc + (data[tag] ?? 0), 0),
          };
        },
      }) satisfies Parameters<typeof useGetTagCountsWithFilterQuery>[1],
    [tagCategory]
  );

  const { count } = useGetTagCountsWithFilterQuery(queryArg, queryOptions);

  return count;
};

const RecentWorkflowButton = memo(({ workflow }: { workflow: S['WorkflowRecordListItemWithThumbnailDTO'] }) => {
  const loadWorkflow = useLoadWorkflow();
  const load = useCallback(() => {
    loadWorkflow.loadWithDialog(workflow.workflow_id, 'view');
  }, [loadWorkflow, workflow.workflow_id]);

  return (
    <Flex
      role="button"
      key={workflow.workflow_id}
      gap={2}
      alignItems="center"
      _hover={{ textDecoration: 'underline' }}
      color="base.300"
      onClick={load}
    >
      <Text as="span" noOfLines={1} w="full" fontWeight="semibold">
        {workflow.name}
      </Text>
      {workflow.category === 'project' && <Icon as={PiUsersBold} boxSize="12px" />}
    </Flex>
  );
});
RecentWorkflowButton.displayName = 'RecentWorkflowButton';

const CategoryButton = memo(({ isSelected, ...rest }: ButtonProps & { isSelected: boolean }) => {
  return (
    <Button
      variant="ghost"
      justifyContent="flex-start"
      size="md"
      flexShrink={0}
      w="full"
      {...rest}
      bg={isSelected ? 'base.700' : undefined}
      color={isSelected ? 'base.50' : undefined}
    />
  );
});
CategoryButton.displayName = 'NavButton';

const TagCategory = memo(({ tagCategory, isDisabled }: { tagCategory: WorkflowTagCategory; isDisabled: boolean }) => {
  const { t } = useTranslation();
  const count = useCountForTagCategory(tagCategory);

  if (count === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" gap={2}>
      <Text fontWeight="semibold" color="base.300" opacity={isDisabled ? 0.5 : 1} flexShrink={0}>
        {t(tagCategory.categoryTKey)}
      </Text>
      <Flex flexDir="column" gap={2} pl={4}>
        {tagCategory.tags.map((tag) => (
          <TagCheckbox key={tag} tag={tag} isDisabled={isDisabled} />
        ))}
      </Flex>
    </Flex>
  );
});
TagCategory.displayName = 'TagCategory';

const TagCheckbox = memo(({ tag, ...rest }: CheckboxProps & { tag: string }) => {
  const dispatch = useAppDispatch();
  const selectedTags = useAppSelector(selectWorkflowLibraryTags);
  const isSelected = selectedTags.includes(tag);
  const count = useCountForIndividualTag(tag);

  const onChange = useCallback(() => {
    dispatch(workflowLibraryTagToggled(tag));
  }, [dispatch, tag]);

  if (count === 0) {
    return null;
  }

  return (
    <Checkbox isChecked={isSelected} onChange={onChange} {...rest} flexShrink={0}>
      <Text>{`${tag} (${count})`}</Text>
    </Checkbox>
  );
});
TagCheckbox.displayName = 'TagCheckbox';
