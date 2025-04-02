import type { ButtonProps, CheckboxProps } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Checkbox,
  Collapse,
  Flex,
  Icon,
  IconButton,
  Spacer,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import type { WorkflowLibraryView, WorkflowTagCategory } from 'features/nodes/store/workflowLibrarySlice';
import {
  $workflowLibraryCategoriesOptions,
  $workflowLibraryTagCategoriesOptions,
  $workflowLibraryTagOptions,
  selectWorkflowLibrarySelectedTags,
  selectWorkflowLibraryView,
  workflowLibraryTagsReset,
  workflowLibraryTagToggled,
  workflowLibraryViewChanged,
} from 'features/nodes/store/workflowLibrarySlice';
import { selectAllowPublishWorkflows } from 'features/system/store/configSlice';
import { NewWorkflowButton } from 'features/workflowLibrary/components/NewWorkflowButton';
import { UploadWorkflowButton } from 'features/workflowLibrary/components/UploadWorkflowButton';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiStarFill, PiUsersBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';
import { useGetCountsByTagQuery } from 'services/api/endpoints/workflows';

export const WorkflowLibrarySideNav = () => {
  const { t } = useTranslation();
  const categoryOptions = useStore($workflowLibraryCategoriesOptions);
  const view = useAppSelector(selectWorkflowLibraryView);
  const allowPublishWorkflows = useAppSelector(selectAllowPublishWorkflows);

  return (
    <Flex h="full" minH={0} overflow="hidden" flexDir="column" w={64} gap={0}>
      <Flex flexDir="column" w="full" pb={2} gap={2}>
        <WorkflowLibraryViewButton view="recent">{t('workflows.recentlyOpened')}</WorkflowLibraryViewButton>
        <WorkflowLibraryViewButton view="yours">{t('workflows.yourWorkflows')}</WorkflowLibraryViewButton>
        {categoryOptions.includes('project') && (
          <Collapse in={view === 'yours' || view === 'shared' || view === 'private'}>
            <Flex flexDir="column" gap={2} pl={4} pt={2}>
              <WorkflowLibraryViewButton size="sm" view="private">
                {t('workflows.private')}
              </WorkflowLibraryViewButton>
              <WorkflowLibraryViewButton size="sm" rightIcon={<PiUsersBold />} view="shared">
                {t('workflows.shared')}
                <Spacer />
              </WorkflowLibraryViewButton>
            </Flex>
          </Collapse>
        )}
        {allowPublishWorkflows && (
          <WorkflowLibraryViewButton view="published">{t('workflows.published')}</WorkflowLibraryViewButton>
        )}
      </Flex>
      <Flex h="full" minH={0} overflow="hidden" flexDir="column">
        <BrowseWorkflowsButton />
        <DefaultsViewCheckboxesCollapsible />
      </Flex>
      <Spacer />
      <NewWorkflowButton />
      <UploadWorkflowButton />
    </Flex>
  );
};

const BrowseWorkflowsButton = memo(() => {
  const { t } = useTranslation();
  const view = useAppSelector(selectWorkflowLibraryView);
  const dispatch = useAppDispatch();
  const selectedTags = useAppSelector(selectWorkflowLibrarySelectedTags);
  const resetTags = useCallback(() => {
    dispatch(workflowLibraryTagsReset());
  }, [dispatch]);

  if (view === 'defaults' && selectedTags.length > 0) {
    return (
      <ButtonGroup>
        <WorkflowLibraryViewButton view="defaults" w="auto">
          {t('workflows.browseWorkflows')}
        </WorkflowLibraryViewButton>
        <Tooltip label={t('workflows.deselectAll')}>
          <IconButton
            onClick={resetTags}
            size="md"
            aria-label={t('workflows.deselectAll')}
            icon={<PiArrowCounterClockwiseBold size={12} />}
            variant="ghost"
            bg="base.700"
            color="base.50"
          />
        </Tooltip>
      </ButtonGroup>
    );
  }

  return <WorkflowLibraryViewButton view="defaults">{t('workflows.browseWorkflows')}</WorkflowLibraryViewButton>;
});
BrowseWorkflowsButton.displayName = 'BrowseWorkflowsButton';

const overlayscrollbarsOptions = getOverlayScrollbarsParams({ visibility: 'visible' }).options;

const DefaultsViewCheckboxesCollapsible = memo(() => {
  const tagCategoryOptions = useStore($workflowLibraryTagCategoriesOptions);
  const view = useAppSelector(selectWorkflowLibraryView);

  return (
    <Collapse in={view === 'defaults'}>
      <Flex flexDir="column" gap={2} pl={4} py={2} overflow="hidden" h="100%" minH={0}>
        <OverlayScrollbarsComponent style={overlayScrollbarsStyles} options={overlayscrollbarsOptions}>
          <Flex flexDir="column" gap={2} overflow="auto">
            {tagCategoryOptions.map((tagCategory) => (
              <TagCategory key={tagCategory.categoryTKey} tagCategory={tagCategory} />
            ))}
          </Flex>
        </OverlayScrollbarsComponent>
      </Flex>
    </Collapse>
  );
});
DefaultsViewCheckboxesCollapsible.displayName = 'DefaultsViewCheckboxes';

const useCountForIndividualTag = (tag: string) => {
  const allTags = useStore($workflowLibraryTagOptions);
  const queryArg = useMemo(
    () =>
      ({
        tags: allTags.map((tag) => tag.label),
        categories: ['default'],
      }) satisfies Parameters<typeof useGetCountsByTagQuery>[0],
    [allTags]
  );
  const queryOptions = useMemo(
    () =>
      ({
        selectFromResult: ({ data }) => ({
          count: data?.[tag] ?? 0,
        }),
      }) satisfies Parameters<typeof useGetCountsByTagQuery>[1],
    [tag]
  );

  const { count } = useGetCountsByTagQuery(queryArg, queryOptions);

  return count;
};

const useCountForTagCategory = (tagCategory: WorkflowTagCategory) => {
  const allTags = useStore($workflowLibraryTagOptions);
  const queryArg = useMemo(
    () =>
      ({
        tags: allTags.map((tag) => tag.label),
        categories: ['default'], // We only allow filtering by tag for default workflows
      }) satisfies Parameters<typeof useGetCountsByTagQuery>[0],
    [allTags]
  );
  const queryOptions = useMemo(
    () =>
      ({
        selectFromResult: ({ data }) => {
          if (!data) {
            return { count: 0 };
          }
          return {
            count: tagCategory.tags.reduce((acc, tag) => acc + (data[tag.label] ?? 0), 0),
          };
        },
      }) satisfies Parameters<typeof useGetCountsByTagQuery>[1],
    [tagCategory]
  );

  const { count } = useGetCountsByTagQuery(queryArg, queryOptions);

  return count;
};

const WorkflowLibraryViewButton = memo(({ view, ...rest }: ButtonProps & { view: WorkflowLibraryView }) => {
  const dispatch = useDispatch();
  const selectedView = useAppSelector(selectWorkflowLibraryView);
  const onClick = useCallback(() => {
    dispatch(workflowLibraryViewChanged(view));
  }, [dispatch, view]);

  return (
    <Button
      variant="ghost"
      justifyContent="flex-start"
      size="md"
      flexShrink={0}
      w="full"
      onClick={onClick}
      {...rest}
      bg={selectedView === view ? 'base.700' : undefined}
      color={selectedView === view ? 'base.50' : undefined}
    />
  );
});
WorkflowLibraryViewButton.displayName = 'NavButton';

const TagCategory = memo(({ tagCategory }: { tagCategory: WorkflowTagCategory }) => {
  const { t } = useTranslation();
  const count = useCountForTagCategory(tagCategory);

  if (count === 0) {
    return null;
  }

  return (
    <Flex flexDir="column" gap={2}>
      <Text fontWeight="semibold" color="base.300" flexShrink={0}>
        {t(tagCategory.categoryTKey)}
      </Text>
      <Flex flexDir="column" gap={2} pl={4}>
        {tagCategory.tags.map((tag) => (
          <TagCheckbox key={tag.label} tag={tag} />
        ))}
      </Flex>
    </Flex>
  );
});
TagCategory.displayName = 'TagCategory';

const TagCheckbox = memo(({ tag, ...rest }: CheckboxProps & { tag: { label: string; recommended?: boolean } }) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedTags = useAppSelector(selectWorkflowLibrarySelectedTags);
  const isChecked = selectedTags.includes(tag.label);
  const count = useCountForIndividualTag(tag.label);

  const onChange = useCallback(() => {
    dispatch(workflowLibraryTagToggled(tag.label));
  }, [dispatch, tag]);

  if (count === 0) {
    return null;
  }

  return (
    <Flex alignItems="center" gap={2}>
      <Checkbox isChecked={isChecked} onChange={onChange} {...rest} flexShrink={0} />
      <Text>{`${tag.label} (${count})`}</Text>
      {tag.recommended && (
        <Tooltip label={t('workflows.recommended')}>
          <Box as="span" lineHeight={0}>
            <Icon as={PiStarFill} boxSize={4} fill="invokeYellow.500" />
          </Box>
        </Tooltip>
      )}
    </Flex>
  );
});
TagCheckbox.displayName = 'TagCheckbox';
