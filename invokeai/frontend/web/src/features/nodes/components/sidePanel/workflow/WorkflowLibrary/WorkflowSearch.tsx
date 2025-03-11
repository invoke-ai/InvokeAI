import { Flex, IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectWorkflowLibrarySearchTerm,
  workflowLibrarySearchTermChanged,
} from 'features/nodes/store/workflowLibrarySlice';
import type { ChangeEvent, KeyboardEvent, RefObject } from 'react';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const WorkflowSearch = memo(({ searchInputRef }: { searchInputRef: RefObject<HTMLInputElement> }) => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectWorkflowLibrarySearchTerm);
  const { t } = useTranslation();

  const handleWorkflowSearch = useCallback(
    (newSearchTerm: string) => {
      dispatch(workflowLibrarySearchTermChanged(newSearchTerm));
    },
    [dispatch]
  );

  const clearWorkflowSearch = useCallback(() => {
    dispatch(workflowLibrarySearchTermChanged(''));
  }, [dispatch]);

  const handleKeydown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        clearWorkflowSearch();
      }
    },
    [clearWorkflowSearch]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      handleWorkflowSearch(e.target.value);
    },
    [handleWorkflowSearch]
  );

  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchInputRef]);

  return (
    <Flex justifyContent="flex-end" w="300px">
      <InputGroup>
        <Input
          ref={searchInputRef}
          placeholder={t('workflows.searchPlaceholder')}
          value={searchTerm}
          onKeyDown={handleKeydown}
          onChange={handleChange}
        />
        {searchTerm && searchTerm.length && (
          <InputRightElement h="full" pe={2}>
            <IconButton
              onClick={clearWorkflowSearch}
              size="sm"
              variant="link"
              aria-label={t('boards.clearSearch')}
              icon={<PiXBold />}
            />
          </InputRightElement>
        )}
      </InputGroup>
    </Flex>
  );
});

WorkflowSearch.displayName = 'WorkflowSearch';
