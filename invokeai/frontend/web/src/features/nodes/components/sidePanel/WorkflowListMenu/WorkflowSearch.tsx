import { IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSearchTerm, workflowSearchTermChanged } from 'features/nodes/store/workflowSlice';
import type { ChangeEvent, KeyboardEvent, RefObject } from 'react';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const WorkflowSearch = memo(({ searchInputRef }: { searchInputRef: RefObject<HTMLInputElement> }) => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const { t } = useTranslation();

  const handleWorkflowSearch = useCallback(
    (newSearchTerm: string) => {
      dispatch(workflowSearchTermChanged(newSearchTerm));
    },
    [dispatch]
  );

  const clearWorkflowSearch = useCallback(() => {
    dispatch(workflowSearchTermChanged(''));
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
    <InputGroup>
      <Input
        ref={searchInputRef}
        placeholder={t('stylePresets.searchByName')}
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
  );
});

WorkflowSearch.displayName = 'WorkflowSearch';
