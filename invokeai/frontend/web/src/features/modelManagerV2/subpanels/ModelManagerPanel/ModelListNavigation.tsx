import { Flex, IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectSearchTerm,
  setSearchTerm,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { t } from 'i18next';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { PiXBold } from 'react-icons/pi';

import { ModelListBulkActions } from './ModelListBulkActions'
import { ModelTypeFilter } from './ModelTypeFilter';

export const ModelListNavigation = memo(() => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectSearchTerm);

  const handleSearch: ChangeEventHandler<HTMLInputElement> = useCallback(
    (event) => {
      dispatch(setSearchTerm(event.target.value));
    },
    [dispatch]
  );

  const clearSearch = useCallback(() => {
    dispatch(setSearchTerm(''));
  }, [dispatch]);

  return (
    <Flex flexDirection="column" gap={2} bg="base.800" p={3} pb={2} rounded="base">
      <Flex gap={2} alignItems="center">
        <Flex alignItems="center" w="100%">
          <InputGroup>
            <Input
              placeholder={t('modelManager.search')}
              value={searchTerm || ''}
              data-testid="board-search-input"
              onChange={handleSearch}
            />

            {!!searchTerm?.length && (
              <InputRightElement h="full" pe={2}>
                <IconButton
                  size="sm"
                  variant="link"
                  aria-label={t('boards.clearSearch')}
                  icon={<PiXBold />}
                  onClick={clearSearch}
                />
              </InputRightElement>
            )}
          </InputGroup>
        </Flex>
        <Flex shrink={0}>
          <ModelTypeFilter />
        </Flex>
      </Flex>
      <ModelListBulkActions />
    </Flex>
  );
});

ModelListNavigation.displayName = 'ModelListNavigation';
