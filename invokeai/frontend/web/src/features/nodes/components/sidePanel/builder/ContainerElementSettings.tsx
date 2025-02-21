import {
  Button,
  ButtonGroup,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementContainerDataChanged } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWrenchFill } from 'react-icons/pi';

export const ContainerElementSettings = memo(({ element }: { element: ContainerElement }) => {
  const { id, data } = element;
  const { layout } = data;
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const setLayoutToRow = useCallback(() => {
    dispatch(formElementContainerDataChanged({ id, changes: { layout: 'row' } }));
  }, [dispatch, id]);

  const setLayoutToColumn = useCallback(() => {
    dispatch(formElementContainerDataChanged({ id, changes: { layout: 'column' } }));
  }, [dispatch, id]);

  return (
    <Popover placement="top" isLazy lazyBehavior="unmount">
      <PopoverTrigger>
        <IconButton aria-label="settings" icon={<PiWrenchFill />} variant="link" size="sm" alignSelf="stretch" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <FormControl>
            <FormLabel m={0}>{t('workflows.builder.layout')}</FormLabel>
            <ButtonGroup variant="outline" size="sm">
              <Button onClick={setLayoutToRow} colorScheme={layout === 'row' ? 'invokeBlue' : 'base'}>
                {t('workflows.builder.row')}
              </Button>
              <Button onClick={setLayoutToColumn} colorScheme={layout === 'column' ? 'invokeBlue' : 'base'}>
                {t('workflows.builder.column')}
              </Button>
            </ButtonGroup>
          </FormControl>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
ContainerElementSettings.displayName = 'ContainerElementSettings';
