import {
  Button,
  ButtonGroup,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementContainerDataChanged } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import { memo, useCallback } from 'react';
import { PiWrenchFill } from 'react-icons/pi';

export const ContainerElementSettings = memo(({ element }: { element: ContainerElement }) => {
  const dispatch = useAppDispatch();
  const toggleDirection = useCallback(() => {
    dispatch(
      formElementContainerDataChanged({
        id: element.id,
        changes: { direction: element.data.direction === 'column' ? 'row' : 'column' },
      })
    );
  }, [dispatch, element.data.direction, element.id]);
  return (
    <Popover>
      <PopoverTrigger>
        <IconButton aria-label="settings" icon={<PiWrenchFill />} variant="link" size="sm" alignSelf="stretch" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <FormControl>
            <FormLabel m={0}>Direction</FormLabel>
            <ButtonGroup variant="outline" size="sm">
              <Button onClick={toggleDirection} colorScheme={element.data.direction === 'row' ? 'invokeBlue' : 'base'}>
                Row
              </Button>
              <Button
                onClick={toggleDirection}
                colorScheme={element.data.direction === 'column' ? 'invokeBlue' : 'base'}
              >
                Column
              </Button>
            </ButtonGroup>
          </FormControl>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
ContainerElementSettings.displayName = 'ContainerElementSettings';
