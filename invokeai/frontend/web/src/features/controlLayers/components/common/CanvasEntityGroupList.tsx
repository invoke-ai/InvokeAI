import { Flex, Switch, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  allEntitiesOfTypeToggled,
  selectAllEntitiesOfType,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';

type Props = PropsWithChildren<{
  title: string;
  isSelected: boolean;
  type: CanvasEntityIdentifier['type'];
}>;

export const CanvasEntityGroupList = memo(({ title, isSelected, type, children }: Props) => {
  const dispatch = useAppDispatch();
  const selectAreAllEnabled = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        return selectAllEntitiesOfType(canvasV2, type).every((entity) => entity.isEnabled);
      }),
    [type]
  );
  const areAllEnabled = useAppSelector(selectAreAllEnabled);
  const onChange = useCallback(() => {
    dispatch(allEntitiesOfTypeToggled({ type }));
  }, [dispatch, type]);
  return (
    <Flex flexDir="column" gap={2}>
      <Flex justifyContent="space-between" alignItems="center">
        <Text color={isSelected ? 'base.200' : 'base.500'} fontWeight="semibold" userSelect="none">
          {title}
        </Text>
        <Switch size="sm" isChecked={areAllEnabled} onChange={onChange} pe={1} />
      </Flex>
      {children}
    </Flex>
  );
});

CanvasEntityGroupList.displayName = 'CanvasEntityGroupList';
