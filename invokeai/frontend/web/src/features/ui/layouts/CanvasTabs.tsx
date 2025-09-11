import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasAdded, canvasDeleted, canvasSelected } from 'features/controlLayers/store/canvasSlice';
import { selectCanvases } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold, PiXBold } from 'react-icons/pi';

import { CanvasTabEditableTitle } from './CanvasTabEditableTitle';

const _hover: SystemStyleObject = {
  bg: 'base.650',
};

const AddCanvasButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(canvasAdded({ isSelected: true }));
  }, [dispatch]);

  return (
    <IconButton
      size="sm"
      onClick={onClick}
      aria-label={t('canvas.addNewCanvas')}
      tooltip={t('canvas.addNewCanvas')}
      icon={<PiPlusBold />}
      bg="base.650"
      w={8}
      h={8}
    />
  );
});
AddCanvasButton.displayName = 'AddCanvasButton';

interface CloseCanvasButtonProps {
  id: string;
  canDelete: boolean;
}

const CloseCanvasButton = memo(({ id, canDelete }: CloseCanvasButtonProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(canvasDeleted({ id }));
  }, [dispatch, id]);

  return (
    <IconButton
      size="sm"
      onClick={onClick}
      aria-label={t('canvas.closeCanvas')}
      tooltip={t('canvas.closeCanvas')}
      icon={<PiXBold />}
      disabled={!canDelete}
      variant="link"
      w={8}
      h={8}
    />
  );
});
CloseCanvasButton.displayName = 'CloseCanvasButton';

interface CanvasTabProps {
  id: string;
  name: string;
  isSelected: boolean;
  canDelete: boolean;
}

const CanvasTab = memo(({ id, name, isSelected, canDelete }: CanvasTabProps) => {
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    if (!isSelected) {
      dispatch(canvasSelected({ id }));
    }
  }, [dispatch, id, isSelected]);

  return (
    <Box position="relative" w="full" h={8}>
      <Flex
        onClick={onClick}
        alignItems="center"
        borderRadius="base"
        cursor="pointer"
        py={1}
        ps={1}
        pe={1}
        gap={4}
        bg={isSelected ? 'base.650' : 'base.850'}
        _hover={_hover}
        w="full"
        h="full"
      >
        <Flex flex={1} justifyContent="center">
          <CanvasTabEditableTitle id={id} name={name} isSelected={isSelected} />
        </Flex>
        <Flex justifyContent="flex-end">
          <CloseCanvasButton id={id} canDelete={canDelete} />
        </Flex>
      </Flex>
    </Box>
  );
});
CanvasTab.displayName = 'CanvasTab';

export const CanvasTabs = () => {
  const canvases = useAppSelector(selectCanvases);

  return (
    <Flex w="full" gap={2} alignItems="center" px={2}>
      <AddCanvasButton />
      {canvases.map(({ id, name, isSelected, canDelete }) => (
        <CanvasTab key={id} id={id} name={name} isSelected={isSelected} canDelete={canDelete} />
      ))}
    </Flex>
  );
};
