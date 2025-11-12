import { Button, Flex, Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clearModelSelection, selectSelectedModelKeys } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type ModelListHeaderProps = {
  onBulkDelete: () => void;
};

export const ModelListHeader = memo(({ onBulkDelete }: ModelListHeaderProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedModelKeys = useAppSelector(selectSelectedModelKeys);
  const selectionCount = selectedModelKeys.length;

  const handleClearSelection = useCallback(() => {
    dispatch(clearModelSelection());
  }, [dispatch]);

  if (selectionCount === 0) {
    return null;
  }

  return (
    <Flex
      position="sticky"
      top={0}
      bg="base.800"
      px={3}
      py={2}
      gap={3}
      alignItems="center"
      justifyContent="space-between"
      zIndex={1}
      borderBottomWidth={1}
      borderColor="base.700"
    >
      <Tag size="lg" colorScheme="invokeBlue" variant="subtle">
        <TagLabel>
          {selectionCount} {t('common.selected')}
        </TagLabel>
        <TagCloseButton onClick={handleClearSelection} />
      </Tag>
      <Button
        size="sm"
        colorScheme="error"
        leftIcon={<PiTrashSimpleBold />}
        onClick={onBulkDelete}
        flexShrink={0}
      >
        {t('modelManager.deleteModels', { count: selectionCount })}
      </Button>
    </Flex>
  );
});

ModelListHeader.displayName = 'ModelListHeader';
