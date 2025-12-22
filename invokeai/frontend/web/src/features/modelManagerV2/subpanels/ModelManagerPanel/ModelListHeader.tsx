import {
  Button,
  Flex,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Tag,
  TagCloseButton,
  TagLabel,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clearModelSelection, selectSelectedModelKeys } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiTrashSimpleBold } from 'react-icons/pi';

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
      <Menu>
        <MenuButton as={Button} size="sm" rightIcon={<PiCaretDownBold />} flexShrink={0}>
          {t('modelManager.actions')}
        </MenuButton>
        <MenuList>
          <MenuItem icon={<PiTrashSimpleBold />} onClick={onBulkDelete} color="error.300">
            {t('modelManager.deleteModels', { count: selectionCount })}
          </MenuItem>
        </MenuList>
      </Menu>
    </Flex>
  );
});

ModelListHeader.displayName = 'ModelListHeader';
