import { DeleteIcon } from '@chakra-ui/icons';
import { Box, Flex, Spacer, Text, Tooltip } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaEdit } from 'react-icons/fa';
import {
  MainModelConfigEntity,
  useDeleteMainModelsMutation,
} from 'services/api/endpoints/models';

type ModelListItemProps = {
  model: MainModelConfigEntity;
  isSelected: boolean;
  setSelectedModelId: (v: string | undefined) => void;
};

export default function ModelListItem(props: ModelListItemProps) {
  const isBusy = useAppSelector(selectIsBusy);
  const { t } = useTranslation();
  const [deleteMainModel] = useDeleteMainModelsMutation();

  const { model, isSelected, setSelectedModelId } = props;

  const handleSelectModel = useCallback(() => {
    setSelectedModelId(model.id);
  }, [model.id, setSelectedModelId]);

  const handleModelDelete = useCallback(() => {
    deleteMainModel(model);
    setSelectedModelId(undefined);
  }, [deleteMainModel, model, setSelectedModelId]);

  return (
    <Flex sx={{ gap: 2, alignItems: 'center', w: 'full' }}>
      <Flex
        as={IAIButton}
        isChecked={isSelected}
        sx={{
          p: 2,
          borderRadius: 'base',
          w: 'full',
          alignItems: 'center',
          bg: isSelected ? 'accent.200' : 'base.100',
          _hover: {
            bg: isSelected ? 'accent.250' : 'base.150',
          },
          _dark: {
            bg: isSelected ? 'accent.600' : 'base.850',
            _hover: {
              bg: isSelected ? 'accent.550' : 'base.800',
            },
          },
        }}
        onClick={handleSelectModel}
      >
        <Box cursor="pointer">
          <Tooltip label={model.description} hasArrow placement="bottom">
            <Text fontWeight="600">{model.model_name}</Text>
          </Tooltip>
        </Box>
        <Spacer onClick={handleSelectModel} cursor="pointer" />
        <IAIIconButton
          icon={<FaEdit />}
          size="sm"
          onClick={handleSelectModel}
          aria-label={t('accessibility.modifyConfig')}
          isDisabled={isBusy}
          variant="link"
        />
      </Flex>
      <IAIAlertDialog
        title={t('modelManager.deleteModel')}
        acceptCallback={handleModelDelete}
        acceptButtonText={t('modelManager.delete')}
        triggerComponent={
          <IAIIconButton
            icon={<DeleteIcon />}
            aria-label={t('modelManager.deleteConfig')}
            isDisabled={isBusy}
            colorScheme="error"
          />
        }
      >
        <Flex rowGap={4} flexDirection="column">
          <p style={{ fontWeight: 'bold' }}>{t('modelManager.deleteMsg1')}</p>
          <p>{t('modelManager.deleteMsg2')}</p>
        </Flex>
      </IAIAlertDialog>
    </Flex>
  );
}
