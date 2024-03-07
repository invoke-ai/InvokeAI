import { Box, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoAdd } from 'react-icons/io5';
import { useInstallModelMutation } from 'services/api/endpoints/models';

type Props = {
  result: string;
};
export const HuggingFaceResultItem = ({ result }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [installModel] = useInstallModelMutation();

  const handleQuickAdd = useCallback(() => {
    installModel({ source: result })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.modelAddedSimple'),
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: `${error.data.detail} `,
                status: 'error',
              })
            )
          );
        }
      });
  }, [installModel, result, dispatch, t]);

  return (
    <Flex justifyContent="space-between" w="100%">
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.split('/').slice(-1)[0]}</Text>
        <Text variant="subtext">{result}</Text>
      </Flex>
      <Box>
        <Tooltip label={t('modelManager.quickAdd')}>
          <IconButton aria-label={t('modelManager.quickAdd')} icon={<IoAdd />} onClick={handleQuickAdd} />
        </Tooltip>
      </Box>
    </Flex>
  );
};
