import { Flex, Text, Box, Button, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import { IoAdd } from 'react-icons/io5';
import { useAppDispatch } from '../../../../../app/store/storeHooks';
import { useImportMainModelsMutation } from '../../../../../services/api/endpoints/models';
import { addToast } from '../../../../system/store/systemSlice';
import { makeToast } from '../../../../system/util/makeToast';

export const ScanModelResultItem = ({ result }: { result: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();

  const handleQuickAdd = () => {
    importMainModel({ source: result, config: undefined })
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
  };

  return (
    <Flex justifyContent={'space-between'}>
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.split('\\').slice(-1)[0]}</Text>
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
