import { Badge, Box, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useIsImported } from 'features/modelManagerV2/hooks/useIsImported';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { IoAdd } from 'react-icons/io5';
import { useImportMainModelsMutation } from 'services/api/endpoints/models';

export const ScanModelResultItem = ({ result }: { result: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { isImported } = useIsImported();

  const [importMainModel] = useImportMainModelsMutation();

  const isAlreadyImported = useMemo(() => {
    const prettyName = result.split('\\').slice(-1)[0];

    if (prettyName) {
      return isImported({ name: prettyName });
    } else {
      return false;
    }
  }, [result, isImported]);

  const handleQuickAdd = useCallback(() => {
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
  }, [importMainModel, result, dispatch, t]);

  return (
    <Flex justifyContent="space-between">
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.split('\\').slice(-1)[0]}</Text>
        <Text variant="subtext">{result}</Text>
      </Flex>
      <Box>
        {isAlreadyImported ? (
          <Badge>{t('common.installed')}</Badge>
        ) : (
          <Tooltip label={t('modelManager.quickAdd')}>
            <IconButton aria-label={t('modelManager.quickAdd')} icon={<IoAdd />} onClick={handleQuickAdd} />
          </Tooltip>
        )}
      </Box>
    </Flex>
  );
};
