import {
  Button,
  Box,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Progress,
  Text,
} from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { useMemo } from 'react';
import { useGetModelImportsQuery } from '../../../services/api/endpoints/models';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { PiXBold } from 'react-icons/pi';

export const ImportQueue = () => {
  const dispatch = useAppDispatch();

  // start with this data then pull from sockets (idk how to do that yet, also might not even use this and just use socket)
  const { data } = useGetModelImportsQuery();

  const progressValues = useMemo(() => {
    if (!data) {
      return [];
    }
    const values = [];
    for (let i = 0; i < data.length; i++) {
      let value;
      if (data[i] && data[i]?.bytes && data[i]?.total_bytes) {
        value = (data[i]?.bytes / data[i]?.total_bytes) * 100;
      }
      values.push(value || undefined);
    }
    return values;
  }, [data]);

  return (
    <Box mt={3} layerStyle="first" p={3} borderRadius="base" w="full" h="full">
      <Flex direction="column" gap="2">
        {data?.map((model, i) => (
          <Flex key={i} gap="3" w="full" alignItems="center" textAlign="center">
            <Text w="20%" whiteSpace="nowrap" overflow="hidden" text-overflow="ellipsis">
              {model.source.repo_id}
            </Text>
            <Progress
              value={progressValues[i]}
              isIndeterminate={progressValues[i] === undefined}
              aria-label={t('accessibility.invokeProgressBar')}
              h={2}
              w="50%"
            />
            <Text w="20%">{model.status}</Text>
            {model.status === 'completed' ? (
              <IconButton
                isRound={true}
                size="xs"
                tooltip={t('modelManager.removeFromQueue')}
                aria-label={t('modelManager.removeFromQueue')}
                icon={<PiXBold />}
                //   onClick={handleRemove}
              />
            ) : (
              <IconButton
                isRound={true}
                size="xs"
                tooltip={t('modelManager.cancel')}
                aria-label={t('modelManager.cancel')}
                icon={<PiXBold />}
                //   onClick={handleCancel}
                colorScheme="error"
              />
            )}
          </Flex>
        ))}
      </Flex>
    </Box>
  );
};
