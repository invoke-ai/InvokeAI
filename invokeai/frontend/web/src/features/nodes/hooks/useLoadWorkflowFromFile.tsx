import { ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { useLogger } from 'app/logging/useLogger';
import { useAppDispatch } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { zWorkflow } from 'features/nodes/types/types';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { ZodError } from 'zod';
import { fromZodError, fromZodIssue } from 'zod-validation-error';
import { workflowLoadRequested } from '../store/actions';
import { useTranslation } from 'react-i18next';

export const useLoadWorkflowFromFile = () => {
  const dispatch = useAppDispatch();
  const logger = useLogger('nodes');
  const { t } = useTranslation();
  const loadWorkflowFromFile = useCallback(
    (file: File | null) => {
      if (!file) {
        return;
      }
      const reader = new FileReader();
      reader.onload = async () => {
        const rawJSON = reader.result;

        try {
          const parsedJSON = JSON.parse(String(rawJSON));
          const result = zWorkflow.safeParse(parsedJSON);

          if (!result.success) {
            const { message } = fromZodError(result.error, {
              prefix: t('nodes.workflowValidation'),
            });

            logger.error({ error: parseify(result.error) }, message);

            dispatch(
              addToast(
                makeToast({
                  title: t('nodes.unableToValidateWorkflow'),
                  status: 'error',
                  duration: 5000,
                })
              )
            );
            reader.abort();
            return;
          }

          dispatch(workflowLoadRequested(result.data));

          reader.abort();
        } catch {
          // file reader error
          dispatch(
            addToast(
              makeToast({
                title: t('nodes.unableToLoadWorkflow'),
                status: 'error',
              })
            )
          );
        }
      };

      reader.readAsText(file);
    },
    [dispatch, logger, t]
  );

  return loadWorkflowFromFile;
};

const WorkflowValidationErrorContent = memo((props: { error: ZodError }) => {
  if (props.error.issues[0]) {
    return (
      <Text>
        {fromZodIssue(props.error.issues[0], { prefix: null }).toString()}
      </Text>
    );
  }
  return (
    <UnorderedList>
      {props.error.issues.map((issue, i) => (
        <ListItem key={i}>
          <Text>{fromZodIssue(issue, { prefix: null }).toString()}</Text>
        </ListItem>
      ))}
    </UnorderedList>
  );
});

WorkflowValidationErrorContent.displayName = 'WorkflowValidationErrorContent';
