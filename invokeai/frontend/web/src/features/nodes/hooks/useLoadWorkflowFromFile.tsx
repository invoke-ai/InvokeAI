import { ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { useLogger } from 'app/logging/useLogger';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { RefObject, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ZodError } from 'zod';
import { fromZodIssue } from 'zod-validation-error';
import { workflowLoadRequested } from 'features/nodes/store/actions';

export const useLoadWorkflowFromFile = (resetRef: RefObject<() => void>) => {
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
          dispatch(workflowLoadRequested(parsedJSON));
        } catch (e) {
          // There was a problem reading the file
          logger.error(t('nodes.unableToLoadWorkflow'));
          dispatch(
            addToast(
              makeToast({
                title: t('nodes.unableToLoadWorkflow'),
                status: 'error',
              })
            )
          );
          reader.abort();
        }
      };

      reader.readAsText(file);
      // Reset the file picker internal state so that the same file can be loaded again
      resetRef.current?.();
    },
    [dispatch, logger, resetRef, t]
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
