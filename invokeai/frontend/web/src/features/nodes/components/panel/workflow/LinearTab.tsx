import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AddFieldToLinearViewDropData } from 'features/dnd/types';
import { memo } from 'react';
import LinearViewField from '../../fields/LinearViewField';
import ScrollableContent from '../ScrollableContent';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    return {
      fields: nodes.workflow.exposedFields,
    };
  },
  defaultSelectorOptions
);

const droppableData: AddFieldToLinearViewDropData = {
  id: 'add-field-to-linear-view',
  actionType: 'ADD_FIELD_TO_LINEAR',
};

const LinearTabContent = () => {
  const { fields } = useAppSelector(selector);

  return (
    <Box
      sx={{
        position: 'relative',
        w: 'full',
        h: 'full',
      }}
    >
      <ScrollableContent>
        <Flex
          sx={{
            position: 'relative',
            flexDir: 'column',
            alignItems: 'flex-start',
            gap: 2,
            h: 'full',
            w: 'full',
          }}
        >
          {fields.length ? (
            fields.map(({ nodeId, fieldName }) => (
              <LinearViewField
                key={`${nodeId}-${fieldName}`}
                nodeId={nodeId}
                fieldName={fieldName}
              />
            ))
          ) : (
            <IAINoContentFallback
              label="No fields added to Linear View"
              icon={null}
            />
          )}
        </Flex>
      </ScrollableContent>
      <IAIDroppable data={droppableData} dropLabel="Add Field to Linear View" />
    </Box>
  );
};

export default memo(LinearTabContent);
