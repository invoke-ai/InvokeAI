import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AddFieldToLinearViewDropData } from 'features/dnd/types';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  isInvocationNode,
} from 'features/nodes/types/types';
import { forEach } from 'lodash-es';
import { memo } from 'react';
import LinearViewField from '../../fields/LinearViewField';
import ScrollableContent from '../ScrollableContent';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const fields: {
      nodeData: InvocationNodeData;
      nodeTemplate: InvocationTemplate;
      field: InputFieldValue;
      fieldTemplate: InputFieldTemplate;
    }[] = [];
    const { exposedFields } = nodes.workflow;
    nodes.nodes.filter(isInvocationNode).forEach((node) => {
      const nodeTemplate = nodes.nodeTemplates[node.data.type];
      if (!nodeTemplate) {
        return;
      }
      forEach(node.data.inputs, (field) => {
        if (
          !exposedFields.some(
            (f) => f.nodeId === node.id && f.fieldName === field.name
          )
        ) {
          return;
        }
        const fieldTemplate = nodeTemplate.inputs[field.name];
        if (!fieldTemplate) {
          return;
        }
        fields.push({
          nodeData: node.data,
          nodeTemplate,
          field,
          fieldTemplate,
        });
      });
    });

    return {
      fields,
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
            fields.map(({ nodeData, nodeTemplate, field, fieldTemplate }) => (
              <LinearViewField
                key={field.id}
                nodeData={nodeData}
                nodeTemplate={nodeTemplate}
                field={field}
                fieldTemplate={fieldTemplate}
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
