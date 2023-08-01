import { Box, Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { AddFieldToLinearViewDropData } from 'features/dnd/types';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  isInvocationNode,
} from 'features/nodes/types/types';
import { forEach } from 'lodash-es';
import { memo } from 'react';
import InputFieldRenderer from '../../fields/InputFieldRenderer';
import ScrollableContent from '../ScrollableContent';
import FieldTooltipContent from '../../fields/FieldTooltipContent';
import LinearViewField from '../../fields/LinearViewField';

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

    // const fields = nodes.nodes.filter(isInvocationNode).reduce((acc, node) => {
    //   const nodeTemplate = nodes.nodeTemplates[node.data.type];
    //   if (!nodeTemplate) {
    //     return acc;
    //   }

    //   forEach(node.data.inputs, (input) => {
    //     if (!input.isExposed) {
    //       return;
    //     }

    //     const fieldTemplate = nodeTemplate.inputs[input.name];
    //     if (!fieldTemplate) {
    //       return;
    //     }

    //     acc.push({
    //       nodeData: node.data,
    //       nodeTemplate,
    //       field: input,
    //       fieldTemplate,
    //     });
    //   });

    //   return acc;
    // }, [] as { nodeData: InvocationNodeData; nodeTemplate: InvocationTemplate; field: InputFieldValue; fieldTemplate: InputFieldTemplate }[]);

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
          {fields.map(({ nodeData, nodeTemplate, field, fieldTemplate }) => (
            <LinearViewField
              key={field.id}
              nodeData={nodeData}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
            />
          ))}
        </Flex>
      </ScrollableContent>
      <IAIDroppable data={droppableData} dropLabel="Add Field to Linear View" />
    </Box>
  );
};

export default memo(LinearTabContent);
