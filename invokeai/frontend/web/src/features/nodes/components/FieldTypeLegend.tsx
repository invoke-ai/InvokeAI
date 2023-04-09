import 'reactflow/dist/style.css';
import { Tooltip, Badge, HStack } from '@chakra-ui/react';
import { map } from 'lodash';
import { FIELDS } from '../constants';

export const FieldTypeLegend = () => {
  return (
    <HStack>
      {map(FIELDS, ({ title, description, color }, key) => (
        <Tooltip key={key} label={description}>
          <Badge colorScheme={color} sx={{ userSelect: 'none' }}>
            {title}
          </Badge>
        </Tooltip>
      ))}
    </HStack>
  );
};
