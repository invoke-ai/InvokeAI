import { Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { useBuildNodeData } from 'features/nodes/hooks/useBuildNodeData';
import { nodeAdded } from 'features/nodes/store/nodesSlice';
import { map } from 'lodash-es';
import { forwardRef, useCallback } from 'react';
import 'reactflow/dist/style.css';
import { AnyInvocationType } from 'services/events/types';

type NodeTemplate = {
  label: string;
  value: string;
  description: string;
  tags: string[];
};
const filter = (value: string, item: NodeTemplate) => {
  const regex = new RegExp(
    value
      .toLowerCase()
      .trim()
      // strip out regex special characters to avoid errors
      .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
      .split(' ')
      .join('.*'),
    'g'
  );
  return (
    regex.test(item.label.toLowerCase()) ||
    regex.test(item.description.toLowerCase()) ||
    item.tags.some((tag) => regex.test(tag))
  );
};

const selector = createSelector(
  [stateSelector],
  ({ nodes }) => {
    const data: NodeTemplate[] = map(nodes.nodeTemplates, (template) => {
      return {
        label: template.title,
        value: template.type,
        description: template.description,
        tags: template.tags,
      };
    });

    data.push({
      label: 'Progress Image',
      value: 'current_image',
      description: 'Displays the current image in the Node Editor',
      tags: ['progress'],
    });

    data.push({
      label: 'Notes',
      value: 'notes',
      description: 'Add notes about your workflow',
      tags: ['notes'],
    });

    data.sort((a, b) => a.label.localeCompare(b.label));

    return { data };
  },
  defaultSelectorOptions
);

const AddNodeMenu = () => {
  const dispatch = useAppDispatch();
  const { data } = useAppSelector(selector);

  const buildInvocation = useBuildNodeData();

  const toaster = useAppToaster();

  const addNode = useCallback(
    (nodeType: AnyInvocationType) => {
      const invocation = buildInvocation(nodeType);

      if (!invocation) {
        toaster({
          status: 'error',
          title: `Unknown Invocation type ${nodeType}`,
        });
        return;
      }

      dispatch(nodeAdded(invocation));
    },
    [dispatch, buildInvocation, toaster]
  );

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      addNode(v as AnyInvocationType);
    },
    [addNode]
  );

  return (
    <Flex sx={{ gap: 2, alignItems: 'center' }}>
      <IAIMantineSearchableSelect
        selectOnBlur={false}
        placeholder="Add Node"
        value={null}
        data={data}
        maxDropdownHeight={400}
        nothingFound="No matching nodes"
        itemComponent={SelectItem}
        filter={filter}
        onChange={handleChange}
        hoverOnSearchChange={true}
        sx={{
          width: '24rem',
        }}
      />
    </Flex>
  );
};

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  value: string;
  label: string;
  description: string;
}

const SelectItem = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, description, ...others }: ItemProps, ref) => {
    return (
      <div ref={ref} {...others}>
        <div>
          <Text fontWeight={600}>{label}</Text>
          <Text
            size="xs"
            sx={{ color: 'base.600', _dark: { color: 'base.500' } }}
          >
            {description}
          </Text>
        </div>
      </div>
    );
  }
);

SelectItem.displayName = 'SelectItem';

export default AddNodeMenu;
