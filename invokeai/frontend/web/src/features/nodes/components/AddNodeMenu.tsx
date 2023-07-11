import { Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { map } from 'lodash-es';
import { forwardRef, useCallback } from 'react';
import 'reactflow/dist/style.css';
import { AnyInvocationType } from 'services/events/types';
import { useBuildInvocation } from '../hooks/useBuildInvocation';
import { nodeAdded, nodesSelector } from '../store/nodesSlice';

type NodeTemplate = {
  label: string;
  value: string;
  description: string;
};

const selector = createSelector(
  nodesSelector,
  (nodes) => {
    const data: NodeTemplate[] = map(nodes.invocationTemplates, (template) => {
      return {
        label: template.title,
        value: template.type,
        description: template.description,
      };
    });

    data.push({
      label: 'Progress Image',
      value: 'progress_image',
      description: 'Displays the progress image in the Node Editor',
    });

    return { data };
  },
  defaultSelectorOptions
);

const AddNodeMenu = () => {
  const dispatch = useAppDispatch();
  const { data } = useAppSelector(selector);

  const buildInvocation = useBuildInvocation();

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
      <IAIMantineSelect
        selectOnBlur={false}
        placeholder="Add Node"
        value={null}
        data={data}
        maxDropdownHeight={400}
        nothingFound="No matching nodes"
        itemComponent={SelectItem}
        filter={(value, item: NodeTemplate) =>
          item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.value.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.description.toLowerCase().includes(value.toLowerCase().trim())
        }
        onChange={handleChange}
        sx={{
          width: '18rem',
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
          <Text>{label}</Text>
          <Text size="xs" color="base.600">
            {description}
          </Text>
        </div>
      </div>
    );
  }
);

SelectItem.displayName = 'SelectItem';

export default AddNodeMenu;
